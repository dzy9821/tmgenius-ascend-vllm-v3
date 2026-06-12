"""
RNNoise 风格音频降噪 — 流式处理。

提供两种使用方式：
1. RnnoiseStreamProcessor — 流式逐帧处理，适用于 WebSocket 实时音频管线
2. rnnoise_denoise() — 整段批量处理（兼容旧接口）

当 RNNoise C 库可用时，优先使用 ctypes 调用原生库。
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

# ── 常量（与 C 库 denoise.c 一致）─────────────────────────────────────────
SAMPLE_RATE    = 16000
FRAME_SIZE     = 640   # 40ms @ 16kHz (实际处理帧)
FRAME_SIZE_EXT = 1280  # 80ms window (用于 FFT 分析窗口)
WINDOW_SIZE    = 1280
FREQ_SIZE      = WINDOW_SIZE // 2 + 1  # 641

# ERB 频带边界 — 32 个 bark 尺度频带（bin 索引按 641 点 FFT 重新缩放）
EBAND20MS = np.array([
    0,   3,   5,   8,  11,  13,  16,  20,  24,  28,  32,
    37,  43,  48,  55,  63,  71,  80,  91, 103, 116, 131,
    147, 165, 187, 209, 235, 264, 297, 335, 376, 422, 474, 533
], dtype=np.int32)
NB_BANDS = len(EBAND20MS) - 2  # 32

# Hann 半窗（用于重叠相加分析/合成）
_hann = np.hanning(WINDOW_SIZE)
HALF_WINDOW = _hann[:FRAME_SIZE].astype(np.float32)


# ── 特征提取 ──────────────────────────────────────────────────────────────

def compute_band_energy(x_fft: np.ndarray) -> np.ndarray:
    """从 FFT 频谱计算 bark 频带能量。"""
    bandE = np.zeros(NB_BANDS + 2, dtype=np.float32)
    for i in range(NB_BANDS + 1):
        band_size = EBAND20MS[i + 1] - EBAND20MS[i]
        for j in range(band_size):
            frac = j / band_size
            pwr = x_fft[EBAND20MS[i] + j].real ** 2 + x_fft[EBAND20MS[i] + j].imag ** 2
            bandE[i]     += (1 - frac) * pwr
            bandE[i + 1] += frac * pwr
    bandE[1]       = (bandE[0] + bandE[1]) * 2 / 3
    bandE[NB_BANDS] = (bandE[NB_BANDS] + bandE[NB_BANDS + 1]) * 2 / 3
    return bandE[1:NB_BANDS + 1].astype(np.float32)


def band_gain_to_full(gains: np.ndarray) -> np.ndarray:
    """将 32 频带增益插值到 641 个 FFT bin。"""
    g = np.zeros(FREQ_SIZE, dtype=np.float32)
    for i in range(1, NB_BANDS):
        band_size = EBAND20MS[i + 1] - EBAND20MS[i]
        for j in range(band_size):
            frac = j / band_size
            g[EBAND20MS[i] + j] = (1 - frac) * gains[i - 1] + frac * gains[i]
    g[:EBAND20MS[1]] = gains[0]
    g[EBAND20MS[NB_BANDS]:EBAND20MS[NB_BANDS + 1]] = gains[NB_BANDS - 1]
    return g


# ── 流式降噪处理器 ────────────────────────────────────────────────────────

class RnnoiseStreamProcessor:
    """
    流式 RNNoise 降噪处理器。

    模拟 WebSocket 管线中的实时处理：逐 chunk 喂入 PCM 音频，
    内部按 40ms (640 samples) 帧滑动处理，每帧通过 80ms 分析窗 +
    重叠相加合成输出。已就绪的降噪片段即时返回。

    典型用法（模拟 WebSocket 音频流）::

        processor = RnnoiseStreamProcessor(noise_reduce_db=12.0)

        # 每次收到 Opus 解码后的 PCM chunk
        for pcm_chunk in audio_chunks:
            denoised = processor.process(pcm_chunk)
            if len(denoised) > 0:
                # 送入 VAD 和 Online ASR 缓冲区
                vad.feed(denoised)
                online_asr_buffer.extend(denoised)

        # 流结束
        tail = processor.flush()
        if len(tail) > 0:
            vad.feed(tail)
    """

    def __init__(
        self,
        noise_reduce_db: float = 12.0,
        smoothing: float = 0.95,
        noise_floor: float = 1e-6,
    ):
        self.noise_reduce_db = noise_reduce_db
        self.smoothing = smoothing
        self.noise_floor = noise_floor

        # 分析窗缓存（1280 samples，用于 FFT）
        self._analysis_buf = np.zeros(WINDOW_SIZE, dtype=np.float32)

        # 噪声估计状态
        self._noise_bandE = np.zeros(NB_BANDS, dtype=np.float32)
        self._noise_updates = 0
        self._frame_idx = 0
        self._init_frames = 12  # 前 ~480ms 用于噪声初始化

        # 重叠相加输出缓存（随帧数动态扩展）
        self._output = np.array([], dtype=np.float32)
        self._window_sum = np.array([], dtype=np.float32)

        # 已返回给调用方的样本数
        self._cursor = 0

        # 输入余留（不足一帧的样本暂存于此）
        self._leftover = np.array([], dtype=np.float32)

    # ── 内部：单帧处理 ─────────────────────────────────────────────────

    def _process_one_frame(self, frame_pcm: np.ndarray):
        """处理一个 640-sample 帧：FFT → 噪声估计 → Wiener 增益 → IFFT → 重叠相加。"""
        # 1. 滑动分析窗：左移 640，右补新帧
        self._analysis_buf[:FRAME_SIZE_EXT - FRAME_SIZE] = self._analysis_buf[FRAME_SIZE:]
        self._analysis_buf[FRAME_SIZE_EXT - FRAME_SIZE:] = frame_pcm

        # 2. 加窗 + FFT
        windowed = self._analysis_buf * _hann
        X = np.fft.rfft(windowed)

        # 3. 频带能量
        bandE = compute_band_energy(X)

        # 4. 噪声估计
        if self._frame_idx < self._init_frames:
            self._noise_bandE = self._noise_bandE + bandE
            self._noise_updates += 1
        else:
            self._noise_bandE = (
                self.smoothing * self._noise_bandE
                + (1 - self.smoothing) * np.minimum(self._noise_bandE, bandE)
            )

        noise = self._noise_bandE / max(self._noise_updates, 1)

        # 5. Wiener 增益
        signal_est = np.maximum(bandE - noise, self.noise_floor * noise)
        gain = signal_est / np.maximum(bandE, self.noise_floor)
        alpha = 10 ** (-self.noise_reduce_db / 20)
        gain = np.maximum(gain, alpha)
        gain = np.sqrt(gain)

        # 6. 插值到全频谱 + 应用增益
        full_gain = band_gain_to_full(gain.astype(np.float32))
        Y = X * full_gain

        # 7. IFFT + 加窗 + 重叠相加
        y = np.fft.irfft(Y)
        y_windowed = y * _hann
        win_sq = _hann ** 2

        offset = self._frame_idx * FRAME_SIZE
        needed_len = offset + WINDOW_SIZE
        if len(self._output) < needed_len:
            self._output = np.pad(self._output, (0, needed_len - len(self._output)))
            self._window_sum = np.pad(self._window_sum, (0, needed_len - len(self._window_sum)))

        self._output[offset:offset + WINDOW_SIZE] += y_windowed
        self._window_sum[offset:offset + WINDOW_SIZE] += win_sq

        self._frame_idx += 1

    # ── 公共接口 ───────────────────────────────────────────────────────

    def process(self, pcm_chunk: np.ndarray) -> np.ndarray:
        """
        喂入一段 PCM 音频（任意长度），返回新近就绪的降噪数据。

        参数
        ----------
        pcm_chunk : 1-D float32 numpy array, 16kHz 单声道

        返回
        -------
        denoised : 1-D float32 numpy array（可能为空，表示尚无足够数据）
        """
        chunk = pcm_chunk.astype(np.float32).ravel()

        # 拼接上一轮余留
        if len(self._leftover) > 0:
            chunk = np.concatenate([self._leftover, chunk])
            self._leftover = np.array([], dtype=np.float32)

        # 逐帧处理
        pos = 0
        while pos + FRAME_SIZE <= len(chunk):
            self._process_one_frame(chunk[pos:pos + FRAME_SIZE])
            pos += FRAME_SIZE

        # 保存不足一帧的余留
        if pos < len(chunk):
            self._leftover = chunk[pos:].copy()

        # 返回已就绪的样本
        # 第 k 帧完成后，[k*640, (k+1)*640) 已收到全部重叠贡献，可以输出
        ready = self._frame_idx * FRAME_SIZE
        if ready <= self._cursor:
            return np.array([], dtype=np.float32)

        segment = self._output[self._cursor:ready].copy()
        win_seg = self._window_sum[self._cursor:ready]
        mask = win_seg > self.noise_floor
        segment[mask] /= win_seg[mask]

        self._cursor = ready
        return segment.astype(np.float32)

    def flush(self) -> np.ndarray:
        """
        流结束时调用，返回分析窗内残余的降噪数据。

        返回
        -------
        tail : 1-D float32 numpy array（可能为空）
        """
        # 将输入余留作为最后一帧（不足 640 时零填充）
        if len(self._leftover) > 0:
            padded = np.zeros(FRAME_SIZE, dtype=np.float32)
            padded[:len(self._leftover)] = self._leftover
            self._process_one_frame(padded)
            self._leftover = np.array([], dtype=np.float32)

        ready = len(self._output)
        if ready <= self._cursor:
            return np.array([], dtype=np.float32)

        segment = self._output[self._cursor:ready].copy()
        win_seg = self._window_sum[self._cursor:ready]
        mask = win_seg > self.noise_floor
        segment[mask] /= win_seg[mask]

        self._cursor = ready
        return segment.astype(np.float32)

    @property
    def sample_delay(self) -> int:
        """
        算法固有延迟（采样数）。

        80ms 分析窗，50% 重叠 → 第一帧输出需要累积 1280 个输入样本，
        此后每 640 个输入样本输出 640 个降噪样本。
        延迟 = 1280 - 640 = 640 samples = 40ms @ 16kHz。
        """
        return WINDOW_SIZE - FRAME_SIZE


# ── 批量降噪（兼容旧接口）────────────────────────────────────────────────

def rnnoise_denoise(
    audio: np.ndarray,
    noise_reduce_db: float = 12.0,
    smoothing: float = 0.95,
    noise_floor: float = 1e-6,
) -> np.ndarray:
    """
    基于 bark 频带 Wiener 滤波的降噪（整段批量处理）。

    Parameters
    ----------
    audio : 一维 float32 numpy 数组，16kHz 采样
    noise_reduce_db : 降噪强度 (dB)
    smoothing : 噪声谱平滑系数 (0~1, 越大越保守)
    noise_floor : 避免除零的最小值

    Returns
    -------
    denoised : 一维 float32 numpy 数组
    """
    processor = RnnoiseStreamProcessor(
        noise_reduce_db=noise_reduce_db,
        smoothing=smoothing,
        noise_floor=noise_floor,
    )
    denoised = processor.process(audio)
    tail = processor.flush()
    if len(tail) > 0:
        denoised = np.concatenate([denoised, tail])
    return denoised.astype(np.float32)


# ── 流式示例主程序 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RNNoise 流式音频降噪")
    parser.add_argument("input", type=str, help="输入 WAV 文件路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 WAV 路径（默认: <input>_denoised.wav）")
    parser.add_argument("--noise-reduce-db", type=float, default=12.0,
                        help="降噪强度 (dB), default: 12")
    parser.add_argument("--smoothing", type=float, default=0.95,
                        help="噪声平滑系数, default: 0.95")
    parser.add_argument("--chunk-ms", type=int, default=40,
                        help="模拟流式 chunk 时长 (ms), default: 40")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        return 1

    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_denoised.wav"
    else:
        output_path = Path(args.output)

    print(f"[INFO] 读取: {input_path}")
    audio, sr = sf.read(input_path, dtype="float32")
    print(f"[INFO] 采样率: {sr} Hz, 时长: {len(audio) / sr:.2f}s, 形状: {audio.shape}")

    # 转单声道
    if audio.ndim > 1:
        print("[INFO] 转为单声道...")
        audio = audio.mean(axis=1).astype(np.float32)

    # 重采样到 16kHz（如需要）
    if sr != SAMPLE_RATE:
        print(f"[INFO] 重采样 {sr} -> {SAMPLE_RATE} Hz...")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = audio.astype(np.float32)
        sr = SAMPLE_RATE

    # ── 流式处理（模拟 WebSocket 管线） ──
    chunk_size = int(SAMPLE_RATE * args.chunk_ms / 1000)  # e.g. 640 samples @ 40ms
    print(f"[INFO] 流式降噪 (chunk={args.chunk_ms}ms/{chunk_size}samples, "
          f"strength={args.noise_reduce_db}dB, smoothing={args.smoothing})...")

    processor = RnnoiseStreamProcessor(
        noise_reduce_db=args.noise_reduce_db,
        smoothing=args.smoothing,
    )

    denoised_chunks = []
    total_samples = len(audio)
    for offset in range(0, total_samples, chunk_size):
        chunk = audio[offset:offset + chunk_size]
        out = processor.process(chunk)
        if len(out) > 0:
            denoised_chunks.append(out)

    # 流结束，冲刷残余
    tail = processor.flush()
    if len(tail) > 0:
        denoised_chunks.append(tail)

    denoised = np.concatenate(denoised_chunks) if denoised_chunks else np.array([], dtype=np.float32)
    print(f"[INFO] 输入: {total_samples} samples, 输出: {len(denoised)} samples "
          f"(延迟: {processor.sample_delay} samples / {processor.sample_delay / SAMPLE_RATE * 1000:.0f}ms)")

    # 写回原始采样率
    if sr != SAMPLE_RATE:
        print(f"[INFO] 重采样回 {sr} Hz...")
        import librosa
        denoised = librosa.resample(denoised, orig_sr=SAMPLE_RATE, target_sr=sr)

    # 归一化防止削波
    peak = np.abs(denoised).max()
    if peak > 0.99:
        denoised = denoised * 0.95 / peak

    print(f"[INFO] 写入: {output_path}")
    sf.write(str(output_path), denoised, sr)

    # 统计
    energy_in = np.mean(audio ** 2)
    energy_out = np.mean(denoised ** 2)
    print(f"[INFO] 输入 RMS: {np.sqrt(energy_in):.4f}, 输出 RMS: {np.sqrt(energy_out):.4f}")
    print(f"[INFO] 完成!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
