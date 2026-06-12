#!/usr/bin/env python3
"""
并发测试脚本 - WebSocket ASR 服务

发现并验证两个并发 Bug：
  Bug 1：online_busy 标志竞态（_trigger_offline 强行清零，导致多任务并发）
  Bug 2：Progressive 消息在 sentence 之后到达（无排序保护，语义错误）

用法：
    python test/test_concurrent.py
"""

# ── 必须最先执行：mock C 扩展和 prometheus（在任何 src.* import 之前）────
import sys
import os

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest.mock as _mock

# prometheus_client 会在 import 时注册全局 metrics，多次 import 会冲突；
# ten_vad 依赖 C 扩展（.so），在无 GPU 环境可能不可用。
sys.modules["prometheus_client"] = _mock.MagicMock()

# ── 正常 import ──────────────────────────────────────────────────────────
import asyncio
import base64
import json
import socket
import threading
import time
import traceback
from typing import Optional

import numpy as np
import uvicorn
import websockets.legacy.client
import websockets.exceptions
from fastapi import FastAPI

# 触发 src.metrics 用 mock 的 prometheus_client 创建所有计数器
import src.metrics  # noqa: F401

# 导入并保存对 websocket_handler 模块的引用（后续直接替换模块级名字）
import src.api.websocket_handler as wh
from src.api.websocket_handler import (
    handle_websocket,
    set_itn_service,
    set_rnnoise_service,
)
from src.services.session_manager import SessionManager


# ── Mock 基础设施 ────────────────────────────────────────────────────────

class MockVADSession:
    """
    可配置的 VAD mock，三种工作模式：
      silence      : feed_audio 永不触发；flush 返回 None
      flush_seg    : feed_audio 永不触发；flush 返回一个 segment（end_frame 时触发 offline）
      mid_stream_N : 第 N 次 feed_audio 调用时返回一个 segment（用于 Bug1 测试）
    """

    def __init__(self, sid: str, mode: str = "silence", seg_samples: int = 6400):
        self._sid = sid
        self._mode = mode
        self._seg_samples = seg_samples
        self._call_count = 0
        self._trigger_on: Optional[int] = (
            int(mode.split("_")[-1]) if mode.startswith("mid_stream_") else None
        )

    def _make_segment(self) -> dict:
        return {
            "audio": np.zeros(self._seg_samples, dtype=np.int16),
            "start_sample": 0,
            "end_sample": self._seg_samples,
        }

    async def feed_audio(self, pcm: np.ndarray) -> list:
        self._call_count += 1
        if self._trigger_on is not None and self._call_count == self._trigger_on:
            return [self._make_segment()]
        return []

    def flush(self) -> Optional[dict]:
        if self._mode == "flush_seg":
            return self._make_segment()
        return None

    def close(self) -> None:
        pass


class ControllableASRClient:
    """
    可控延迟的 ASR mock，支持并发计数跟踪。

    tracker 是一个共享 dict：
        {"current": int, "max": int, "total": int}
    asyncio 单线程执行，无需加锁。
    """

    def __init__(
        self,
        text: str = "识别结果",
        delay: float = 0.0,
        tracker: Optional[dict] = None,
    ):
        self.text = text
        self.delay = delay
        self.tracker = tracker

    async def transcribe(self, audio: np.ndarray, hotwords: str = "") -> str:
        if self.tracker is not None:
            self.tracker["current"] = self.tracker.get("current", 0) + 1
            self.tracker["total"] = self.tracker.get("total", 0) + 1
            self.tracker["max"] = max(
                self.tracker.get("max", 0), self.tracker["current"]
            )
        try:
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            return self.text
        finally:
            if self.tracker is not None:
                self.tracker["current"] -= 1


class ServerFixture:
    """在独立 daemon 线程中启动 uvicorn，start() 阻塞直到端口就绪。"""

    def __init__(self, app: FastAPI):
        self._app = app
        self._port = self._free_port()
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self._port}/ast/v1"

    def start(self, timeout: float = 5.0) -> None:
        loop = asyncio.new_event_loop()
        config = uvicorn.Config(
            self._app, host="127.0.0.1", port=self._port, log_level="error"
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                s = socket.socket()
                s.settimeout(0.1)
                s.connect(("127.0.0.1", self._port))
                s.close()
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.05)
        raise RuntimeError(f"Server did not start on port {self._port} within {timeout}s")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=3.0)


def make_app(
    vad_mode: str = "silence",
    online_client: Optional[ControllableASRClient] = None,
    offline_client: Optional[ControllableASRClient] = None,
    max_connections: int = 64,
) -> FastAPI:
    """
    创建测试专用 FastAPI app。
    直接替换 wh 模块级名字，绕过 lru_cache，每次调用都生效。
    """
    _online = online_client or ControllableASRClient("online")
    _offline = offline_client or ControllableASRClient("offline")

    wh.get_online_client = lambda: _online   # type: ignore[attr-defined]
    wh.get_offline_client = lambda: _offline  # type: ignore[attr-defined]
    # 用 lambda 包装，使其行为像类（接受 sid 参数）
    wh.TenVADSession = lambda sid: MockVADSession(sid, mode=vad_mode)  # type: ignore[attr-defined]

    app = FastAPI()
    app.add_api_websocket_route("/ast/v1", handle_websocket)
    app.state.session_manager = SessionManager(max_connections)
    set_itn_service(None)      # 禁用 ITN（offline 结果直接输出）
    set_rnnoise_service(None)  # 禁用 RNNoise

    return app


def make_frame(status: int, samples: int = 6400) -> str:
    """
    生成 JSON 音频帧。
    samples=6400 → 400ms @ 16kHz → 恰好触发 online ASR（online_trigger_ms=400）。
    samples=0    → 空音频，用于结束帧。
    """
    pcm_bytes = np.zeros(samples, dtype=np.int16).tobytes()
    return json.dumps(
        {
            "header": {
                "traceId": "test-trace",
                "appId": "test",
                "bizId": "test",
                "status": status,
            },
            "payload": {
                "audio": {"audio": base64.b64encode(pcm_bytes).decode()}
            },
        }
    )


async def collect_messages(url: str, frames: list, timeout: float = 5.0) -> list:
    """
    连接 WebSocket，依次发送 frames，收集所有消息。
    遇到 status=2 或超时则停止，返回消息列表。
    遇到 1013 拒绝则返回 [{"_rejected": True}]。
    """
    msgs: list = []
    try:
        async with websockets.legacy.client.connect(url) as ws:
            for frame in frames:
                await ws.send(frame)
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    msg = json.loads(raw)
                    msgs.append(msg)
                    if msg.get("header", {}).get("status") == 2:
                        break
                except asyncio.TimeoutError:
                    break
                except websockets.exceptions.ConnectionClosed:
                    break
    except websockets.exceptions.ConnectionClosedError as e:
        if e.rcvd and e.rcvd.code == 1013:
            msgs.append({"_rejected": True, "code": 1013})
    except Exception:
        pass
    return msgs


def msgtype(msg: dict) -> Optional[str]:
    return msg.get("payload", {}).get("result", {}).get("msgtype")


def msg_summary(msgs: list) -> str:
    return str([(msgtype(m), m.get("header", {}).get("status")) for m in msgs])


# ── 测试用例 ─────────────────────────────────────────────────────────────

def test_t1_basic_order():
    """
    T1：单连接基础验证。
    Online ASR 快（10ms），Offline ASR 慢（50ms）。
    Online 先完成 → Progressive(status=0)，Offline 后完成 → sentence(status=2)。
    断言：最后一条消息 msgtype=sentence, status=2。
    """
    online = ControllableASRClient("在线识别", delay=0.01)
    offline = ControllableASRClient("离线识别", delay=0.05)
    app = make_app(vad_mode="flush_seg", online_client=online, offline_client=offline)
    srv = ServerFixture(app)
    srv.start()

    frames = [make_frame(0, 6400), make_frame(2, 0)]
    msgs = asyncio.run(collect_messages(srv.url, frames))
    srv.stop()

    print(f"  消息序列: {msg_summary(msgs)}")
    assert msgs, "未收到任何消息"
    last = msgs[-1]
    assert last["header"]["status"] == 2, f"最后消息 status={last['header']['status']}，期望 2"
    assert msgtype(last) == "sentence", f"最后消息 msgtype={msgtype(last)}，期望 sentence"


def test_t2_bug2_progressive_after_sentence():
    """
    T2（Bug2 修复验证）：Online ASR 慢（150ms），Offline ASR 快（10ms）。

    修复方案：_trigger_offline 递增 online_epoch，_do_online_asr 完成时若 epoch
    不匹配则丢弃结果，确保过期的 Progressive 不会在 sentence 之后到达客户端。

    期望：最后一条消息是 sentence(status=2)；Online 的结果被静默丢弃。
    """
    online = ControllableASRClient("在线识别结果", delay=0.15)
    offline = ControllableASRClient("离线识别结果", delay=0.01)
    app = make_app(vad_mode="flush_seg", online_client=online, offline_client=offline)
    srv = ServerFixture(app)
    srv.start()

    frames = [make_frame(0, 6400), make_frame(2, 0)]
    msgs = asyncio.run(collect_messages(srv.url, frames))
    srv.stop()

    print(f"  消息序列: {msg_summary(msgs)}")
    assert msgs, "未收到任何消息"

    last = msgs[-1]
    last_type = msgtype(last)

    assert last_type == "sentence", (
        f"Bug2 未修复：最后消息是 {last_type}，期望 sentence"
    )
    assert last["header"]["status"] == 2


def test_t3_bug1_online_busy_race():
    """
    T3（Bug1 修复验证）：online_busy epoch 机制正确性。

    修复方案：_trigger_offline 递增 online_epoch，_do_online_asr.finally
    只在 epoch 匹配时才重置 online_busy，防止旧任务清零新任务的标志。

    预期（Bug1 修复后）：
      - 每次 VAD 切句产生一个新 online 任务（共 2 个），最高并发 = 2
      - 不会出现"旧 finally 清零新 busy 导致无限增长"
      - online1 的结果被 epoch 机制丢弃；online2 的结果正常发出
    """
    tracker = {"current": 0, "max": 0, "total": 0}
    online = ControllableASRClient("在线", delay=0.2, tracker=tracker)
    offline = ControllableASRClient("离线", delay=0.0)
    app = make_app(vad_mode="mid_stream_2", online_client=online, offline_client=offline)
    srv = ServerFixture(app)
    srv.start()

    frames = [
        make_frame(0, 6400),  # 握手：触发 online1 (epoch=0)
        make_frame(1, 6400),  # 第2帧：VAD fire → epoch=1, online_busy=False → online2 (epoch=1)
        make_frame(1, 6400),  # 第3帧：online_busy=True（online2 运行中）→ 不触发 online3
        make_frame(2, 0),     # 结束帧
    ]
    asyncio.run(collect_messages(srv.url, frames, timeout=5.0))
    srv.stop()

    print(f"  Online ASR 并发追踪: total={tracker['total']}, max_concurrent={tracker['max']}")

    # 修复后：只有 2 个任务（每段一个），不会无限增长
    assert tracker["total"] == 2, (
        f"Bug1 未修复或测试逻辑错误：期望 2 个 online 任务，实际 {tracker['total']}"
    )
    assert tracker["max"] == 2, (
        f"并发数异常：期望最高并发 2（两段各一个），实际 {tracker['max']}"
    )
    print("  epoch 机制正常：2 个任务并发运行，旧任务结果被丢弃，无额外任务触发")


def test_t4_concurrent_isolation():
    """
    T4：5 个并发连接相互隔离。
    每个连接独立处理，结果不跨连接干扰。
    断言：每个连接都收到 status=2 的最终消息；5 个连接的 sid 互不重复。
    """
    online = ControllableASRClient("online", delay=0.01)
    offline = ControllableASRClient("offline", delay=0.05)
    app = make_app(vad_mode="flush_seg", online_client=online, offline_client=offline)
    srv = ServerFixture(app)
    srv.start()

    N = 5
    frames = [make_frame(0, 6400), make_frame(2, 0)]

    async def run_all() -> list:
        tasks = [collect_messages(srv.url, frames, timeout=5.0) for _ in range(N)]
        return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    srv.stop()

    all_sids: set = set()
    for i, msgs in enumerate(all_results):
        assert msgs, f"客户端 {i} 未收到任何消息"
        last = msgs[-1]
        assert last["header"]["status"] == 2, (
            f"客户端 {i} 最后消息 status={last['header']['status']}"
        )
        for m in msgs:
            sid = m.get("header", {}).get("sid")
            if sid:
                all_sids.add(sid)

    assert len(all_sids) == N, f"期望 {N} 个不同 sid，实际 {len(all_sids)}: {all_sids}"
    print(f"  {N} 个并发连接均收到正确消息，sid 互不重复 ✓")


def test_t5_session_limit():
    """
    T5：超出 max_connections=2 的第三个连接应被 1013 拒绝。

    服务器在 accept() 之后立即调用 acquire()，若 slot 满则发送 close(1013)。
    客户端在 send() 或 recv() 时会收到 ConnectionClosedError(code=1013)。

    注意：必须精确捕获 ConnectionClosedError 并检查 code，不能用宽泛的
    except ConnectionClosed，否则会把 1013 误判为"正常超时"。
    """
    # vad_mode=silence + 320 samples(20ms) → 不触发 online ASR（阈值400ms）
    # 两个"被接受"的连接将持续等待，直到 recv() 超时再关闭
    app = make_app(
        vad_mode="silence",
        online_client=ControllableASRClient("online", delay=2.0),
        offline_client=ControllableASRClient("offline", delay=2.0),
        max_connections=2,
    )
    srv = ServerFixture(app)
    srv.start()

    async def connect_one(connect_delay: float) -> dict:
        """
        连接后发一个不足以触发 ASR 的握手帧，等待 recv()。
        - 若 recv() 超时 → 连接被接受（slot 有空位）
        - 若 recv() / send() 收到 1013 → 连接被拒绝
        """
        await asyncio.sleep(connect_delay)
        try:
            async with websockets.legacy.client.connect(srv.url) as ws:
                # send() 可能因服务器先发 close(1013) 而抛异常
                try:
                    await ws.send(make_frame(0, 320))
                except websockets.exceptions.ConnectionClosedError as e:
                    if e.rcvd and e.rcvd.code == 1013:
                        return {"rejected": True, "code": 1013}
                    return {"rejected": False, "send_err": str(e)}

                # recv() 超时 → 正常占用；收到 1013 → 被拒绝
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                    return {"rejected": False}          # 真的收到消息（不期望）
                except asyncio.TimeoutError:
                    return {"rejected": False}          # slot 有空位，一直等待中
                except websockets.exceptions.ConnectionClosedError as e:
                    if e.rcvd and e.rcvd.code == 1013:
                        return {"rejected": True, "code": 1013}
                    return {"rejected": False, "recv_err": str(e)}

        except websockets.exceptions.ConnectionClosedError as e:
            # connect() 上下文退出时发现 1013
            if e.rcvd and e.rcvd.code == 1013:
                return {"rejected": True, "code": 1013}
            return {"rejected": False, "ctx_err": str(e)}
        except Exception as e:
            return {"rejected": False, "other_err": str(e)}

    async def run():
        return await asyncio.gather(
            connect_one(0.00),   # 第1个：应接受
            connect_one(0.02),   # 第2个：应接受
            connect_one(0.10),   # 第3个（稍晚，确保前两个已 acquire slot）：应被 1013 拒绝
        )

    results = asyncio.run(run())
    srv.stop()

    rejected = [r for r in results if r.get("rejected")]
    print(f"  连接结果: {results}")
    print(f"  被 1013 拒绝数: {len(rejected)}")

    assert len(rejected) == 1, (
        f"期望 1 个连接被 1013 拒绝，实际 {len(rejected)}: {results}"
    )
    assert rejected[0]["code"] == 1013


def test_t6_regression_fix():
    """
    T6（Bug2 修复回归测试）：与 T2 相同条件，独立再次验证 sentence 最后到达。
    两个测试均通过才能确认修复在不同运行时序下均有效。
    """
    online = ControllableASRClient("在线识别结果", delay=0.15)
    offline = ControllableASRClient("离线识别结果", delay=0.01)
    app = make_app(vad_mode="flush_seg", online_client=online, offline_client=offline)
    srv = ServerFixture(app)
    srv.start()

    frames = [make_frame(0, 6400), make_frame(2, 0)]
    msgs = asyncio.run(collect_messages(srv.url, frames))
    srv.stop()

    print(f"  消息序列: {msg_summary(msgs)}")
    assert msgs, "未收到任何消息"

    last = msgs[-1]
    last_type = msgtype(last)

    # 修复验收断言（当前失败，修复后通过）
    assert last_type == "sentence", (
        f"Bug2 未修复：最后消息是 {last_type}，期望 sentence"
    )
    assert last["header"]["status"] == 2


# ── 测试运行器 ───────────────────────────────────────────────────────────

_PASS = "✓ PASS"
_FAIL = "✗ FAIL"
_BUG  = "⚠ BUG"

_TESTS = [
    ("T1", "单连接基础消息顺序",                      test_t1_basic_order,                  False),
    ("T2", "Bug2 修复验证 - online epoch 丢弃过期结果", test_t2_bug2_progressive_after_sentence, False),
    ("T3", "Bug1 修复验证 - online_busy epoch 机制",   test_t3_bug1_online_busy_race,         False),
    ("T4", "5 个并发连接相互隔离",                     test_t4_concurrent_isolation,          False),
    ("T5", "Session 连接数限制（max=2）",              test_t5_session_limit,                 False),
    ("T6", "Bug2 修复验收（与 T2 相同条件再验）",       test_t6_regression_fix,                False),
]


def main() -> None:
    print("=" * 65)
    print("WebSocket ASR 服务并发测试")
    print("=" * 65)

    summary = []
    for tid, name, fn, is_bug_test in _TESTS:
        print(f"\n{'─' * 65}")
        print(f"{tid}: {name}")
        print("─" * 65)
        passed = False
        try:
            fn()
            passed = True
        except AssertionError as e:
            print(f"  AssertionError: {e}")
        except Exception:
            traceback.print_exc()

        if passed:
            mark = _PASS
        elif is_bug_test:
            mark = _BUG + " (预期，Bug 未修复)"
        else:
            mark = _FAIL

        print(f"  → {mark}: {tid} {name}")
        summary.append((tid, name, passed, is_bug_test))

    print(f"\n{'=' * 65}")
    print("汇总")
    print("=" * 65)
    for tid, name, passed, is_bug_test in summary:
        if passed:
            label = _PASS
        elif is_bug_test:
            label = _BUG + " (预期，修复后重验)"
        else:
            label = _FAIL
        print(f"  {label}: {tid} {name}")

    normal_fail = [t for t in summary if not t[3] and not t[2]]
    # Bug 检测测试（T2/T3）：PASS 表示 Bug 已确认；FAIL 表示本次未触发
    bug_detect_pass = [t for t in summary if t[3] and t[0] in ("T2", "T3") and t[2]]
    bug_detect_fail = [t for t in summary if t[3] and t[0] in ("T2", "T3") and not t[2]]
    # 修复验收测试（T6）：FAIL 表示 Bug 未修复（当前预期）
    t6_fixed = next((t[2] for t in summary if t[0] == "T6"), False)

    print()
    if not normal_fail:
        bugs_str = (
            f"Bug1/Bug2 已确认" if bug_detect_pass else
            f"部分 Bug 本次未触发（时序敏感）: {[t[0] for t in bug_detect_fail]}"
        )
        fix_str = "T6 修复验收通过 ✓" if t6_fixed else "T6 等待 Bug2 修复后再验"
        print(f"结论：T1/T4/T5 通过；{bugs_str}；{fix_str}。")
    else:
        print(f"结论：正常测试失败 {[t[0] for t in normal_fail]}，请检查输出。")


if __name__ == "__main__":
    main()
