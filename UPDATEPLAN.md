# 任务清单

## 执行顺序

1. **任务二**（SSE 线程安全）— 独立、低风险、方案确定
2. **任务三**（日志完善）— 先统一两套 JSONFormatter，再做修改
3. **任务一**（VAD gap merging）— 多段 gap 逻辑复杂，先测后改

---

## 任务二：SSE 日志 `/api/v1/logs/stream` 线程安全修复

### 问题

`src/core/logging.py` `InMemoryLogHandler.emit()` 两个 bug：

**Bug A：`asyncio.Queue.put_nowait()` 跨线程不安全**

```python
for q in list(self._subscribers):
    try:
        q.put_nowait(msg)      # emit() 可能在非事件循环线程被调用
    except asyncio.QueueFull:
        pass
```

**Bug B：`_subscribers` 集合无锁，与 `subscribe()`/`unsubscribe()` 竞态**

### 修复

**文件：`src/core/logging.py` — `InMemoryLogHandler` 类**

改动：
1. 新增 `_subs_lock = threading.Lock()` 保护订阅者集合
2. 重命名 `_lock` → `_buffer_lock`（语义明确）
3. 新增 `set_loop()` 存储事件循环引用
4. `emit()` 用 `loop.call_soon_threadsafe()` 跨线程投递
5. `subscribe()`/`unsubscribe()` 加锁

```python
import threading

class InMemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 2000):
        super().__init__()
        self._buffer: collections.deque[str] = collections.deque(maxlen=capacity)
        self._buffer_lock = threading.Lock()
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._subs_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._buffer_lock:
                self._buffer.append(msg)

            loop = self._loop
            if loop is not None:
                with self._subs_lock:
                    subs = list(self._subscribers)
                for q in subs:
                    loop.call_soon_threadsafe(self._safe_put, q, msg)
        except Exception:
            self.handleError(record)

    @staticmethod
    def _safe_put(q: asyncio.Queue[str], msg: str) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        with self._subs_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        with self._subs_lock:
            self._subscribers.discard(q)

    def get_recent(self, n: int = 50) -> list[str]:
        with self._buffer_lock:
            items = list(self._buffer)
        return items[-n:]
```

**文件：`main.py` — `setup_logging()` 末尾加:**

```python
log_buffer.set_loop(asyncio.get_running_loop())
```

---

## 任务三：日志完善（DEBUG + 修复 INFO）

### 3.1 统一两套 JSONFormatter（前置条件）

当前存在两套 formatter 且只有 `main.py` 的版本实际生效：

| 文件 | 是否生效 |
|---|---|
| `main.py:29-43` `JSONFormatter` | ✅ 生效 |
| `src/core/logging.py:26-39` `JSONFormatter` | ❌ 死代码（`setup_logging()` 未被调用） |

**步骤：**

1. 删除 `src/core/logging.py` 中的 `JSONFormatter` 类、`setup_logging()` 函数、`trace_id_var`
2. 将 `main.py` 中的 `JSONFormatter` 移到 `src/core/logging.py`，作为一个统一的 formatter
3. 更新 `main.py` 的 import

### 3.2 统一后的 JSONFormatter（在 `src/core/logging.py` 中）

```python
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

class JSONFormatter(logging.Formatter):
    def __init__(self, datefmt: str = "%Y-%m-%dT%H:%M:%S"):
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get("-"),
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)
```

注：不遍历 `record.__dict__`，避免第三方库注入的属性导致日志膨胀。
`extra={"port": 8856}` 的修复方式改为使用 `trace_id_var` 相同的 ContextVar 机制，
或直接将 port 写入 message 字符串。

### 3.3 修复 INFO 日志

#### 3.3.1 trace_id 注入日志

**文件：`src/api/websocket_handler.py` — `handle_websocket`**

解析 traceId 后调用：
```python
from src.core.logging import trace_id_var
trace_id_var.set(trace_id)
```

`asyncio.create_task` 创建的子任务会自动继承 ContextVar（Python 3.7+）。

#### 3.3.2 连接/断开日志加 trace_id

formatter 已自动从 `trace_id_var` 读取，无需额外改动。

### 3.4 新增 DEBUG 级别日志

#### 3.4.1 每条发出的响应

**文件：`src/api/websocket_handler.py:_send_msg`**

在 `json.dumps(data, ...)` 之后、`send_text` 之前：
```python
json_str = json.dumps(data, ensure_ascii=False)
logger.debug("send: sid=%s trace_id=%s raw=%s", sid, trace_id, json_str)
await websocket.send_text(json_str)
```

#### 3.4.2 流程节点

**文件：`src/api/websocket_handler.py`**

| 函数 | 日志 |
|---|---|
| `_process_audio_frame` | `logger.debug("audio: sid=%s samples=%d", sid, len(pcm))` |
| `_do_trigger_offline` | `logger.debug("offline trigger: sid=%s seg_id=%d bg=%d ed=%d is_final=%s")` |
| `_maybe_trigger_online` | `logger.debug("online trigger: sid=%s seg_id=%d samples=%d epoch=%d")` |
| `_do_online_asr` 结果 | `logger.debug("online done: sid=%s seg=%d epoch=%d text=%s")` |
| `_do_offline_asr` 结果 | `logger.debug("offline done: sid=%s seg=%d text=%s")` |
| `_advance_reorder_pointer` | `logger.debug("reorder: seg=%d text=%s next=%d final_seg=%s")` |
| `_result_sender` FINALIZE | `logger.debug("sender done: sid=%s sent_final=%s", sid, sent_final)` |
| `handle_websocket` 关闭 | `logger.debug("ws closed: sid=%s trace_id=%s", sid, trace_id)` |

### 3.5 默认日志级别

保持默认 `INFO`。不改为 DEBUG（高并发下 DEBUG 日志量过大）。

---

## 任务一：VAD 静音间隙合并（gap merging）

### 问题

当前 `_process_frame` 中，`flag=1` 到来时无条件 `_silence_frame_count = 0`。
短语音突增（< 0.5s）应合入静音，且**不进 ASR 音频**。

### 设计修正（针对可行性分析风险 1）

原始方案的问题：每次 `flag=0` 就立即做 gap 判断，无法处理**连续多次短突增**。

修正方案：引入"gap 窗口"——在一个连续的静音窗口内，允许累积多次短语音突增。
静音窗口本身持续累加 `silence_frame_count`，直到**整个窗口的累计静音**达到切分阈值。

核心思路改为：
1. 遇到语音→静音时，将语前静音和这段短语音合并进静音计数
2. **不重置 gap 状态**，静音继续累加
3. 只有当一段语音 ≥ 0.5s（真正有效语音）时，才终结 gap 窗口
4. gap 窗口内所有语音帧暂存在 `_gap_buffer`，最终合并时丢弃

### 状态机

```
状态: NORMAL, GAP_TRACKING

NORMAL ── flag=0 during in_speech ──→ silence 累加，检查 should_cut
NORMAL ── flag=1 ──→ speech 累加，正常的 _segment_frames 追加

flag=1 且前面有 silence 时:
  → 进入 GAP_TRACKING
  → 记录 _gap_base_silence = 前面静音帧数
  → 后续 flag=1 帧进 _gap_buffer（不进 _segment_frames）

GAP_TRACKING ── flag=0 ──→ gap 结束判断:
  gap_speech_dur < 0.5s:
    → silence_count = gap_base_silence + gap_speech (合并)
    → gap_buffer 丢弃
    → 保持 GAP_TRACKING 状态（!!! 关键修正）
    → 后续静音继续累加 silence_count
  gap_speech_dur >= 0.5s:
    → 回退到 NORMAL
    → gap_buffer flush 进 _segment_frames
    → silence 从 0 开始正常计数

NORMAL ── flag=1 且 silence>0 ──→ 再次进入 GAP_TRACKING
```

### 改动位置

**文件：`src/services/vad_service.py`**

**1) `__init__` 新增字段：**

```python
self._gap_active: bool = False        # 当前是否在 gap 窗口内
self._gap_base_silence: int = 0       # gap 前的静音帧数
self._gap_speech: int = 0             # gap 内累计语音帧数
self._gap_buffer: list[np.ndarray] = []  # gap 内语音帧暂存
```

**2) `_process_frame` 完整重写（核心状态机）：**

```python
async def _process_frame(self, frame: np.ndarray) -> Optional[dict]:
    prob, flag_i = await asyncio.to_thread(self._vad.process, frame)
    self._total_samples += self.hop_size
    flag = int(flag_i)

    if flag == 1 and not self._in_speech:
        self._pre_snapshot = list(self._pre_buffer)

    self._pre_buffer.append(frame)
    while len(self._pre_buffer) > self._pad_frames:
        self._pre_buffer.pop(0)

    if flag == 1:
        if not self._in_speech:
            # 新语音段开始
            self._in_speech = True
            self._speech_frame_count = 0
            self._silence_frame_count = 0
            self._segment_frames = []
            self._gap_active = False
            self._gap_base_silence = 0
            self._gap_speech = 0
            self._gap_buffer.clear()
            self._speech_start_sample = (
                self._total_samples - self.hop_size
                - len(self._pre_snapshot) * self.hop_size
            )

        if self._silence_frame_count > 0:
            # 静音后出现语音 → 进入 gap 窗口
            self._gap_active = True
            self._gap_base_silence = self._silence_frame_count
            self._gap_speech = 0
            self._gap_buffer.clear()

        if self._gap_active:
            # gap 窗口内：帧进 _gap_buffer
            self._gap_speech += 1
            self._gap_buffer.append(frame)
        else:
            # 正常语音
            self._speech_frame_count += 1
            self._segment_frames.append(frame)
            self._last_speech_end_sample = self._total_samples

        self._silence_frame_count = 0
    else:
        if self._in_speech:
            if self._gap_active and self._gap_speech > 0:
                # gap 窗口内遇到静音 → 判断是否合并
                gap_speech_dur = self._gap_speech * self.frame_duration
                if gap_speech_dur < MIN_SPEECH_DURATION:
                    # 合并：gap 语音算作静音
                    self._silence_frame_count = (
                        self._gap_base_silence + self._gap_speech
                    )
                    # gap_buffer 丢弃，_gap_active 保持 True 允许后续突增
                    self._gap_speech = 0
                    self._gap_buffer.clear()
                else:
                    # 真实语音：flush gap_buffer
                    self._segment_frames.extend(self._gap_buffer)
                    self._speech_frame_count += self._gap_speech
                    self._last_speech_end_sample = (
                        self._total_samples - self.hop_size
                    )
                    self._gap_active = False
                    self._gap_speech = 0
                    self._gap_buffer.clear()

            self._segment_frames.append(frame)
            self._silence_frame_count += 1

            speech_dur = self._speech_frame_count * self.frame_duration
            pause_dur = self._silence_frame_count * self.frame_duration

            if _should_cut_segment(speech_dur, pause_dur):
                return self._finalize_segment(speech_dur)

    # 强制上限检查：用实际段长而非 speech_frame_count
    if self._in_speech:
        total_frames = (len(self._segment_frames) +
                        len(self._gap_buffer))
        total_dur = total_frames * self.frame_duration
        if total_dur > MAX_SPEECH_DURATION:
            # flush gap_buffer 再切
            if self._gap_active and self._gap_buffer:
                self._segment_frames.extend(self._gap_buffer)
                self._speech_frame_count += self._gap_speech
                self._last_speech_end_sample = (
                    self._total_samples - self.hop_size
                )
                self._gap_active = False
                self._gap_speech = 0
                self._gap_buffer.clear()
            return self._finalize_segment(speech_dur)

    return None
```

**3) `_reset` 新增：**

```python
self._gap_active = False
self._gap_base_silence = 0
self._gap_speech = 0
self._gap_buffer.clear()
```

### 示例推演

#### 例1: 2.0s语→0.2s静→0.3s语→0.5s静（核心场景）

```
2.0s语音: NORMAL, speech=50, in _segment_frames
0.2s静音: silence=5, gap_active=False
0.3s语音: silence>0 → gap_active=True, gap_base=5, gap_speech=7,
           gap_buffer有7帧，不进_segment_frames
0.5s静音第1帧: gap结束判断, 0.28s<0.5s → 合并
           silence=5+7=12, gap_speech=0, gap_active保持True
           gap_buffer丢弃
后续静音帧: silence从12累加，gap_active=True但gap_speech=0
           第11帧: silence=23, 0.92s≥0.9s → 切！

Segment 1 送ASR: [2.0s语音] + [0.2s静音] + [200ms pad]  ≈ 2.4s
                 不含被合并的0.28s语音
_reset()后，剩余0.24s静音 + 后续音频 → Segment 2
```

#### 例3（修正后）: 1.0s语→0.2s静→0.1s语→0.1s静→0.2s语→0.5s静

```
1.0s语: NORMAL, speech=25
0.2s静: silence=5
0.1s语: gap_active=True, gap_base=5, gap_speech=2, gap_buffer有2帧
0.1s静: 0.08s<0.5s → 合并, silence=5+2+1=8, gap保持True
0.2s语: gap_active=True(已在gap中), gap_base=5不变(!!不对,gap_base不改)
         → hmm, gap_base_silence应该保持为最初进入gap时的值

修正: gap_base_silence只在首次进入gap时设置，后续静音合并不更新它。
gap窗口内的"有效静音"通过silence_frame_count持续累加。

0.1s语: gap进入, gap_base=5, gap_speech=2
0.1s静: 合并, silence=5+2+1=8, gap保持True, gap_speech归零
0.2s语: gap仍在, gap_base仍为5, gap_speech=5
0.5s静: 第1帧判断, 0.2s<0.5s → 合并, silence=8+5+13=26(1.04s)≥0.9s → 切！

关键: gap_base_silence不变，silence通过合并持续累加
```

修正实际代码中 gap_speech 归零后，下次进入 flag=1 时：
```python
if self._gap_active:
    self._gap_speech += 1
    self._gap_buffer.append(frame)
```
_gap_active 保持 True，silence_frame_count 在 flag=0 时被归零（因为 flag=1 分支最后 `self._silence_frame_count = 0`）。

**再次修正**：合并后 silence 已被设为 `gap_base + gap_speech`，进入 flag=1 时被归零，这不合理。

需要在 flag=1 时保留合并后的 silence：

```python
if not self._gap_active:
    self._silence_frame_count = 0
```

这样合并后的 silence 值在 flag=1 时不被归零。下一次 flag=0 的静音帧从合并后的值继续累加。

OK 这只在更新计划阶段，细节应放到实现时处理。文档写思路即可。

### 需同步改动的文件

**`test/ten_vad_service.py`**：与生产代码 `_process_frame` 逻辑完全一致，需同步 4 处改动。

**`src/config.py`**：不需要改动。

**`src/api/websocket_handler.py`**：不需功能性改动。注意：`online_buffer` 在 VAD 之前就 append 了音频（line 187），被 gap 丢弃的帧仍会在 online_buffer 中。这是已知行为，offline ASR 收到完整音频、VAD 切分点不同，不影响正确性。

### 自测

任务一完成后，写单测文件覆盖以下场景，运行通过后删除测试文件：

1. 例1: 2.0s语→0.2s静→0.3s语→0.5s静 → 在0.5s静音期间切
2. 例2: 0.2s静→0.6s语→0.5s静 → 不合并，不切
3. 例3: 多次短突增 → 累积合并，在最后切
4. 边界：gap 语音刚好 0.5s → 不合并
5. 正常无 gap 场景 → 行为不变
