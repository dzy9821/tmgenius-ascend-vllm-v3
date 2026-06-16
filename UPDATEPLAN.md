# 改进计划

## 改进一：幻觉后处理规则重写

### 现状

`src/services/asr_service.py` `_filter_hallucination()` 有三套规则：
1. 黑名单关键词匹配（`_HALLUCINATION_BLACKLIST`）
2. "热词：" 前缀检测
3. 所有热词均出现在结果中 → 判定幻觉

### 改为

**删除上述所有规则**，仅保留一条：

> 离线 ASR（1.7B）结果如果以热词列表中**相邻两个热词**开头，返回空字符串。

示例：热词 `警单,警情,张三疯`
- 相邻对：`("警单", "警情")`、`("警情", "张三疯")`
- `"警单,警情创建警单"` → 以 `"警单,警情"` 开头 → 返回 `""`
- `"警情、张三疯"` → 以 `"警情、张三疯"` 开头 → 返回 `""`
- `"创建警单，创建警情"` → 不以相邻对开头 → 保留

### 影响范围

| 文件 | 改动 |
|------|------|
| `src/services/asr_service.py` | 删除 `_HALLUCINATION_BLACKLIST`；重写 `_filter_hallucination()` |

### 风险

- **低风险**：仅影响离线 ASR 后处理（在线 ASR 传 `hotwords=""`，新规则自然跳过）
- 旧规则中的黑名单和"热词："前缀检测被移除，那些幻觉模式不再被拦截。但新规则针对的是最常见的 prompt 泄漏模式（模型直接复述热词列表），命中率更高、误杀率更低
- 如果热词 < 2 个，规则永不触发，行为等价于不过滤

---

## 改进二：在线 ASR VAD 分段（0.5s 断句）

### 现状

在线 ASR 的 `online_buffer` 累积音频，直到**离线 VAD 切段**（PAUSE_THRESHOLD，如 0.8s）才清空。若离线 VAD 迟迟不切段，`online_buffer` 持续增长，送入在线模型（0.6B）的音频越来越长，诱发幻觉。

```
当前流程：
audio → online_buffer 累加 → 每 400ms 触发在线 ASR（发全量 buffer）
                              → 离线 VAD 切段 → online_buffer 清空
```

### 改为

在线 ASR 也受 VAD 控制分段，但用更短的静音阈值（`ONLINE_VAD_PAUSE_MS=500`，即 0.5s）。当 VAD 检测到 0.5s 静音时，在线上下文"切段"——后续在线推理只用切段后的新音频，不再累加旧音频。

**关键约束**：发送给客户端的 Progressive 消息格式不变。同属一个离线 VAD 分段的多个在线子分段，文本需要**累加返回**——后续子分段的结果要拼接前一个子分段的最后一帧完整文本，保证前端看到的 progressive 文本持续增长而不回退。

```
改进后流程：
audio → online_buffer 累加 → 每 400ms 触发在线 ASR（只发 cursor 之后的音频）
                              ↘ 在线结果拼接 accumulated_text 后发给客户端
         ↘ VAD silence ≥ 0.5s → cursor 推进，epoch++，save last_text → accumulated_text
         ↘ 离线 VAD 切段（0.8s）→ online_buffer/cursor/accumulated_text 全部清空
```

示例推演（同一离线段内）：

```
子段1: audio[0...2s] → online ASR 返回 "创建警单"
  → send: Progressive text="创建警单"
  → fs.online_last_text = "创建警单"

0.5s 静音 → online cut:
  → fs.online_accumulated_text = "创建警单，"
  → fs.online_cut_cursor = 当前 online_total
  → epoch++

子段2: audio[cursor...4s] → online ASR 返回 "创建警情"
  → 拼接: "创建警单，" + "创建警情" = "创建警单，创建警情"
  → send: Progressive text="创建警单，创建警情"
```

### 影响范围

| 文件 | 改动 |
|------|------|
| `src/config.py` | 新增 `online_vad_pause_ms: int = 500` |
| `src/services/vad_service.py` | 新增 `silence_duration` 属性，暴露当前静音时长 |
| `src/api/websocket_handler.py` | `FrameState` 新增 `online_cut_cursor`、`online_last_text`、`online_accumulated_text`；`_process_audio_frame` 增加在线 VAD 切段+文本保存；`_maybe_trigger_online` 切片发送；`_do_online_asr` 拼接 accumulated_text；`_do_trigger_offline` 重置全部 |

### 风险

- **中风险**：改变在线 ASR 的输入窗口和文本拼装逻辑
- 用户说话中如有 ≥ 0.5s 的自然停顿，在线 progressive 文本会拼接前一段内容继续增长（不会回退）。**离线 final 结果不受影响**（完整音频仍然送给 1.7B）
- 正在飞行中的在线 ASR 请求会被 epoch 检查拦截丢弃（已有机制）。丢弃前如果拿到了结果，会先保存到 `online_last_text` 供后续拼接
- 若 0.5s 阈值太激进（频繁切断），可调大 `ONLINE_VAD_PAUSE_MS` 环境变量
- 子段间的拼接使用顿号或逗号分隔，可能出现标点不自然的情况，属于已知可接受的折衷
- `bg`/`ed` 时间戳：在线结果使用 `seg_start_abs` 和 `abs_samples`，不受 cursor 影响，时间戳仍然正确

---

## 执行顺序

1. **改进一**（幻觉规则）— 独立、低风险、单文件
2. **改进二**（在线 VAD 分段）— 依赖 VAD bug 修复完成、依赖改进一中 `silence_duration` 属性的新增
