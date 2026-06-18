#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 2x 1.7B (离线) + 5x 0.6B (在线)，共 7 个 vLLM 实例
# 内存: 1.7B=0.20, 0.6B=0.09, 总计 0.85
# 图编译: 1,2,4,8,16,32 (去掉 64)
CUDAGRAPH_SIZES='[1,2,4,8,16,32]'

PIDS=()

cleanup() {
    echo "正在停止所有服务..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "已停止"
}
trap cleanup EXIT INT TERM

# ── 离线 1.7B #1 (端口 15002) ──
echo "=== 启动 Qwen3-ASR-1.7B #1 (端口 15002, mem 0.20) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-1.7B" \
  --served-model-name Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.20 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$CUDAGRAPH_SIZES}" \
  --port 15002 &
PIDS+=($!)
sleep 180

# ── 离线 1.7B #2 (端口 15004) ──
echo "=== 启动 Qwen3-ASR-1.7B #2 (端口 15004, mem 0.20) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-1.7B" \
  --served-model-name Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.20 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$CUDAGRAPH_SIZES}" \
  --port 15004 &
PIDS+=($!)
sleep 180

# ── 在线 0.6B #1~#5 (端口 15006, 15008, 15010, 15012, 15014) ──
for i in 1 2 3 4 5; do
    port=$((15004 + i * 2))  # 15006, 15008, 15010, 15012, 15014
    echo "=== 启动 Qwen3-ASR-0.6B #${i} (端口 ${port}, mem 0.09) ==="
    ASCEND_RT_VISIBLE_DEVICES=2 \
      vllm serve "/weights/Qwen3-ASR-0.6B" \
      --served-model-name Qwen3-ASR-0.6B \
      --gpu-memory-utilization 0.09 \
      --max-model-len 4096 \
      --host 0.0.0.0 \
      --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$CUDAGRAPH_SIZES}" \
      --port $port &
    PIDS+=($!)
    if [ $i -eq 1 ]; then
        sleep 120  # 第一个 0.6B 等模型加载
    else
        sleep 120   # 后续的共享已缓存的模型权重，更快
    fi
done

echo "等待全部 7 个服务就绪..."
for i in $(seq 1 120); do
    ALL_OK=1
    for port in 15002 15004 15006 15008 15010 15012 15014; do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" 2>/dev/null || echo "000")
        if [ "$code" != "200" ]; then
            ALL_OK=0
            break
        fi
    done
    if [ "$ALL_OK" = "1" ]; then
        echo "=== 全部 7 个服务均已就绪 ==="
        echo "离线 1.7B #1  → http://localhost:15002"
        echo "离线 1.7B #2  → http://localhost:15004"
        echo "在线 0.6B #1  → http://localhost:15006"
        echo "在线 0.6B #2  → http://localhost:15008"
        echo "在线 0.6B #3  → http://localhost:15010"
        echo "在线 0.6B #4  → http://localhost:15012"
        echo "在线 0.6B #5  → http://localhost:15014"
        wait
        exit 0
    fi
    sleep 5
done

echo "ERROR: 服务启动超时"
exit 1
