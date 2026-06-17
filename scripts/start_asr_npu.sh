#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
    echo "正在停止服务..."
    kill $PID_1_7B $PID_0_6B $PID_0_6B_2 2>/dev/null
    wait $PID_1_7B $PID_0_6B $PID_0_6B_2 2>/dev/null
    echo "已停止"
}
trap cleanup EXIT INT TERM

echo "=== 启动 Qwen3-ASR-1.7B (端口 15002, GPU 0.30) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-1.7B" \
  --served-model-name Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.30 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
  --port 15002 &
PID_1_7B=$!

sleep 210

echo "=== 启动 Qwen3-ASR-0.6B #1 (端口 15004, GPU 0.28) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-0.6B" \
  --served-model-name Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.28 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
  --port 15004 &
PID_0_6B=$!

sleep 180

echo "=== 启动 Qwen3-ASR-0.6B #2 (端口 15006, GPU 0.28) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-0.6B" \
  --served-model-name Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.28 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
  --port 15006 &
PID_0_6B_2=$!

echo "等待三个服务就绪..."
for i in $(seq 1 120); do
    OK_15002=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15002/v1/models 2>/dev/null || echo "000")
    OK_15004=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15004/v1/models 2>/dev/null || echo "000")
    OK_15006=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15006/v1/models 2>/dev/null || echo "000")
    if [ "$OK_15002" = "200" ] && [ "$OK_15004" = "200" ] && [ "$OK_15006" = "200" ]; then
        echo "=== 三个服务均已就绪 ==="
        echo "1.7B     → http://localhost:15002"
        echo "0.6B #1  → http://localhost:15004"
        echo "0.6B #2  → http://localhost:15006"
        wait
        exit 0
    fi
    sleep 5
done

echo "ERROR: 服务启动超时"
exit 1