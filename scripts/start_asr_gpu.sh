#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cleanup() {
    echo "正在停止服务..."
    kill $PID_1_7B $PID_0_6B 2>/dev/null
    wait $PID_1_7B $PID_0_6B 2>/dev/null
    echo "已停止"
}
trap cleanup EXIT INT TERM

echo "=== 启动 Qwen3-ASR-1.7B (端口 15002) ==="
FLASHINFER_DISABLE_VERSION_CHECK=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  qwen-asr-serve "$PROJECT_DIR/weights/Qwen3-ASR-1.7B" \
  --served-model-name Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 15002 &
PID_1_7B=$!

echo "=== 启动 Qwen3-ASR-0.6B (端口 15004) ==="
FLASHINFER_DISABLE_VERSION_CHECK=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  qwen-asr-serve "$PROJECT_DIR/weights/Qwen3-ASR-0.6B" \
  --served-model-name Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port 15004 &
PID_0_6B=$!

echo "等待两个服务就绪..."
for i in $(seq 1 120); do
    OK_15002=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15002/v1/models 2>/dev/null || echo "000")
    OK_15004=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15004/v1/models 2>/dev/null || echo "000")
    if [ "$OK_15002" = "200" ] && [ "$OK_15004" = "200" ]; then
        echo "=== 两个服务均已就绪 ==="
        echo "1.7B → http://localhost:15002"
        echo "0.6B → http://localhost:15004"
        wait
        exit 0
    fi
    sleep 5
done

echo "ERROR: 服务启动超时"
exit 1
