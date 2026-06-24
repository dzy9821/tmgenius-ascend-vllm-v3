#!/bin/bash
# 注意：不使用 set -e
# 此脚本通常作为后台进程运行（从 entrypoint.sh 调用），
# set -e 会导致 curl 健康检查返回非零时脚本静默退出

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PIDS=()

cleanup() {
    echo "正在停止服务..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "已停止"
}
trap cleanup EXIT INT TERM

echo "=== 启动 Qwen3-ASR-1.7B (端口 15002) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-1.7B" \
  --served-model-name Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
  --port 15002 &
PIDS+=($!)

sleep 150

echo "=== 启动 6 个 Qwen3-ASR-0.6B (端口 15004-15014, 步进 2) ==="
for i in $(seq 0 5); do
    PORT=$((15004 + i * 2))
    echo "  Qwen3-ASR-0.6B #$((i+1)) → 端口 $PORT"
    ASCEND_RT_VISIBLE_DEVICES=2 \
      vllm serve "/weights/Qwen3-ASR-0.6B" \
      --served-model-name Qwen3-ASR-0.6B \
      --gpu-memory-utilization 0.08 \
      --max-model-len 4096 \
      --host 0.0.0.0 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
      --port $PORT &
    PIDS+=($!)
    if [ $i -lt 5 ]; then
        sleep 120
    fi
done

echo "等待全部 ${#PIDS[@]} 个服务就绪..."
for t in $(seq 1 120); do
    OK_15002=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15002/v1/models 2>/dev/null || echo "000")
    OK_15004=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15004/v1/models 2>/dev/null || echo "000")
    OK_15006=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15006/v1/models 2>/dev/null || echo "000")
    OK_15008=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15008/v1/models 2>/dev/null || echo "000")
    OK_15010=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15010/v1/models 2>/dev/null || echo "000")
    OK_15012=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15012/v1/models 2>/dev/null || echo "000")
    OK_15014=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15014/v1/models 2>/dev/null || echo "000")
    if [ "$OK_15002" = "200" ] && [ "$OK_15004" = "200" ] && [ "$OK_15006" = "200" ] && [ "$OK_15008" = "200" ] && [ "$OK_15010" = "200" ] && [ "$OK_15012" = "200" ] && [ "$OK_15014" = "200" ]; then
        echo "=== 全部 ${#PIDS[@]} 个服务已就绪 ==="
        echo "1.7B → http://localhost:15002"
        for i in $(seq 0 5); do
            PORT=$((15004 + i * 2))
            echo "0.6B #$((i+1)) → http://localhost:$PORT"
        done
        wait
        exit 0
    fi
    sleep 5
done

echo "ERROR: 服务启动超时"
exit 1
