#!/bin/bash

# ---------- 环境变量 ----------
export LD_LIBRARY_PATH=/app/weights/vad/ten-vad/lib/Linux/aarch64:/usr/local/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/cann-8.5.1/tools/aml/lib64:/usr/local/Ascend/cann-8.5.1/tools/aml/lib64/plugin:/usr/local/Ascend/cann-8.5.1/lib64:/usr/local/Ascend/cann-8.5.1/lib64/plugin/opskernel:/usr/local/Ascend/cann-8.5.1/lib64/plugin/nnengine:/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/python3.11.14/lib::/usr/local/lib

# ---------- 信号转发 + 清理 ----------
VLLM_PIDS=()

cleanup() {
    echo "[entrypoint] 收到终止信号，正在停止所有 vLLM 进程..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${VLLM_PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "[entrypoint] 清理完成"
    exit 0
}
trap cleanup SIGTERM SIGINT SIGQUIT

# ==========================================================
#  1. 启动 Qwen3-ASR-1.7B (端口 15002)
# ==========================================================
echo "=== [entrypoint] 启动 Qwen3-ASR-1.7B (端口 15002) ==="
ASCEND_RT_VISIBLE_DEVICES=2 \
  vllm serve "/weights/Qwen3-ASR-1.7B" \
  --served-model-name Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
  --port 15002 &
VLLM_PIDS+=($!)
echo "[entrypoint] 1.7B PID=${VLLM_PIDS[-1]}"

# 等 1.7B 先加载一段时间再启动 0.6B 实例
sleep 150

# ==========================================================
#  2. 启动 6 个 Qwen3-ASR-0.6B (端口 15004-15014, 步进 2)
# ==========================================================
echo "=== [entrypoint] 启动 6 个 Qwen3-ASR-0.6B ==="
for i in $(seq 0 5); do
    PORT=$((15004 + i * 2))
    echo "[entrypoint] Qwen3-ASR-0.6B #$((i+1)) → 端口 $PORT"
    ASCEND_RT_VISIBLE_DEVICES=2 \
      vllm serve "/weights/Qwen3-ASR-0.6B" \
      --served-model-name Qwen3-ASR-0.6B \
      --gpu-memory-utilization 0.08 \
      --max-model-len 4096 \
      --host 0.0.0.0 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64]}' \
      --port $PORT &
    VLLM_PIDS+=($!)
    echo "[entrypoint] 0.6B #$((i+1)) PID=${VLLM_PIDS[-1]}"
    if [ $i -lt 5 ]; then
        sleep 120
    fi
done

# ==========================================================
#  3. 等待所有 vLLM 端口就绪
# ==========================================================
PORTS="15002 15004 15006 15008 15010 15012 15014"
echo "[entrypoint] 等待全部 ${#VLLM_PIDS[@]} 个 vLLM 服务就绪..."

for attempt in $(seq 1 120); do
    ALL_OK=true
    for port in $PORTS; do
        CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 http://127.0.0.1:$port/v1/models 2>/dev/null || echo "000")
        if [ "$CODE" != "200" ]; then
            ALL_OK=false
            break
        fi
    done

    if $ALL_OK; then
        echo "[entrypoint] === 全部 ${#VLLM_PIDS[@]} 个 vLLM 服务已就绪 ==="
        echo "[entrypoint] 1.7B → http://localhost:15002"
        for i in $(seq 0 5); do
            PORT=$((15004 + i * 2))
            echo "[entrypoint] 0.6B #$((i+1)) → http://localhost:$PORT"
        done
        break
    fi

    if [ $((attempt % 12)) -eq 0 ]; then
        echo "[entrypoint] 仍在等待 vLLM 就绪... (${attempt}/120)"
    fi
    sleep 5
done

if ! $ALL_OK; then
    echo "[entrypoint] WARNING: vLLM 服务未全部就绪，继续启动主程序"
fi

# ==========================================================
#  4. 启动主程序（exec 接管进程）
# ==========================================================
echo "=== [entrypoint] 启动主程序 main.py ==="
exec python /app/main.py