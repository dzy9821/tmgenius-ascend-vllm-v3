#!/bin/bash
set -e

echo "=== 启动 ASR 模型服务 ==="
/app/scripts/start_asr.sh &

echo "=== 等待 ASR 服务就绪 ==="
for i in $(seq 1 120); do
    OK_15002=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15002/v1/models 2>/dev/null || echo "000")
    OK_15004=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:15004/v1/models 2>/dev/null || echo "000")
    if [ "$OK_15002" = "200" ] && [ "$OK_15004" = "200" ]; then
        echo "=== ASR 服务均已就绪，启动主应用 ==="
        exec python /app/main.py
    fi
    sleep 5
done

echo "ERROR: ASR 服务启动超时"
exit 1
