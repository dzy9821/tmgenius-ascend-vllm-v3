import asyncio
import base64
import json
import uuid

import websockets

WS_URL = "ws://localhost:8856/ast/v1"
# WS_URL = "ws://182.150.59.81:31848/ast/v1"
#AUDIO_PATH = "data/120报警电话16k.wav"
AUDIO_PATH = "data/zhangsanfeng.wav"
FRAME_SIZE = 1280  # 1280 bytes = 40ms
INTERVAL = 0.04
OUTPUT_FILE = "recognition_results.json"
HOTWORDS =  "警单"  # 握手帧热词，如 "张三疯,向钱看"
#HOTWORDS = ""


def gen_trace_id():
    return str(uuid.uuid4())


def extract_text(result):
    return "".join(
        cw.get("w", "")
        for ws_item in result.get("ws", [])
        for cw in ws_item.get("cw", [])
    )


async def send_audio(ws, audio_path):
    trace_id = gen_trace_id()

    with open(audio_path, "rb") as f:
        chunks = []
        while data := f.read(FRAME_SIZE):
            chunks.append(data)

    for i, data in enumerate(chunks):
        status = 0 if i == 0 else (2 if i == len(chunks) - 1 else 1)

        payload = {"audio": {"audio": base64.b64encode(data).decode()}}
        if status == 0 and HOTWORDS:
            payload["text"] = {"text": HOTWORDS}

        msg = {
            "header": {
                "traceId": trace_id,
                "appId": "123456",
                "bizId": "test_bizid_001",
                "status": status,
            },
            "payload": payload,
        }
        await ws.send(json.dumps(msg))
        print(f"发送成功, status: {status}")

        if status == 2:
            break
        await asyncio.sleep(INTERVAL)


async def receive_result(ws):
    results = []
    accumulated_text = ""
    idle_timeout = 30.0  # 30 秒无新消息则视为识别结束（vLLM 推理需要 10-15s）

    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                print(f"超过 {idle_timeout}s 未收到消息，视为识别结束")
                break

            try:
                resp = json.loads(message)
            except Exception:
                resp = None

            if resp:
                results.append(resp)

                result = resp.get("payload", {}).get("result")
                if result:
                    text = extract_text(result)
                    seg_id = result.get("segId")
                    bg = result.get("bg")
                    ed = result.get("ed")
                    if text:
                        accumulated_text += text
                        print(f"segId={seg_id} [{bg}-{ed}ms]: {text}")

                if resp.get("header", {}).get("status") == 2:
                    print("识别结束（服务端信号）")
                    break

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket 连接已关闭")
    finally:
        # 无论何种情况退出，都保存已收到的结果
        if results:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到 {OUTPUT_FILE}（共 {len(results)} 条消息）")
            print(f"完整文本: {accumulated_text}")

            all_errors = []

            # 验证第一条消息
            first_msg = results[0]
            first_payload = first_msg.get("payload")
            if first_payload is None:
                all_errors.append("第一条消息 payload 为空")
            else:
                first_result = first_payload.get("result")
                if first_result is None:
                    all_errors.append("第一条消息 payload.result 为空")
                else:
                    if first_result.get("msgtype") != "Progressive":
                        all_errors.append(
                            f"第一条消息 msgtype 期望 Progressive，实际 {first_result.get('msgtype')}"
                        )
                    first_text = extract_text(first_result)
                    if not first_result.get("ws") or not first_text:
                        all_errors.append("第一条消息 ws 为空或无文字内容")
                    if first_msg.get("header", {}).get("status") != 0:
                        all_errors.append(
                            f"第一条消息 status 期望 0，实际 {first_msg.get('header', {}).get('status')}"
                        )
                    print(
                        f"\n=== 第一条消息验证通过 ===  status=0, msgtype=Progressive, "
                        f"segId={first_result.get('segId')}, text=\"{first_text[:50]}{'...' if len(first_text) > 50 else ''}\""
                    )

            # 验证最后一条消息的完整性
            last_msg = results[-1]

            # 1. 不能为空 — 必须有 payload
            payload = last_msg.get("payload")
            if payload is None:
                all_errors.append("最后一条消息 payload 为空")
            else:
                result = payload.get("result")
                if result is None:
                    all_errors.append("最后一条消息 payload.result 为空")
                else:
                    # 2. msgtype 必须是 sentence
                    if result.get("msgtype") != "sentence":
                        all_errors.append(
                            f"最后一条消息 msgtype 期望 sentence，实际 {result.get('msgtype')}"
                        )
                    # 3. 必须有文字内容
                    ws = result.get("ws", [])
                    text = extract_text(result)
                    if not ws or not text:
                        all_errors.append("最后一条消息 ws 为空或无文字内容")
                    # 4. status 必须是 2
                    if last_msg.get("header", {}).get("status") != 2:
                        all_errors.append(
                            f"最后一条消息 status 期望 2，实际 {last_msg.get('header', {}).get('status')}"
                        )

            if all_errors:
                print("\n=== 测试失败 ===")
                for e in all_errors:
                    print(f"  FAIL: {e}")
                raise AssertionError("消息验证失败: " + "; ".join(all_errors))
            else:
                print(
                    f"\n=== 测试通过 ===  最后一条消息: status=2, msgtype=sentence, "
                    f"segId={result.get('segId')}, text=\"{text[:50]}{'...' if len(text) > 50 else ''}\""
                )
        else:
            print("未收到任何识别结果")


async def main():
    async with websockets.connect(WS_URL) as ws:
        await asyncio.gather(send_audio(ws, AUDIO_PATH), receive_result(ws))


if __name__ == "__main__":
    asyncio.run(main())
