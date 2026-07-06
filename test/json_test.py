import asyncio
import websockets
import base64
import json
import uuid
import os
from time import time
from datetime import datetime



# WS_URL = "ws://10.148.229.215:8180/pms/ai/agent/v1/ast"
WS_URL = "ws://182.150.59.81:31848/ast/v1"
# WS_URL = "ws://148.148.52.26:10095"
AUDIO_PATH = "/home/ubuntu/project/tmgenius-ascend-vllm-v3/data/120报警电话16k.wav" # 音频文件路径
FRAME_SIZE = 1280
INTERVAL = 0.04
OUTPUT_FILE = "recognition_results.json"  # 输出文件路径

have_msg = False




def gen_trace_id():
    return str(uuid.uuid4())

async def send_audio(ws, audio_path):
    trace_id = gen_trace_id()
    biz_id = "test_bizid_001"
    app_id = "123456"
    status = 0

    with open(audio_path, "rb") as f:
        while True:
            data = f.read(FRAME_SIZE)
            if not data:
                break
            current_time = datetime.now()
            print(current_time)

            # 检查是否为最后一块数据
            current_pos = f.tell()
            f.seek(0, 2)  # 移动到文件末尾
            file_size = f.tell()
            f.seek(current_pos)  # 恢复到当前位置
            
            # 如果当前位置加上当前数据块大小达到或超过文件大小，说明这是最后一块数据
            is_last_chunk = current_pos + len(data) >= file_size

            if is_last_chunk:
                status = 2  # 设置为结束状态
            
            audio_b64 = base64.b64encode(data).decode()
            
            # 构建payload，只在第一块数据时包含text
            payload = {
                "audio": {
                    "audio": audio_b64
                }
            }
            
            # 只在第一块数据时添加热词
            # if status == 0:
            #     payload["text"] = {
            #         "text": "张三疯|向钱看"
            #     }
            
            msg = {
                "header": {
                    "traceId": trace_id,
                    "appId": app_id,
                    "bizId": biz_id,
                    "status": status,
                    "resIdList": []
                },
                "parameter": {
                    "engine": {
                        "wdec_param_LanguageTypeChoice": "5"
                    }
                },
                "payload": payload
            }
            # print(msg)
            await ws.send(json.dumps(msg))
            print(f"发送成功, status: {status}")
        
            
            # 如果是最后一块数据，发送后立即退出循环
            if is_last_chunk or status == 2:
                break
                
            await asyncio.sleep(INTERVAL)
            status = 1  # 仅第一帧 status=0，后续为1

async def receive_result(ws):

    results = []
    sequence_number = 1
    accumulated_text = ""  # 累积的文本内容
    role = "(角色未知)"
    
    async for message in ws:
       
        try:
            resp = json.loads(message)
            
            # 添加序号到结果中
            result_with_sequence = {
                "sequence": sequence_number,
                "timestamp": asyncio.get_event_loop().time(),
                "data": resp
            }
            
            results.append(result_with_sequence)
            
            # 解析识别结果
            if "payload" in resp and "result" in resp["payload"]:
                result = resp["payload"]["result"]
                bg = result.get("bg")
                ed = result.get("ed")
                msgtype = result.get("msgtype", "")
               
                
                # 提取当前帧的文本内容
                current_text = ""
                if "ws" in result:
                    for ws_item in result["ws"]:
                        if "cw" in ws_item:
                            for cw_item in ws_item["cw"]:
                                if "w" in cw_item:
                                    rl = cw_item.get("rl", "")
                                    if rl!=0:
                                        role = f"(角色{rl})"
                                    current_text += (cw_item["w"] + role)
                                    
                                  
                current_text += f"[{bg}-{ed}]"
                # 根据消息类型处理累积文本
                if msgtype == "progressive":
                    # 中间状态：显示当前累积的文本
                    status_label = "【中间状态】"
                    display_text = accumulated_text + current_text
                elif msgtype == "sentence":
                    # 最终状态：将当前文本添加到累积文本中
                    accumulated_text += current_text
                    status_label = "【最终状态】"
                    display_text = accumulated_text
                else:
                    status_label = "【未知状态】"
                    display_text = accumulated_text + current_text
                
                # 打印累积的文本内容
                if current_text or accumulated_text:
		    
                    
                   current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 格式化当前时间
                   print(f"收到结果序号#{sequence_number} {status_label}: {display_text} (时间: {current_time})")
                else:
                    print(f"收到结果序号#{sequence_number}: 无文本内容")
            
            # 可以根据 resp["header"]["status"] == 2 判断是否结束
            if resp.get("header", {}).get("status") == 2:
                print("识别结束")
                
                # 保存所有结果到JSON文件
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"所有识别结果已保存到 {OUTPUT_FILE}")
                print(f"最终累积文本: {accumulated_text}")
                break
                
            sequence_number += 1
            
        except Exception as e:
            print("解析服务端消息失败:", e)

async def main():
    async with websockets.connect(WS_URL) as ws:
        send_task = asyncio.create_task(send_audio(ws, AUDIO_PATH))
        recv_task = asyncio.create_task(receive_result(ws))
        await asyncio.gather(send_task, recv_task)

if __name__ == "__main__":
    asyncio.run(main())