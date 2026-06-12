import base64
from openai import OpenAI
from qwen_asr import parse_asr_output

# 读取音频并编码为 base64
audio_path = "data/zhangsanfeng.wav"
with open(audio_path, "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# 初始化客户端，base_url 指向本地服务，api_key 可任意填写（本地服务通常无需鉴权）
client = OpenAI(
    base_url="http://192.168.5.100:15004/v1",
    api_key="not-needed"
)


messages = []    
messages.append({
    "role": "user",
    "content": [
        {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_base64}"
            }
        }
    ]
})

response = client.chat.completions.create(
    model="Qwen3-ASR-0.6B",
    messages=messages,
    timeout=300  # 超时时间（秒）
)

# 提取识别结果
content = response.choices[0].message.content
print(content)

# 解析语言和文本
language, text = parse_asr_output(content)
print(language)
print(text)
