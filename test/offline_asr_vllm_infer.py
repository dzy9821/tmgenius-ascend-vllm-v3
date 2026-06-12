import base64
from openai import OpenAI
from qwen_asr import parse_asr_output

# 读取音频并编码为 base64
audio_path = "data/zhangsanfeng.wav"
with open(audio_path, "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# 初始化客户端，base_url 指向本地服务，api_key 可任意填写（本地服务通常无需鉴权）
client = OpenAI(
    base_url="http://192.168.5.100:15002/v1",
    api_key="not-needed"
)

# 构造热词上下文（逗号或竖线分隔）
hotwords = "张三疯,向钱看"


def build_hotword_context(hotwords_str: str) -> str:
    """将热词构建为系统提示词，多个热词以中文顿号分隔。"""
    words = list(dict.fromkeys(
        w.strip() for w in hotwords_str.replace("|", ",").split(",") if w.strip()
    ))
    return f"热词：{'、'.join(words)}" if words else ""


messages = []
hotword_ctx = build_hotword_context(hotwords)
if hotword_ctx:
    messages.append({"role": "system", "content": hotword_ctx})

    
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
    model="Qwen3-ASR-1.7B",
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
