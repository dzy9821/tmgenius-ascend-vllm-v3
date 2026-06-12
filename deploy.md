# 部署指南

## 1. 克隆仓库

```bash
git clone <repo-url>
cd tmgenius-ascend-vllm-v3
```

## 2. 创建 Python 虚拟环境

```bash
uv venv --python 3.12
source .venv/bin/activate
```

## 3. 安装依赖

```bash
uv pip install -r wheels/x86_64/requirements.txt
uv pip install wheels/x86_64/*.whl
```

## 4. 编译安装 OpenFst

```bash
# 安装编译工具
apt-get update && apt-get install -y g++ make wget

# 下载并编译 OpenFst
cd /tmp
wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.3.tar.gz
tar xzf openfst-1.8.3.tar.gz
cd openfst-1.8.3
./configure --prefix=/usr/local --enable-grm --enable-static --enable-shared
make -j$(nproc)
make install
```

## 5. 复制动态库

```bash
cp /usr/local/lib/libfst*.so* 3rd-party/openfst1.8.3/lib/
```

## 6. 启动服务

```bash
# 启动 ASR GPU 服务
scripts/start_asr_gpu.sh
python main.py
```

## 7. 测试

```bash
# 运行 WebSocket 客户端端到端测试
python test/e2e_ws_client.py
```


arm
```sh
uv pip install \
    librosa \
    "torchaudio>=2.0.0" \
    "fastapi>=0.115.0" \
    "websockets>=12.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.5.0" \
    "numpy==1.26.4" \
    "httpx>=0.27.0" \
    "prometheus-client>=0.21.0" \
    "soundfile>=0.12.0"
```