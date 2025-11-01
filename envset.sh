#!/bin/bash

# 2. torch, torchvision, torchaudio 삭제
echo "Uninstalling torch, torchvision, and torchaudio..."
sudo pip uninstall torch torchvision torchaudio -y

# 3. 맞는 버전으로 재설치
echo "Installing specific versions of torch, torchvision, and torchaudio..."
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. torchdata, torchtext 삭제
echo "Uninstalling torchdata and torchtext..."
sudo pip uninstall torchdata torchtext -y

# 5. requirements.txt로 다른 의존성 설치
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

# 6. MarkupSafe와 anyio 특정 버전 설치
echo "Installing MarkupSafe and anyio..."
pip install "MarkupSafe>=2.1.1" "anyio<4,>=3.1.0"

# 7. transformer_engine 관련 파일/폴더 삭제 (관리자 권한 필요)
echo "Removing transformer_engine files with sudo..."
sudo rm -rf /usr/local/lib/python3.10/dist-packages/transformer_engine*
sudo rm -f /usr/local/lib/python3.10/dist-packages/transformer_engine_extensions*.so

echo "Script completed successfully."

# export TORCH_HOME=/home/work/Hwang/.cache/torch