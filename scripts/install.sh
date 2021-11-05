# !/bin/sh

# check CUDA
export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

apt update -y && \
    apt install python3 python3-pip git -y && \
    rm -rf /var/lib/apt/lists/*

ln -s /usr/bin/python3 /usr/bin/python

pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip3 install --ignore-installed --upgrade -r requirement.txt