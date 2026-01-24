# 通用式强化学习

## 环境的配置

    conda create -n general_RL python=3.11

    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

    pip install gymnasium==0.28.1
    pip install gym==0.26.2
    pip install AutoROM==0.4.2
    pip install "gymnasium[atari,accept-rom-license]"
    pip install "autorom[accept-rom-license]"
    pip install grpcio==1.76.0
    pip install gym-notices==0.1.0

    pip install Pillow matplotlib
    pip install opencv-python
    pip install imageio
    pip install imageio-ffmpeg
    pip install moviepy
    pip install psutil
    pip install tqdm
    pip install pandas
    pip install requests
    pip install ale-py==0.8.1
    pip install protobuf
    pip install tensorboard

    sudo apt-get update
    sudo apt-get install swig
    pip install box2d-py
    pip install gymnasium[box2d]
    conda install -c conda-forge libstdcxx-ng



## 关于显存使用的查看

    watch -n 2 nvidia-smi



