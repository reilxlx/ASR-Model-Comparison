使用以下仓库进行代码运行：
```
git clone --recurse-submodules https://github.com/MoonshotAI/Kimi-Audio.git
```

总交易时间：
```
total processing time (excluding model loading): 203.12 seconds
```

显存占用：
```
Every 2.0s: nvidia-smi                                                          ubuntu22: Mon Apr 28 12:51:34 2025

Mon Apr 28 12:51:35 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.78                 Driver Version: 550.78         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:00:08.0 Off |                  Off |
| 50%   50C    P2            244W /  450W |   30436MiB /  49140MiB |     60%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A       730      G   /usr/lib/xorg/Xorg                              4MiB |
|    0   N/A  N/A      4910      C   python                                      30418MiB |
+-----------------------------------------------------------------------------------------+
```