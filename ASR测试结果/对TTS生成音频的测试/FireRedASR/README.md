# FireRedASR 使用文档

## 基础环境

- 操作系统版本： RedHat 7.8
- CUDA Version：12.2
- Driver Version：535.183.01

## 安装文件

- NVIDIA-Linux-x86_64-460.106.00.run
- cuda_11.2.0_460.27.04_linux.run
- Miniconda3-latest-Linux-x86_64.sh

## 操作步骤

### 1. 安装NVIDIA驱动
https://developer.nvidia.com/cuda-toolkit-archive

安装NVIDIA-Linux-x86_64-460.106.00.run

参考手册：https://shuyeidc.com/wp/207213.html，注意`lsmod | grep nouveau`和`init 3`命令的使用

### 2. 安装CUDA

安装cuda_11.2.0_460.27.04_linux.run

配置环境变量：
```bash
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/.bashrc
```

### 3. 安装Miniconda

运行Miniconda3-latest-Linux-x86_64.sh进行安装

## 模型测试FireRedASR

克隆代码并设置环境：
```bash
git clone https://github.com/FireRedTeam/FireRedASR.git
conda create --name fireredasr python=3.10
pip install -r requirements.txt

cd examples
bash inference_fireredasr_aed.sh
```

```
如代码报错，可参考 https://github.com/FireRedTeam/FireRedASR/issues/20 解决
```