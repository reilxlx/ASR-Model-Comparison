# ASR模型性能对比评测

## 项目概述

本项目旨在对多种自动语音识别(ASR)模型进行全面评测和比较，包括开源和商业模型。通过使用相同的测试音频数据，对不同ASR系统的准确性、速度和资源消耗等方面进行客观对比。

## 项目结构

```
ASR-Model-Comparison/
├── 测试音频/                     # 用于测试的音频文件
│   ├── TTS生成的音频/            # 使用TTS技术生成的测试音频
│   │   └── Fish-Speech-1.5/     # 使用Fish-Speech-1.5模型生成的音频
│   ├── 真人录制的音频/           # 真人录制的测试音频
│   └── 文本.txt                 # 测试音频对应的文本内容
└── ASR测试结果/                  # 各ASR模型的测试结果
    ├── 对真人录制音频的测试/      # 使用真人音频的测试结果
    │   ├── FireRedASR/         # FireRedASR模型的测试结果
    │   ├── 百度ASR算法/         # 百度ASR的测试结果
    │   ├── 阿里ASR算法/         # 阿里ASR的测试结果
    │   └── 结果分析.md          # 真人音频测试结果的综合分析
    └── 对TTS生成音频的测试/       # 使用TTS生成音频的测试结果
        ├── FunASR-GPU版本/      # FunASR GPU版本的测试结果
        ├── FunASR-CPU版本/      # FunASR CPU版本的测试结果
        ├── Dolphin/            # Dolphin模型的测试结果
        ├── FireRedASR/         # FireRedASR模型的测试结果
        ├── SenseVoice/         # SenseVoice模型的测试结果
        └── Kimi-Audio/         # Kimi-Audio模型的测试结果
```

## 测试数据集

### 基本信息

- **真人录制音频数据集**：
  - 音频文件数量: 171个
  - 总时长：32分18秒 (1937.88秒)
  - 平均时长：11.33秒/文件

- **TTS生成音频数据集**：
  - 音频文件数量：171
  - 总时长（秒）：1633.58
  - 总时长（hh:mm:ss）：00:27:14
  - 平均时长（秒）：9.55
  - 平均时长（hh:mm:ss）：00:00:10

### 文本内容

测试数据集包含171个文本片段，涵盖多种场景和领域的内容，详见`测试音频/文本.txt`文件。

## 模型测试结果

### 对真人录制音频的测试结果

| ASR模型 | 平均字符错误率(CER) | 处理总时间(秒) | 实时因子(RTF) | 成功处理文件数 | 错误文件数 | 平均处理时间/文件(秒) |
| --- | --- | --- | --- | --- | --- | --- |
| FireRedASR | 2.30% | 289.01 | 0.15 | 171 | 0 | 1.69 |
| 阿里 | 2.93% | 447 | 0.23 | 171 | 0 | 2.61 |
| 百度 | 8.57% | 1974.18 | 1.02 | 171 | 0 | 11.54 |

### 对TTS生成音频的测试结果

#### 处理速度和准确性

| ASR模型 | 文件处理成功率 | 平均字符错误率(CER) | 总处理时间(秒) | 平均处理时间/文件(秒) | 显存占用 |
| --- | --- | --- | --- | --- | --- |
| FireRedASR | 100% (171/171) | 0.93% | - | - | - |
| Kimi-Audio | 100% (171/171) | 1.25% | 203.12 | 1.19 | ~30GB (RTX 4090) |
| SenseVoice | 100% (171/171) | 2.23% | 27.51 | 0.16 | ~1.5GB (RTX 4090) |
| Dolphin | 100% (171/171) | 2.92% | 101/52* | 0.59/0.30* | ~2.3GB (RTX 3080/4090*) |
| FunASR-GPU版本 | 100% (171/171) | 3.98% | 85.80 | 0.50 | ~3GB (RTX 3080) |
| FunASR-CPU版本 | 100% (171/171) | 4.02% | 115.11 | 0.67 | N/A (CPU only) |

*注: Dolphin模型分别在RTX 3080(101秒)和RTX 4090(52秒)上进行了测试

## 详细测试结果与分析

### FunASR (GPU版本)

- **环境**: NVIDIA GeForce RTX 3080
- **显存占用**: 约3GB
- **总处理时间**: 85.80秒(171个文件)
- **平均处理时间**: 0.50秒/文件
- **部署方法**:
  ```bash
  # 拉取Docker镜像
  sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-gpu-0.2.0
  
  # 创建模型目录并启动容器
  mkdir -p ./funasr-runtime-resources/models
  sudo docker run --gpus=all -p 10098:10095 -it --privileged=true \
    -v $PWD/funasr-runtime-resources/models:/workspace/models \
    registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-gpu-0.2.0
  ```

### FunASR (CPU版本)

- **环境**: CPU运行
- **总处理时间**: 115.11秒(171个文件)
- **平均处理时间**: 0.67秒/文件
- **部署方法**:
  ```bash
  # 拉取Docker镜像
  sudo docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.6
  
  # 创建模型目录并启动容器
  mkdir -p ./funasr-runtime-resources/models
  sudo docker run -p 10095:10095 -it --privileged=true \
    -v $PWD/funasr-runtime-resources/models:/workspace/models \
    registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.6
  ```

- **压力测试结果**:
  | 并发级别 | 平均响应时间(秒) | 成功率 |
  | --- | --- | --- |
  | 1 | 2.405 | 100% |
  | 5 | 2.681 | 100% |
  | 10 | 3.790 | 100% |
  | 15 | 4.666 | 100% |
  | 20 | 6.350 | 100% |
  | 25 | 7.872 | 100% |
  | 30 | 9.381 | 100% |

### Dolphin

- **环境**: NVIDIA GeForce RTX 4090/RTX 3080
- **显存占用**: 约2.3GB
- **总处理时间**: 
  - RTX 4090: 52秒
  - RTX 3080: 101秒

### SenseVoice

- **环境**: NVIDIA GeForce RTX 4090
- **显存占用**: 约1.5GB
- **总处理时间**: 27.51秒(171个文件)
- **平均处理时间**: 0.16秒/文件

### Kimi-Audio

- **环境**: NVIDIA GeForce RTX 4090
- **显存占用**: 约30GB
- **总处理时间**: 203.12秒(171个文件)
- **平均处理时间**: 1.19秒/文件
- **部署方法**:
  ```bash
  git clone --recurse-submodules https://github.com/MoonshotAI/Kimi-Audio.git
  ```

### FireRedASR

- **安装与配置**:
  ```bash
  # 环境要求
  # - RedHat 7.8
  # - CUDA Version: 12.2
  # - Driver Version: 535.183.01
  
  # 克隆代码并设置环境
  git clone https://github.com/FireRedTeam/FireRedASR.git
  conda create --name fireredasr python=3.10
  pip install -r requirements.txt
  
  # 运行推理
  cd examples
  bash inference_fireredasr_aed.sh
  ```

## 性能分析与结论

1. **准确性比较**:
   - 在真人录制音频测试中，FireRedASR表现最佳，字符错误率(CER)仅为2.30%
   - 阿里ASR性能接近，CER为2.93%
   - 百度ASR性能相对较弱，CER为8.57%

2. **速度比较**:
   - SenseVoice在处理速度上表现最佳，平均每个文件仅需0.16秒
   - FunASR GPU版本和Dolphin也表现优秀，分别为0.50秒和0.30秒(RTX 4090)
   - Kimi-Audio尽管显存占用最大，但处理速度相对较慢（1.19秒/文件）

3. **资源消耗**:
   - Kimi-Audio显存占用最高，约30GB
   - 大多数模型显存占用在1.5GB到3GB之间
   - FunASR提供CPU版本，适合无GPU环境

4. **稳定性**:
   - 所有测试的ASR模型在处理171个测试文件时均达到100%的成功率
   - FunASR在压力测试中表现稳定，即使在30并发下仍保持100%的成功率

## 总结

本项目对多种ASR模型进行了系统性的评测，从结果来看，不同模型在准确性、速度和资源消耗方面各有优势：

- **最佳准确性**: FireRedASR (CER: 2.30%)
- **最快处理速度**: SenseVoice (0.16秒/文件)
- **最低资源消耗**: SenseVoice (~1.5GB显存)
- **最适合无GPU环境**: FunASR CPU版本