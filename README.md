 # ASR模型性能综合对比评测

## 项目概述

本项目旨在对多种自动语音识别(ASR)模型进行全面评测和比较，包括开源和商业模型。通过使用不同来源的测试音频数据（包括真人录制和多种TTS生成的音频），对不同ASR系统的准确性、速度和资源消耗等方面进行客观对比。

## 项目结构

```
ASR-Model-Comparison/
├── 测试音频/                     # 用于测试的音频文件
│   ├── TTS生成的音频/            # 使用TTS技术生成的测试音频
│   │   ├── Fish-Speech-1.5/     # 使用Fish-Speech-1.5模型生成的音频
│   │   ├── Index-TTS/           # 使用Index-TTS模型生成的音频
│   │   └── F5-TTS/              # 使用F5-TTS模型生成的音频
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

### 数据集基本信息

1. **真人录制音频数据集**：
   - 音频文件数量: 171个
   - 总时长：32分18秒 (1937.88秒)
   - 平均时长：11.33秒/文件

2. **Fish-Speech-1.5生成音频数据集**：
   - 音频文件数量：171
   - 总时长（秒）：1633.58
   - 总时长（hh:mm:ss）：00:27:14
   - 平均时长（秒）：9.55
   - 平均时长（hh:mm:ss）：00:00:10

3. **Index-TTS生成音频数据集**：
   - 音频文件数量：500
   - 总音频时长：10616.66秒（约176.94分钟）
   
4. **F5-TTS生成音频数据集**：
   - 音频文件数量：500

### 文本内容

测试数据集包含多种场景和领域的内容，详见`测试音频/文本.txt`文件。

## 模型测试结果

### 对真人录制音频的测试结果 (171个文件)

| ASR模型 | 平均字符错误率(CER) | 处理总时间(秒) | 实时因子(RTF) | 成功处理文件数 | 错误文件数 | 平均处理时间/文件(秒) |
| --- | --- | --- | --- | --- | --- | --- |
| FireRedASR | 2.30% | 289.01 | 0.15 | 171 | 0 | 1.69 |
| 阿里ASR | 2.93% | 447 | 0.23 | 171 | 0 | 2.61 |
| 百度ASR | 8.57% | 1974.18 | 1.02 | 171 | 0 | 11.54 |

### 对Fish-Speech-1.5生成音频的测试结果 (171个文件)

| ASR模型 | 文件处理成功率 | 平均字符错误率(CER) | 总处理时间(秒) | 平均处理时间/文件(秒) | 显存占用 |
| --- | --- | --- | --- | --- | --- |
| FireRedASR | 100% (171/171) | 0.93% | 401.88 | 2.35 | ～5.6GB （T4） |
| Kimi-Audio | 100% (171/171) | 1.25% | 203.12 | 1.19 | ~30GB (RTX 4090) |
| SenseVoice | 100% (171/171) | 2.23% | 27.51 | 0.16 | ~1.5GB (RTX 4090) |
| 阿里ASR | 100% (171/171) | 2.73% | 396 | 2.32 | Kunpeng 920 (CPU only) |
| Dolphin | 100% (171/171) | 2.92% | 101/52* | 0.59/0.30* | ~2.3GB (RTX 3080/4090**) |
| FunASR-GPU版本 | 100% (171/171) | 3.98% | 85.80 | 0.50 | ~3GB (RTX 3080) |
| FunASR-CPU版本 | 100% (171/171) | 4.02% | 115.11 | 0.67 | N/A (CPU only) |
| 百度ASR | 100% (171/171) | 7.70% | 1679.65 | 9.82 | Kunpeng 920 (CPU only) |

*注: Dolphin模型分别在RTX 3080(101秒)和RTX 4090(52秒)上进行了测试

### 对Index-TTS生成音频的测试结果 (500个文件)

| ASR模型 | 平均字错误率(CER) | 总识别时间(秒) | 平均每条音频处理时间(秒) | RTF | 显存占用 | 测试设备 |
|---------|-------------------|--------------|---------------------|------|---------|---------|
| FireRedASR | 0.54% | 866.85 | 1.73 | 0.0816 | 9.6GB | RTX 4090 |
| Kimi-Audio | 0.55% | 754.62 | 1.51 | 0.0711 | 29.7GB | RTX 4090 |
| SenseVoice (4090) | 1.13% | 52.57 | 0.11 | 0.0049 | 1.5GB | RTX 4090 |
| SenseVoice (3080) | 1.51% | 109.65 | 0.22 | 0.0103 | - | RTX 3080 |
| FunASR-CPU | 1.44% | 2811.92 | 5.62 | 0.2648 | 不适用 | CPU |
| FunASR-GPU | 2.17% | 500.63 | 1.00 | 0.0471 | 3.7GB | RTX 4090 |
| Dolphin | 19.39% | 243 | 0.49 | 0.0229 | 2.1GB | RTX 4090 |

### 对F5-TTS生成音频的测试结果 (500个文件)

| 模型名称 | 处理设备 | 总处理时间(秒) | 平均每文件处理时间(秒) | 字符错误率(CER) | 成功处理文件数 |
|---------|---------|--------------|---------------------|----------------|--------------|
| Dolphin | 4090 | 267 | 0.53 | 12.20% | 500/500 |
| FireRedASR | T4 | 6150.43 | 12.3 | 0.82% | 500/500 |
| FunASR-GPU | 4090 | 433.58 | 0.87 | 2.41% | 500/500 |
| Kimi-Audio | 4090 | 755.06 | 1.51 | 0.59% | 500/500 |
| SenseVoice | 4090 | 54.13 | 0.11 | 1.41% | 500/500 |

## 不同TTS音频测试的对比分析

### 准确率对比 (CER从低到高排序)

| ASR模型 | 真人录制音频 | Fish-Speech-1.5 | Index-TTS | F5-TTS |
|---------|------------|-----------------|-----------|--------|
| FireRedASR | 2.30% | 0.93% | 0.54% | 0.82% |
| Kimi-Audio | - | 1.25% | 0.55% | 0.59% |
| SenseVoice | - | 2.23% | 1.13% | 1.41% |
| FunASR-GPU | - | 3.98% | 2.17% | 2.41% |
| Dolphin | - | 2.92% | 19.39% | 12.20% |
| 阿里ASR | 2.93% | 2.73% | - | - |
| 百度ASR | 8.57% | 7.70% | - | - |

### 处理速度对比 (平均每文件处理时间/秒)

| ASR模型 | 真人录制音频 | Fish-Speech-1.5 | Index-TTS | F5-TTS |
|---------|------------|-----------------|-----------|--------|
| SenseVoice | - | 0.16 | 0.11 | 0.11 |
| Dolphin | - | 0.30~0.59 | 0.49 | 0.53 |
| FunASR-GPU | - | 0.50 | 1.00 | 0.87 |
| Kimi-Audio | - | 1.19 | 1.51 | 1.51 |
| FireRedASR | 1.69 | 2.35 | 1.73 | 12.3 |
| 阿里ASR | 2.61 | 2.32 | - | - |
| 百度ASR | 11.54 | 9.82 | - | - |

## 详细测试结果与分析

### FunASR (GPU版本)

- **环境**: NVIDIA GeForce RTX 3080/4090
- **显存占用**: 约3-3.7GB
- **平均处理时间**: 0.50-1.00秒/文件
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
- **平均处理时间**: 0.67-5.62秒/文件 (取决于音频数据集)
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
- **显存占用**: 约2.1-2.3GB
- **平均处理时间**: 0.30-0.59秒/文件
- **准确率差异**: 在不同TTS数据集上表现差异较大 (2.92% vs 19.39% vs 12.20%)

### SenseVoice

- **环境**: NVIDIA GeForce RTX 4090/3080
- **显存占用**: 约1.5GB
- **平均处理时间**: 0.11-0.16秒/文件 (所有模型中最快)
- **准确率**: 在所有TTS数据集上保持稳定良好表现 (1.13%-2.23%)

### Kimi-Audio

- **环境**: NVIDIA GeForce RTX 4090
- **显存占用**: 约30GB (所有模型中最高)
- **平均处理时间**: 1.19-1.51秒/文件
- **准确率**: 在所有TTS数据集上表现优异 (0.55%-1.25%)
- **部署方法**:
  ```bash
  git clone --recurse-submodules https://github.com/MoonshotAI/Kimi-Audio.git
  ```

### FireRedASR

- **环境**: NVIDIA T4/RTX 4090
- **显存占用**: 5.6-9.6GB
- **平均处理时间**: 1.69-12.3秒/文件 (在T4上处理F5-TTS数据集时最慢)
- **准确率**: 在所有测试中表现最优 (0.54%-2.30%)
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

## 综合性能分析与结论

### 多数据集准确性比较

- **最佳准确性模型**:
  - FireRedASR在所有测试数据集上表现最佳，CER均在2.30%以下
  - Kimi-Audio紧随其后，在TTS生成音频上表现尤为突出 (CER < 1%)
  - SenseVoice保持稳定的良好表现，CER通常在1-2%之间

- **准确性表现异常**:
  - Dolphin在不同TTS数据集上表现差异极大 (2.92%-19.39%)，对Index-TTS音频的识别尤其困难
  - 百度ASR在所有测试中表现相对较弱，CER通常在7-9%之间

### 多数据集速度比较

- **最快处理速度**:
  - SenseVoice在所有测试中速度最快，平均处理时间仅0.11-0.16秒/文件
  - Dolphin处理速度次之，平均0.30-0.59秒/文件
  - FunASR-GPU处理速度表现优秀，平均0.50-1.00秒/文件

- **处理速度受硬件影响明显**:
  - 同一模型在不同硬件上表现差异大 (如FireRedASR在T4vs4090)
  - CPU版本模型通常比GPU版本慢3-10倍

### 资源消耗比较

- **最低资源消耗**:
  - SenseVoice只需约1.5GB显存，是显存占用最小的GPU模型
  - FunASR-CPU可在无GPU环境运行，但速度会大幅降低

- **高资源需求**:
  - Kimi-Audio需约30GB显存，要求高性能GPU
  - FireRedASR显存需求中等至较高(5.6-9.6GB)

### 不同TTS音频来源影响

- 不同TTS引擎生成的音频对ASR识别准确率有显著影响
- Index-TTS和F5-TTS生成的音频通常比Fish-Speech-1.5更易被识别
- 模型对不同TTS数据的适应性存在差异，如Dolphin对不同TTS音频的识别能力差异极大

## 最佳选择建议

根据不同应用场景，推荐以下ASR模型：

1. **追求最高准确率场景**:
   - FireRedASR 或 Kimi-Audio
   - 适用于医疗、法律等对准确性有极高要求的场景

2. **实时处理场景**:
   - SenseVoice (低延迟，高吞吐量)
   - 适用于直播转写、实时会议记录等

3. **资源受限场景**:
   - SenseVoice (GPU资源有限)
   - FunASR-CPU (无GPU环境)

4. **平衡性能场景**:
   - FunASR-GPU (准确率和速度较为平衡)
   - 适用于一般商业应用

5. **高并发场景**:
   - FunASR (压力测试表现良好)
   - 适用于需要处理多用户同时请求的系统

## 注意事项

- 测试结果仅基于特定数据集，实际应用中模型表现可能有所不同
- 硬件配置对模型性能有显著影响，应根据实际部署环境选择合适的模型
- 所有测试模型在处理测试文件时都达到了100%的成功率，但在实际应用中可能面临更复杂的情况