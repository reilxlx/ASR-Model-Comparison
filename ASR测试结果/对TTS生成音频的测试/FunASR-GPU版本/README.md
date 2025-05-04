### 显存占用：
```
Every 2.0s: nvidia-smi                                                                                                                                                                                                                                                                     ubuntu22: Thu May  1 19:27:45 2025

Thu May  1 19:27:45 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.78                 Driver Version: 550.78         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080        Off |   00000000:00:08.0 Off |                  N/A |
| 30%   35C    P2             86W /  320W |    2967MiB /  10240MiB |     10%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A       762      G   /usr/lib/xorg/Xorg                              4MiB |
|    0   N/A  N/A      3431      C   ...bsocket/build/bin/funasr-wss-server       2952MiB |
+-----------------------------------------------------------------------------------------+
```

### 总耗时：
```
Total files found:         171
Successfully processed:    171
Failed to process:         0
Total batch duration:      0:01:25.804854
Total API request time (successful): 0:01:25.753410
Average API request time per successful file: 0.50 
```

### 服务部署步骤：
```
docker启动命令：
sudo docker pull \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-gpu-0.2.0
mkdir -p ./funasr-runtime-resources/models
sudo docker run --gpus=all -p 10098:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-gpu-0.2.0
```

```
ASR服务启动命令：
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch  \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt \
  --decoder-thread-num  24 \
  --io-thread-num  24 \
  --model-thread-num  1 \
  --certfile 0 > log.txt 2>&1 &
```

```
应用服务启动命令：
sh start_server.sh
```

```
未建立数据库之前的压力测试，基于main.py运行后测试：
python load_test_by_txt_tqdm.py \
    --url "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&ssl=false&use_itn=true&audio_fs=16000" \
    --audio-file "/root/home/fish-speech-liutao-16k/sentence_104.wav" \
    --concurrency-levels "10,20,30,40,50" \
    --requests-per-level 500
```

```
建立数据库之后的压力测试，基于main_Gemini2_5pro&DataBase.py运行后测试：
python load_test_by_txt_tqdm.py     --url "http://127.0.0.1:8000/transcribe?host=127.0.0.1&port=10098&mode=offline&chunk_size=5,10,5&chunk_interval=10&ssl=false&use_itn=true&audio_fs=16000&req_seqno=TEST001&req_channelid=CURL_BASIC"     --audio-file "/root/home/fish-speech-liutao-16k/sentence_104.wav"     --concurrency-levels "10,20,30,40,50"     --requests-per-level 500
```
