import os
import torchaudio
from fireredtts.models.fireredtts import FireRedTTS # 确保这是您上次修改后的正确导入路径

import torch
import pickle # 用于捕获 _pickle.UnpicklingError，尽管这里主要用于类型提示

# 尝试导入 fairseq.data.dictionary.Dictionary 并将其添加到 PyTorch 的安全全局变量中
# 这应该在任何可能触发模型加载的代码（如 FireRedTTS 初始化）之前完成
try:
    from fairseq.data.dictionary import Dictionary as FairseqDictionary
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([FairseqDictionary])
    else:
        # 如果 add_safe_globals 不可用，您可能使用的是不支持此方法的旧版 PyTorch
        # 或者需要直接修改 fairseq 库（不推荐）
        print("Warning: torch.serialization.add_safe_globals is not available. "
              "If the UnpicklingError persists, you might need to modify the fairseq library "
              "to use torch.load(..., weights_only=False) or ensure your PyTorch version supports this safety feature correctly.")
except ImportError:
    print("Warning: Could not import 'FairseqDictionary' from 'fairseq.data.dictionary'. "
          "Make sure fairseq is installed correctly. This might lead to UnpicklingError if not resolved.")
except Exception as e:
    print(f"An unexpected error occurred while trying to set up safe globals: {e}")

# 注意：请确保将 <pretrained_models_dir> 替换为实际的预训练模型目录路径
# 例如，如果 pretrained_models 文件夹与 test.py 在同一目录下，则使用 "pretrained_models"
# 如果它在 /root/FireRedTTS-fireredtts-1s/pretrained_models，并且test.py也在/root/FireRedTTS-fireredtts-1s/，则路径是 "pretrained_models"
# 请根据您的实际文件结构进行设置
PRETRAINED_MODELS_PATH = "/root/FireRedTTS-fireredtts-1s/pretrained_models" # <--- 修改这里

tts = FireRedTTS(
        config_path="configs/config_24k.json",
        pretrained_path=PRETRAINED_MODELS_PATH,
        device="cuda"
  )

# same language
# For the test-hard evaluation, we enabled the use_tn=True configuration setting.
rec_wavs = tts.synthesize(
  prompt_wav="/root/liutao.mp3",
  prompt_text="哈喽大家好我是刘涛经常会有粉丝问我，为什么身体看起来一只滑溜溜的，今天我就来揭秘啦。",
  text="小红书，是中国大陆的网络购物和社交平台，成立于二零一三年六月。",
  lang="zh",
  use_tn=True
)
rec_wavs = rec_wavs.detach().cpu()
out_wav_path = os.path.join("./example.wav")
torchaudio.save(out_wav_path, rec_wavs, 24000)

print(f"Synthesized audio saved to {out_wav_path}")