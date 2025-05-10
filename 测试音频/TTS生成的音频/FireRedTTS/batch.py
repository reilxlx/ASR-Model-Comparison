import os
import torchaudio
from fireredtts.models.fireredtts import FireRedTTS # Ensure this is the correct import path
import time
import torch # Import torch for device checks
import pickle # For type hinting or potential specific pickle errors

# === Code to add fairseq.data.dictionary.Dictionary to safe globals ===
# This should be placed at the beginning of your script, after imports.
try:
    from fairseq.data.dictionary import Dictionary as FairseqDictionary
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([FairseqDictionary])
        print("Successfully added FairseqDictionary to torch safe globals for model loading.")
    else:
        print("Warning: torch.serialization.add_safe_globals is not available in this PyTorch version. "
              "If UnpicklingError for fairseq occurs, manual patching of fairseq or PyTorch upgrade might be needed.")
except ImportError:
    print("Warning: Could not import 'FairseqDictionary' from 'fairseq.data.dictionary'. "
          "Ensure fairseq is installed correctly. This might lead to UnpicklingError.")
except Exception as e:
    print(f"An unexpected error occurred while setting up torch safe globals: {e}")
# === End of safe loading code ===

def check_module_device(module_name, module_instance):
    """Helper function to check the device of a PyTorch module's parameters."""
    try:
        # Check the device of the first parameter found
        first_param = next(module_instance.parameters())
        print(f"  - {module_name} is on device: {first_param.device}")
    except StopIteration:
        print(f"  - {module_name} has no parameters.")
    except AttributeError:
        print(f"  - {module_name} is not a valid nn.Module or not found.")
    except Exception as e:
        print(f"  - Error checking device for {module_name}: {e}")

def batch_text_to_speech(input_text_file, output_directory, prompt_wav_path, prompt_text):
    """
    批量将文本文件中的每一行转换为语音并保存为 WAV 文件。
    """
    if not os.path.exists(input_text_file):
        print(f"错误: 输入文本文件 '{input_text_file}' 未找到。")
        return

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"创建输出目录: '{output_directory}'")

    # --- Device Selection ---
    if torch.cuda.is_available():
        selected_device = "cuda"
        print(f"CUDA is available. Attempting to use device: {selected_device}")
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        try:
            print(f"  Current CUDA Device Index: {torch.cuda.current_device()}")
            print(f"  Current CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        except Exception as e:
            print(f"  Could not get current CUDA device details: {e}")
    else:
        selected_device = "cpu"
        print("CUDA not available. Using device: cpu")

    try:
        tts = FireRedTTS(
            config_path="configs/config_24k.json",
            pretrained_path='/rootFireRedTTS-fireredtts-1s/pretrained_models',
            device=selected_device  # Explicitly pass the selected device
        )
        print(f"FireRedTTS model initialized. Target device: {tts.device}")

        # --- Verify component devices after initialization ---
        print("Verifying model component devices:")
        if hasattr(tts, 'semantic_llm'):
            check_module_device("Semantic LLM", tts.semantic_llm)
        if hasattr(tts, 'speech_tokenizer') and hasattr(tts.speech_tokenizer, 'model'):
            check_module_device("Speech Tokenizer (Main Model: SemanticVQVAE)", tts.speech_tokenizer.model)
            if hasattr(tts.speech_tokenizer.model, 'ssl_extractor'): # This is HuBERT
                 check_module_device("Speech Tokenizer (HuBERT within SVQVAE)", tts.speech_tokenizer.model.ssl_extractor)
        if hasattr(tts, 'acoustic_decoder'):
            if hasattr(tts.acoustic_decoder, 'acoustic_llm'):
                check_module_device("Acoustic LLM", tts.acoustic_decoder.acoustic_llm)
            if hasattr(tts.acoustic_decoder, 'acoustic_codec'):
                check_module_device("Acoustic Codec", tts.acoustic_decoder.acoustic_codec)
        # --- End of device verification ---

    except Exception as e:
        print(f"加载 FireRedTTS 模型失败: {e}")
        import traceback
        traceback.print_exc()
        print("请检查 'config_path', 'pretrained_path' 和设备设置是否正确，以及相关依赖是否已安装。")
        return

    try:
        with open(input_text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"从 '{input_text_file}' 中读取 {len(lines)} 行文本。")
    except Exception as e:
        print(f"读取输入文本文件失败: {e}")
        return

    if not lines:
        print("输入文本文件为空，没有可处理的文本。")
        return

    output_sample_rate = 24000 # FireRedTTS default output sample rate
    total_synthesis_time = 0
    total_audio_duration_seconds = 0
    synthesized_files_count = 0

    print("\n开始批量语音合成...")

    for i, text_to_synthesize in enumerate(lines):
        start_time_file = time.time()
        # --- 修改文件名格式 ---
        output_wav_filename = f"sentence_{i+1}.wav"
        output_wav_path = os.path.join(output_directory, output_wav_filename)

        print(f"  正在处理第 {i+1}/{len(lines)} 行: '{text_to_synthesize[:50]}...'")

        try:
            rec_wavs = tts.synthesize(
                prompt_wav=prompt_wav_path,
                prompt_text=prompt_text,
                text=text_to_synthesize,
                lang="zh",
                use_tn=True
            )

            if rec_wavs is None:
                print(f"    处理行 '{text_to_synthesize[:50]}...' 时，tts.synthesize 返回 None。跳过此行。")
                print(f"    这可能表示在 synthesize 方法内部发生了错误。请检查 fireredtts/models/fireredtts.py 中 synthesize 方法的实现（特别是异常处理块）和可能的错误日志。")
                continue

            # rec_wavs from tts.synthesize() should already be on CPU due to .detach().cpu() in synthesize_base
            # If not, the following line ensures it.
            rec_wavs_cpu = rec_wavs.cpu() if rec_wavs.is_cuda else rec_wavs

            torchaudio.save(output_wav_path, rec_wavs_cpu, output_sample_rate)
            
            end_time_file = time.time()
            synthesis_time_file = end_time_file - start_time_file
            total_synthesis_time += synthesis_time_file

            audio_duration_file_seconds = rec_wavs_cpu.shape[-1] / output_sample_rate
            total_audio_duration_seconds += audio_duration_file_seconds
            
            synthesized_files_count += 1
            print(f"    成功生成: '{output_wav_path}', 时长: {audio_duration_file_seconds:.2f}s, 合成耗时: {synthesis_time_file:.2f}s")

        except AttributeError as ae:
            if 'NoneType' in str(ae) and 'detach' in str(ae):
                 print(f"    处理行 '{text_to_synthesize[:50]}...' 时发生 AttributeError (synthesize 可能返回 None): {ae}。跳过此行。")
                 print(f"    请确保您已应用了之前关于修复 fireredtts/models/fireredtts.py 中 synthesize 方法的建议。")
            else:
                 print(f"    处理行 '{text_to_synthesize[:50]}...' 时发生 AttributeError: {ae}。跳过此行。")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"    处理行 '{text_to_synthesize[:50]}...' 时发生未知错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n批量语音合成完成。")

    if synthesized_files_count > 0:
        average_time_per_file = total_synthesis_time / synthesized_files_count
        rtf = total_synthesis_time / total_audio_duration_seconds if total_audio_duration_seconds > 0 else float('inf')

        print("\n--- 统计结果 ---")
        print(f"成功生成文件数: {synthesized_files_count}")
        print(f"总合成耗时: {total_synthesis_time:.2f} 秒")
        print(f"平均每个文件合成耗时: {average_time_per_file:.2f} 秒")
        print(f"总音频时长: {total_audio_duration_seconds:.2f} 秒")
        print(f"实时率 (RTF): {rtf:.4f} (越小越好，表示 合成耗时 / 音频总时长)")
    else:
        print("没有成功生成任何音频文件。")

if __name__ == "__main__":
    INPUT_TEXT_FILE_PATH = "/root/文本.txt" 
    OUTPUT_DIRECTORY_PATH = "/root/FireRedTTS-fireredtts-1s/audio"
    PROMPT_WAV_PATH = "/root/liutao.mp3" 
    PROMPT_TEXT_CONTENT = "哈喽大家好我是刘涛经常会有粉丝问我，为什么身体看起来一只滑溜溜的，今天我就来揭秘啦。"
    
    print("--- 开始批量语音合成任务 ---")
    print(f"输入文本文件: '{INPUT_TEXT_FILE_PATH}'")
    print(f"输出目录: '{OUTPUT_DIRECTORY_PATH}'")
    print(f"提示音频路径: '{PROMPT_WAV_PATH}'")
    print(f"提示音频文本: '{PROMPT_TEXT_CONTENT[:30]}...'")
    print("请确保 FireRedTTS 模型的 'config_path' 和 'pretrained_path' 在 batch_text_to_speech 函数中设置正确, 且提示音频文件存在。")

    batch_text_to_speech(
        input_text_file=INPUT_TEXT_FILE_PATH,
        output_directory=OUTPUT_DIRECTORY_PATH,
        prompt_wav_path=PROMPT_WAV_PATH,
        prompt_text=PROMPT_TEXT_CONTENT
    )