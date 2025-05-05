import subprocess
import os
import argparse
import re # 导入正则表达式模块

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='检查并补全缺失的语音合成文件')
    parser.add_argument('--model', type=str, default='F5TTS_v1_Base', help='TTS模型名称')
    parser.add_argument('--ref_audio', type=str, default='liutao.mp3',
                        help='参考音频文件路径')
    parser.add_argument('--ref_text', type=str,
                        default='哈喽大家好我是刘涛经常会有粉丝问我，为什么身体看起来一直滑溜溜的，今天我就来揭秘啦。',
                        help='参考文本内容')
    parser.add_argument('--input_file', type=str,
                        default='wenben.txt',
                        help='需要合成的文本文件路径')
    parser.add_argument('--output_dir', type=str, default='tts', help='输出目录')
    # parser.add_argument('--remove_silence', type=bool, default=False, help='是否移除静音部分') # 注释掉或根据需要保留

    args = parser.parse_args()

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        print(f"输出目录 '{args.output_dir}' 不存在，将创建它。")
        os.makedirs(args.output_dir)
    else:
        print(f"输出目录 '{args.output_dir}' 已存在。")

    # --- 1. 确定应该存在的文件列表 ---
    lines_to_process = {} # 使用字典存储 {行号: 文本}
    expected_indices = set()
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 只处理非空行
                    lines_to_process[i] = line
                    expected_indices.add(i)
        print(f"文件 '{args.input_file}' 中共有 {len(expected_indices)} 个非空行需要处理。")
        if not expected_indices:
            print("输入文件为空或不包含有效文本行，程序退出。")
            return
    except FileNotFoundError:
        print(f"错误：输入文件 '{args.input_file}' 未找到。")
        return
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
        return

    # --- 2. 确定实际存在的文件列表 ---
    existing_indices = set()
    filename_pattern = re.compile(r'^sentence_(\d+)\.wav$') # 正则表达式匹配文件名
    try:
        for filename in os.listdir(args.output_dir):
            match = filename_pattern.match(filename)
            if match:
                # 提取文件名中的数字编号
                index = int(match.group(1))
                existing_indices.add(index)
        print(f"输出目录 '{args.output_dir}' 中已存在 {len(existing_indices)} 个 'sentence_*.wav' 文件。")
    except FileNotFoundError:
        # This case should not happen due to the check above, but included for robustness
        print(f"警告：输出目录 '{args.output_dir}' 在检查文件时未找到（可能在创建后被删除）。")
    except Exception as e:
        print(f"扫描输出目录时出错: {e}")
        # Decide if you want to continue or exit; continuing might be risky.
        # return

    # --- 3. 找出缺失的文件 ---
    missing_indices = expected_indices - existing_indices

    if not missing_indices:
        print("所有预期的音频文件均已存在，无需生成。")
        print("处理完成！")
        return

    print(f"发现 {len(missing_indices)} 个缺失的音频文件，将开始生成...")
    # print(f"缺失的文件编号: {sorted(list(missing_indices))}") # 可选：打印缺失的编号

    # --- 4. 重新生成缺失的文件 ---
    generated_count = 0
    failed_count = 0
    # 按行号顺序处理缺失的文件
    for i in sorted(list(missing_indices)):
        if i not in lines_to_process:
             print(f"警告：在输入文件中找不到缺失索引 {i} 对应的文本行（这不应该发生）。跳过。")
             continue

        line = lines_to_process[i]
        output_file = f"sentence_{i}.wav"
        full_output_path = os.path.join(args.output_dir, output_file)

        # 构建命令
        cmd = [
            'f5-tts_infer-cli',
            '--model', args.model,
            '--ref_audio', args.ref_audio,
            '--ref_text', args.ref_text,
            '--gen_text', line,
            '--output_file', output_file, # f5-tts_infer-cli 可能只接受文件名
            '--output_dir', args.output_dir # 指定输出目录
            # '--remove_silence', 'false' # 根据需要取消注释
        ]

        # 执行命令
        try:
            print(f"\n正在生成缺失文件 (第 {i} 行): {output_file}")
            print(f"文本内容: {line}")
            # 使用 subprocess.run，增加 capture_output=True 以便捕获详细错误
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"成功生成: {full_output_path}")
            # print(f"命令输出:\n{result.stdout}") # 可选：打印成功时的输出
            generated_count += 1
        except subprocess.CalledProcessError as e:
            print(f"处理第 {i} 行时出错: {e}")
            print(f"命令: {' '.join(cmd)}")
            print(f"返回码: {e.returncode}")
            print(f"标准输出:\n{e.stdout}")
            print(f"标准错误:\n{e.stderr}")
            failed_count += 1
        except FileNotFoundError:
             print(f"错误：命令 'f5-tts_infer-cli' 未找到。请确保它已安装并在系统 PATH 中。")
             # 如果命令找不到，后续尝试也没有意义，可以选择退出
             return
        except Exception as e:
             print(f"处理第 {i} 行时发生未知错误: {e}")
             failed_count += 1

    print("\n补全处理完成！")
    print(f"总共需要生成 {len(missing_indices)} 个缺失文件。")
    print(f"成功生成 {generated_count} 个文件。")
    if failed_count > 0:
        print(f"有 {failed_count} 个文件生成失败，请检查上面的错误信息。")

if __name__ == "__main__":
    main()

# python generate_missing_audio.py