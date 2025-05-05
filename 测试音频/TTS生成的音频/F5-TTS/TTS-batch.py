import subprocess
import os
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='逐行生成语音合成文件')
    parser.add_argument('--model', type=str, default='F5TTS_v1_Base', help='TTS模型名称')
    parser.add_argument('--ref_audio', type=str, default='liutao.mp3', 
                        help='参考音频文件路径')
    parser.add_argument('--ref_text', type=str, 
                        default='哈喽大家好我是刘涛经常会有粉丝问我，为什么身体看起来一直滑溜溜的，今天我就来揭秘啦。', 
                        help='参考文本内容')
    parser.add_argument('--input_file', type=str, 
                        default='wenben.txt', 
                        help='需要合成的文本文件路径')
    parser.add_argument('--output_dir', type=str, default='tests', help='输出目录')
    #parser.add_argument('--remove_silence', type=bool, default=False, help='是否移除静音部分')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 读取需要合成的文本文件
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 逐行处理文本并生成语音
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # 跳过空行
            continue
        
        output_file = f"sentence_{i}.wav"
        
        # 构建命令
        cmd = [
            'f5-tts_infer-cli',
            '--model', args.model,
            '--ref_audio', args.ref_audio,
            '--ref_text', args.ref_text,
            '--gen_text', line,
            '--output_file', output_file,
            '--output_dir', args.output_dir
            #'--remove_silence', 'false'
        ]
        
        # 执行命令
        try:
            print(f"正在处理第 {i} 行文本: {line}")
            subprocess.run(cmd, check=True)
            print(f"已生成 {os.path.join(args.output_dir, output_file)}")
        except subprocess.CalledProcessError as e:
            print(f"处理第 {i} 行时出错: {e}")
    
    print("所有文本处理完成！")

if __name__ == "__main__":
    main()


# 一次性生成TTS音频
# f5-tts_infer-cli --model F5TTS_v1_Base \
# --ref_audio "liutao.mp3" \
# --ref_text "哈喽大家好我是刘涛经常会有粉丝问我，为什么身体看起来一直滑溜溜的，今天我就来揭秘啦。" \
# --gen_text "信用评级机构对企业或债券发行人的信用风险进行评估并给出等级，为投资者提供决策参考，但其评级的独立性和准确性也时常受到质疑。"