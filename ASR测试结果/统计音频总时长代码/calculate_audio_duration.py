import os
import sys
import argparse
from pydub.utils import mediainfo

def get_audio_duration(filepath):
    info = mediainfo(filepath)
    return float(info['duration'])

def get_all_audio_files(folder, extensions=('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                yield os.path.join(root, file)

def main(folder):
    total_duration = 0.0
    count = 0
    for filepath in get_all_audio_files(folder):
        try:
            duration = get_audio_duration(filepath)
            total_duration += duration
            count += 1
        except Exception as e:
            print(f"无法读取文件 {filepath}: {e}")

    print(f"\n共找到 {count} 个音频文件")
    print(f"总时长: {total_duration:.2f} 秒（约 {total_duration / 60:.2f} 分钟）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计文件夹下所有音频的总时长")
    parser.add_argument("folder", help="目标文件夹路径")
    args = parser.parse_args()

    main(args.folder)