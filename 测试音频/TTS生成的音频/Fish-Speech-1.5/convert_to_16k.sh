#!/bin/bash

INPUT_DIR="ASR-Model-Comparison/测试音频/真人录制的音频/audio"     # 源目录，请根据需要修改
OUTPUT_DIR="ASR-Model-Comparison/测试音频/真人录制的音频/audio-16k"   # 输出目录，请根据需要修改

find "$INPUT_DIR" -type f -iname "*.wav" | while read -r input_file; do
    # 生成相对路径
    rel_path="${input_file#$INPUT_DIR/}"
    output_file="$OUTPUT_DIR/$rel_path"

    # 创建输出目录（如果不存在）
    mkdir -p "$(dirname "$output_file")"

    # 转换采样率为16kHz
    ffmpeg -y -i "$input_file" -ar 16000 "$output_file"
    echo "✅ Converted: $input_file -> $output_file"
done