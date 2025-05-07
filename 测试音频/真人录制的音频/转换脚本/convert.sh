#!/bin/bash

# 设置起始序号
index=1

# 遍历当前目录下的所有 .m4a 文件，按数字顺序处理
for file in $(ls *.m4a | sort -n); do
    # 生成目标文件名
    output="sentence_${index}.wav"
    
    # 使用 ffmpeg 转换格式并设置采样率为 16000 Hz
    ffmpeg -y -i "$file" -ar 16000 "$output"

    # 序号加一
    index=$((index + 1))
done

echo "转换完成！"