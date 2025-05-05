import Levenshtein
import os
import re
import string # 用于获取英文标点

def remove_punctuation(text):
    """
    移除字符串中的中英文标点符号。
    """
    # 定义要移除的标点符号
    # string.punctuation 包含英文标点: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # 添加常见的中文标点（全角和半角），以及空格（\s）也常被视为空白需要处理
    # 注意：\s 会匹配任何空白字符，包括空格、制表符、换行符等。
    # 如果你只想移除标点，保留空格，可以将 \s 去掉。
    # 如果只想移除特定类型的空格（如全角空格），可以明确列出：　
    punctuation = string.punctuation + "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。" + "　" # 添加了全角空格

    # 使用正则表达式移除标点和空白（如果包含\s）
    # re.escape() 用于转义 punctuation 字符串中的特殊正则字符
    # text_without_punct = re.sub(r"[{}]".format(re.escape(punctuation)), "", text)
    # --- 或者，更简洁地匹配 Unicode 标点类别 ---
    # 使用 \p{P} 匹配 Unicode 标点类别，\p{Z} 匹配 Unicode 分隔符（包括空格）
    # 需要安装 'regex' 库: pip install regex
    # import regex
    # text_without_punct = regex.sub(r"[\p{P}\p{Z}]+", "", text) # 移除所有标点和空白
    # text_without_punct = regex.sub(r"\p{P}+", "", text) # 只移除标点，保留空白

    # --- 使用内建 re 库，移除标点和特定空格 ---
    text_without_punct = re.sub(r"[{}\\s]".format(re.escape(punctuation)), "", text) # 移除定义的标点和所有空白

    return text_without_punct

def calculate_cer_ignore_punctuation(reference, hypothesis):
    """
    计算两个字符串之间忽略标点的字符错误率 (CER)。

    Args:
        reference (str): 参考字符串 (ground truth)。
        hypothesis (str): 假设字符串 (待评估)。

    Returns:
        float: 这一对字符串的 CER。如果移除标点后的参考字符串为空，则特殊处理。
    """
    # 去除首尾可能存在的空白字符
    ref_orig = reference.strip()
    hyp_orig = hypothesis.strip()

    # 移除标点符号
    ref_cleaned = remove_punctuation(ref_orig)
    hyp_cleaned = remove_punctuation(hyp_orig)

    # 计算编辑距离 (基于清理后的文本)
    edit_distance = Levenshtein.distance(ref_cleaned, hyp_cleaned)

    # 参考字符串的长度 (基于清理后的文本)
    ref_len_cleaned = len(ref_cleaned)

    if ref_len_cleaned == 0:
        if len(hyp_cleaned) == 0:
            return 0.0
        else:
            # 如果清理后的参考为空，但假设不为空，视为 100% 错误
            # 或者根据下面的平均策略，不计入平均值
            return 1.0 # 保持与原逻辑一致，但在 average 函数中处理
    else:
        # 计算 CER
        cer = edit_distance / ref_len_cleaned
        return cer

def average_cer_for_files_ignore_punctuation(ref_filepath, hyp_filepath):
    """
    计算两个文件之间忽略标点的平均 CER。

    Args:
        ref_filepath (str): 参考文本文件的路径。
        hyp_filepath (str): 假设文本文件的路径。

    Returns:
        float or None: 文件间的平均 CER。如果文件无法读取或没有有效内容行，返回 None 或 0.0。
    """
    total_cer = 0.0
    total_edit_distance = 0
    total_ref_len_cleaned = 0
    valid_line_count = 0 # 只计算清理后参考文本非空的行的数量

    try:
        with open(ref_filepath, 'r', encoding='utf-8') as f_ref, \
             open(hyp_filepath, 'r', encoding='utf-8') as f_hyp:

            for line_num, (ref_line, hyp_line) in enumerate(zip(f_ref, f_hyp), 1):
                ref_line_stripped = ref_line.strip()
                hyp_line_stripped = hyp_line.strip()

                # 移除标点
                ref_cleaned = remove_punctuation(ref_line_stripped)
                hyp_cleaned = remove_punctuation(hyp_line_stripped)

                ref_len_cleaned = len(ref_cleaned)

                # --- 关键处理逻辑 ---
                # 只在清理后的参考行非空时才计算CER并计入平均值
                if ref_len_cleaned > 0:
                    edit_distance = Levenshtein.distance(ref_cleaned, hyp_cleaned)
                    # line_cer = edit_distance / ref_len_cleaned # 可以单行计算，但累加距离和长度更精确
                    # total_cer += line_cer
                    total_edit_distance += edit_distance
                    total_ref_len_cleaned += ref_len_cleaned
                    valid_line_count += 1
                elif len(hyp_cleaned) > 0:
                     # 清理后的参考行为空，但假设行不为空。这种情况不计入 CER 平均值。
                     print(f"信息: 第 {line_num} 行：清理后的参考行为空，但假设行非空，已跳过。")
                     pass
                else:
                     # 两行清理后都为空，不计入计算。
                     pass

            # 注意: zip 会在最短的文件结束后停止。

    except FileNotFoundError:
        print(f"错误: 文件未找到。请确保 '{ref_filepath}' 和 '{hyp_filepath}' 存在。")
        return None
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return None

    if valid_line_count == 0:
        print("警告: 没有找到有效的行（清理后的参考文本非空）来计算 CER。")
        return 0.0

    # 基于总编辑距离和总参考长度计算平均 CER，这比平均行 CER 更标准
    average_cer = total_edit_distance / total_ref_len_cleaned
    return average_cer

# --- 主程序 ---
if __name__ == "__main__":
    # 定义文件路径
    reference_file = "文本.txt"
    hypothesis_file = "识别结果.txt"

    # --- 创建示例文件 (如果不存在，与之前相同) ---
    if not os.path.exists(reference_file):
        print(f"创建示例文件: {reference_file}")
        with open(reference_file, 'w', encoding='utf-8') as f:
            f.write("这是参考的第一行。\n")
            f.write("这是第二行。\n")
            f.write("第三行完全不同。\n")
            f.write("\n") # 空行
            f.write("最后一行参考文本\n")
            f.write("中文标点，很重要。\n")
            f.write("只有标点。。。\n")

    if not os.path.exists(hypothesis_file):
        print(f"创建示例文件: {hypothesis_file}")
        with open(hypothesis_file, 'w', encoding='utf-8') as f:
            f.write("这是参考的第一行。\n") # 完全匹配
            f.write("这是第 Эр 行。\n")     # 替换: 二 -> Эр (2个字符替换1个)
            f.write("假设文本。\n")        # 大量不同
            f.write("假设这边有内容\n")    # 参考为空，假设有内容
            f.write("最后一行假设文本\n")  # 部分匹配
            f.write("中文标点，不重要。\n") # 标点不同 -> 忽略标点后比较 很 vs 不
            f.write("！？，\n")          # 只有标点

    print(f"\n正在计算文件 '{reference_file}' (参考) 和 '{hypothesis_file}' (假设) 之间忽略标点的平均 CER...")

    avg_cer = average_cer_for_files_ignore_punctuation(reference_file, hypothesis_file)

    if avg_cer is not None:
        print("\n计算完成。")
        print(f"平均字符错误率 (Average CER, 忽略标点): {avg_cer:.4f} ({avg_cer * 100:.2f}%)")

    # 示例计算分析 (忽略标点):
    # Cleaned Ref vs Cleaned Hyp -> Dist, Cleaned Ref Len, Contributes to Avg?
    # 行1: "这是参考的第一行" vs "这是参考的第一行" -> dist=0, len=8, Yes
    # 行2: "这是第二行" vs "这是第Эр行" -> dist=2 (二->Эр), len=5, Yes
    # 行3: "第三行完全不同" vs "假设文本" -> dist=6 (估算), len=7, Yes
    # 行4: "" vs "假设这边有内容" -> ref_len=0, No (打印信息)
    # 行5: "最后一行参考文本" vs "最后一行假设文本" -> dist=1 (参->设), len=8, Yes
    # 行6: "中文标点很重要" vs "中文标点不重要" -> dist=1 (很->不), len=7, Yes
    # 行7: "只有标点" vs "" -> dist=4 (只有标点 -> 空), len=4, Yes (因为清理后参考非空)
    #
    # Total Edit Distance = 0 + 2 + 6 + 1 + 1 + 4 = 14
    # Total Cleaned Ref Length = 8 + 5 + 7 + 8 + 7 + 4 = 39
    # Valid Line Count = 6
    # Average CER = 14 / 39 ≈ 0.3590
    # (实际 Levenshtein 距离库计算可能更精确，这里手动计算仅为说明)