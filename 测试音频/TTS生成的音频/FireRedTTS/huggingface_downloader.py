# 导入必要的库
from huggingface_hub import snapshot_download
import os
import argparse # 用于从命令行解析参数

def download_model_from_huggingface(repo_id: str, local_dir: str, token: str = None, revision: str = None):
    """
    从 Hugging Face Hub 下载模型或数据集到指定的本地目录。

    参数:
    repo_id (str): Hugging Face Hub 上的模型或数据集的 ID (例如 "bert-base-uncased" 或 "meta-llama/Llama-2-7b-chat-hf")。
    local_dir (str): 下载文件要保存到的本地目录路径。
    token (str, optional): 您的 Hugging Face API token。如果模型是私有的或需要认证，则需要此参数。默认为 None。
    revision (str, optional): 要下载的特定模型版本 (例如，分支名、标签或提交哈希)。默认为 None (下载默认分支，通常是 'main')。
    """
    try:
        print(f"开始下载模型 '{repo_id}' (版本: {revision if revision else '默认'}) 到 '{local_dir}'...")

        # 创建本地目录 (如果不存在)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"已创建目录: {local_dir}")

        # 使用 snapshot_download 下载模型
        # ignore_patterns 参数可以用来排除不需要的文件，例如 "*.safetensors" 或 "*.h5"
        # resume_download=True 允许在下载中断时续传
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 建议设为 False 以复制文件而不是创建符号链接，除非您有特定需求
            token=token,
            revision=revision,
            resume_download=True, # 允许断点续传
            # 可以添加 ignore_patterns 来排除特定文件，例如：
            # ignore_patterns=["*.safetensors", "*.bin.index.json"],
            # allow_patterns=["*.bin", "*.json", "*.txt"] # 或者只允许特定文件
        )

        print(f"模型 '{repo_id}' 已成功下载到: {downloaded_path}")
        print(f"所有文件已保存在目录: {os.path.abspath(local_dir)}")

    except Exception as e:
        print(f"下载模型 '{repo_id}' 时发生错误: {e}")
        print("请检查以下几点：")
        print(f"1. 模型 ID '{repo_id}' 是否正确？")
        print("2. 您是否有网络连接？")
        print("3. 如果模型是私有的或需要门控访问，您是否提供了有效的 token？")
        print(f"4. 目标目录 '{local_dir}' 是否具有写入权限？")

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="从 Hugging Face Hub 下载模型。")
    parser.add_argument(
        "repo_id",
        type=str,
        help="要下载的 Hugging Face 模型/仓库 ID (例如 'bert-base-uncased')"
    )
    parser.add_argument(
        "local_dir",
        type=str,
        help="模型文件保存的本地目录路径。"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="可选的 Hugging Face API token (用于私有模型)。"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="可选的模型版本 (分支名, 标签, 或 commit hash)。"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用下载函数
    download_model_from_huggingface(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        token=args.token,
        revision=args.revision
    )

# 如何运行脚本的示例:
# 1. 下载公开模型:
#    python your_script_name.py bert-base-uncased ./models/bert-base-uncased
#
# 2. 下载特定版本的公开模型:
#    python your_script_name.py gpt2 ./models/gpt2 --revision block_sparse
#
# 3. 下载私有模型 (需要 token):
#    python your_script_name.py your-username/private-model ./models/private-model --token YOUR_HF_TOKEN
#    或者先通过 huggingface-cli login 登录
#
# 4. 下载 Llama-2 模型 (需要 token 并且已获得 Meta 授权):
#    python your_script_name.py meta-llama/Llama-2-7b-chat-hf ./models/Llama-2-7b-chat-hf --token YOUR_HF_TOKEN
