# format_model.py
import os
import sys
from pathlib import Path
from modelscope import snapshot_download

# 把项目根目录加入环境变量，保证能正常导入config
sys.path.append(str(Path(__file__).parent.parent))
from config import LOCAL_ROOT, EMBED_MODEL_NAME

def model_name_to_local_dir(model_id: str) -> str:
    """
    转换modelscope模型ID为本地文件夹名：
    BAAI/bge-small-zh-v1.5  →  BAAI--bge-small-zh-v1.5
    """
    return model_id.replace("/", "--")

def find_model_root(model_dir: str, max_depth: int = 8):
    """
    定位模型真实根目录（含模型配置文件的目录），兼容多种本地结构：
    1. ModelScope直下：model_dir/config.json
    2. Hugging快照：model_dir/snapshots/<hash>/config.json
    3. 快照+models子目录：model_dir/snapshots/models/config.json
    4. 更深层嵌套：model_dir/snapshots/<hash>/models/config.json
    递归查找（放宽深度限制），返回找到的目录路径；找不到返回None
    """
    if not os.path.isdir(model_dir):
        return None
    # 直下结构
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir
    # 子目录中递归查找含配置文件的目录
    for root, dirs, files in os.walk(model_dir):
        depth = root[len(model_dir):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        if "config.json" in files or "config.yaml" in files or "config.yml" in files:
            return root
    return None

def check_model_complete(model_dir: str) -> bool:
    """
    校验模型目录是否完整（存在config.json/config.yaml视为完整可用）。
    兼容 config 位于 model_dir 或 model_dir/snapshots/models 等嵌套结构。
    """
    return find_model_root(model_dir) is not None

def get_real_load_path(model_root: str) -> str:
    """
    返回模型真实可加载路径（含配置文件的目录）。
    兼容 ModelScope直下 / Hugging快照(snapshots) / snapshots/models 嵌套结构，
    与 find_model_root 使用同一套定位逻辑，保证 check_model_complete 通过后一定能加载。
    """
    real_path = find_model_root(model_root)
    if real_path is None:
        raise FileNotFoundError(
            f"无效模型目录 {model_root}，递归查找config.json失败（既无直下config.json也无snapshots快照）"
        )
    # 打印真实配置文件路径，便于核对
    for cfg_name in ("config.json", "config.yaml", "config.yml"):
        cfg_path = os.path.join(real_path, cfg_name)
        if os.path.exists(cfg_path):
            print(f"[模型加载] 找到配置文件：{cfg_path}")
            break
    print(f"[模型加载] 使用模型目录：{real_path}")
    return real_path

def get_or_download_model(model_id: str) -> str:
    """
    外部调用入口【核心对外方法】
    :param model_id: ModelScope模型标识，例 "BAAI/bge-small-zh-v1.5"
    :return: 可直接传入embedding函数的本地模型完整路径
    逻辑：
    1. 根据config.LOCAL_ROOT拼接本地文件夹名称
    2. 本地存在完整模型 → 直接返回真实加载路径
    3. 本地缺失 → ModelScope自动下载后再返回路径
    """
    # 1. 拼接本地完整文件夹路径
    folder_name = model_name_to_local_dir(model_id)
    full_model_dir = os.path.join(LOCAL_ROOT, folder_name)
    print(f"[模型校验] 目标本地目录：{full_model_dir}")

    # 2. 判断本地是否已有完整模型
    if check_model_complete(full_model_dir):
        print(f"[模型校验] 本地已存在完整模型，跳过下载")
        return get_real_load_path(full_model_dir)

    # 3. 本地不存在，执行下载
    print(f"[模型下载] 本地无完整模型，开始拉取 {model_id} ...")
    snapshot_download(
        model_id=model_id,
        local_dir=full_model_dir
    )
    print(f"[模型下载] 下载完成，保存至 {full_model_dir}")

    # 4. 返回可加载真实路径
    return get_real_load_path(full_model_dir)

# 仅当前脚本单独运行时测试，被其他脚本import不会执行
if __name__ == "__main__":
    # 测试示例，使用config中默认embedding模型
    model_path = get_or_download_model(EMBED_MODEL_NAME)
    print(f"\n✅ 最终可用模型路径：{model_path}")