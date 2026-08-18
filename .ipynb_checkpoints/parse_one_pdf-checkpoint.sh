#!/bin/bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# =========MinerU模型源：强制使用modelscope魔搭下载=========
#第一次使用 设置成modelscope
export MINERU_MODEL_SOURCE=modelscope
#export MINERU_MODEL_SOURCE=local
# modelscope的缓存目录（模型下载到这里，不要和huggingface hub目录搞混！）
export MODELSCOPE_CACHE=/root/autodl-tmp/models/modelscope_cache

# 其他旧变量
export HF_HOME=/root/autodl-tmp/models
export HF_ENDPOINT=https://hf-mirror.com
export MINERU_HYBRID_BATCH_RATIO=1
export MINERU_CLEAR_CACHE=true
export OMP_NUM_THREADS=2
export ORT_DISABLE_CPU_AFFINITY=1
export NCCL_IGNORE_DISABLED_P2P=1
export TORCH_NCCL_DISABLE=1
#source /usr/local/miniconda3/etc/profile.d/conda.sh
#conda activate py312

PDF_PATH="$1"
OUT_DIR="$2"

if [ $# -lt 2 ];then
    echo "用法： $0 【pdf完整路径】 【输出目录】"
    exit 1
fi

mkdir -p "${OUT_DIR}"
mineru -p "${PDF_PATH}" -o "${OUT_DIR}" --task doc --low-vram
