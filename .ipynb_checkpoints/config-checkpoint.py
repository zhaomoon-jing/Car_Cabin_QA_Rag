import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data/raw")
#CHUNK_DATA_PATH = os.path.join(BASE_DIR, "data/raw/chunks/chunks.jsonl")
#QA_TRAIN_PATH = os.path.join(BASE_DIR, "data/qa_train/train.jsonl")
#FAISS_SAVE_DIR = os.path.join(BASE_DIR, "vector_store/faiss_index")

RAW_OUTPUT_ROOT = os.path.join(BASE_DIR, "data/raw_output/")

# 统一分块数据（BM25 与 Chroma 共用同一份切分结果，保证两路索引同源、RRF可对齐）
CHUNKS_JSONL_PATH = os.path.join(BASE_DIR, "data/chunks/chunks.jsonl")
# 分块参数（与 build_chunk_chroma.py 保持一致：chunk_size=600, overlap=120）
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

LOCAL_ROOT = "/root/autodl-tmp/models"
# 分块参数
#CHUNK_SIZE = 300
#CHUNK_OVERLAP = 50

# 向量模型
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 向量数据库
MINERU_DIR = "data/raw_output"
# 2. Chroma向量数据库持久化保存路径
CHROMA_PERSIST_PATH = "chroma_car_db"
# 3. 向量集合名称
COLLECTION_NAME = "car_manual"
# 4. 本地离线embedding模型



#向量检索超参数
TOP_K_DENSE = 5
#意图分类
#INTENT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INTENT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 检索参数
TOP_K_BM25 = 10
TOP_K_DENSE = 10
TOP_K_RERANK = 3

# 新增RRF粗筛数量
TOP_K_RRF = 10

# LLM配置（轻量化Qwen0.5B）
#LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOAD_4BIT = False
LORA_WEIGHT_DIR = os.path.join(BASE_DIR, "llm_infer/lora_car")

# ASR
ASR_MODEL_NAME = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  # paraformer-large
#ASR_MODEL_QUANTIZE = True   # int8 量化开关（约90MB，显存占用降75%）
ASR_MODEL_QUANTIZE = False  
ASR_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"      # VAD断句
ASR_PUNC_MODEL = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"  # 标点恢复

# 意图分类标签
INTENT_LABELS = [
    "control",    # 座舱控制：空调、车窗、座椅
    "qa_know",    # 车辆知识问答（走RAG）
    "nav",        # 导航
    "chat_none"   # 无关闲聊
]
