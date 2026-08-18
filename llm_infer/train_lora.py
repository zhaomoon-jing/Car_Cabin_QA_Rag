import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import LLM_MODEL_NAME, QA_TRAIN_PATH, LORA_WEIGHT_DIR


import os
# 线上仓库地址 LLM_MODEL_NAME 
# 国内镜像（防止下载失败，放在最前面）
cache_dir = "/root/autodl-tmp/models"
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#os.environ["HF_HUB_TIMEOUT"] = "300"
# 本地存放路径
name_lst = LLM_MODEL_NAME.split("/")
model_format_name= "models--"+  name_lst[0] +"--"+ name_lst[1]
local_dir = Path("/root/autodl-tmp/models/"+ model_format_name)
#modelsope 下载
from modelscope import snapshot_download


# 加载训练QA数据（兼容两种格式：整体JSON数组 或 逐行JSONL）
def load_qa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.startswith("["):          # 整体数组：[{...},{...}]
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]  # 逐行JSONL

if not os.path.exists(QA_TRAIN_PATH):
    raise FileNotFoundError(
        f"QA训练数据不存在：{QA_TRAIN_PATH}\n"
        "请先准备 data/qa_train/train.jsonl（格式：[{\"question\":\"...\",\"answer\":\"...\"}, ...]）"
    )
qa_data = load_qa_data(QA_TRAIN_PATH)
if not qa_data:
    raise SystemExit("QA训练数据为空，请检查 train.jsonl 内容")
ds = Dataset.from_list(qa_data)

# 量化
'''
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
'''
# 判断本地是否存在完整模型（核心：检测config.json）
config_paths = list(local_dir.rglob("config.json"))
if local_dir.exists() and config_paths[0].exists():
    print(f"✅ 检测到本地模型，离线加载：{local_dir}")
    real_path = config_paths[0].parent
    tokenizer = AutoTokenizer.from_pretrained(
        str(real_path),
        trust_remote_code=True,
        local_files_only=True  # 强制禁止联网
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(real_path),
        device_map="auto", 
        trust_remote_code=True,
        local_files_only=True
    )
else:
    print(f"⚠️ 本地模型不存在，从modelscope下载：{LLM_MODEL_NAME}")
    model_dir = snapshot_download(LLM_MODEL_NAME,cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,  device_map="auto", trust_remote_code=True
    )
'''
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
'''
# LoRA配置
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# 格式化对话
def format_prompt(sample):
    text = f"""<|im_start|>system
你是车载座舱助手，仅依据车辆手册回答，禁止编造信息。
<|im_end|>
<|im_start|>user
{sample["question"]}<|im_end|>
<|im_start|>assistant
{sample["answer"]}<|im_end|>"""
    # 包成列表返回！！
    return [text]

from transformers import TrainingArguments
train_args_dict={
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "num_train_epochs": 4,
    "output_dir": LORA_WEIGHT_DIR,
    "logging_steps": 10,
    "fp16": True,   # bf16/fp16加速训练
    "save_strategy": "epoch",
    "optim": "adamw_torch"
}
training_args = TrainingArguments(**train_args_dict)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    tokenizer=tokenizer,
    formatting_func=format_prompt,
    max_seq_length=512,
    args=training_args
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(LORA_WEIGHT_DIR)
    print("LoRA微调完成，权重保存至", LORA_WEIGHT_DIR)