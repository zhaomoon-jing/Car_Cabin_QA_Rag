import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import QA_TRAIN_PATH

# 读取QA数据（兼容两种格式：整体JSON数组 或 逐行JSONL）
def load_qa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.startswith("["):          # 整体数组：[{...},{...}]
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]  # 逐行JSONL

if not os.path.exists(QA_TRAIN_PATH):
    print(f"[错误] QA训练数据不存在：{QA_TRAIN_PATH}")
    print("请先准备 data/qa_train/train.jsonl（格式：[{\"question\":\"...\",\"answer\":\"...\"}, ...]）")
    sys.exit(1)

data = load_qa_data(QA_TRAIN_PATH)

# 校验格式
err_count = 0
for idx, item in enumerate(data):
    if "question" not in item or "answer" not in item:
        print(f"第{idx}条字段缺失：{item}")
        err_count += 1
        continue
    if len(item["question"]) < 2 or len(item["answer"]) < 5:
        print(f"第{idx}条内容过短：{item}")
        err_count += 1

print(f"总问答条数：{len(data)}，错误条数：{err_count}")