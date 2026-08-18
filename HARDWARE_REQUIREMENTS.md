# 硬件需求与车机选型说明

> 本文档说明车载座舱 RAG 问答系统（离线推理）的模型体积、内存/显存需求及车机硬件建议。
> 配套代码配置见 `config.py`；详细架构见 `README_internal.md`。

---

## 一、各模型体积明细（权重大小）

| 模型 | 用途 | 参数量 | fp32 | fp16/bf16 | int8 量化 | 磁盘下载 |
|---|---|---|---|---|---|---|
| bge-small-zh-v1.5 | 向量检索 + 意图分类底模 | 24M | ~100MB | ~50MB | — | ~100MB |
| bge-reranker-v2-m3 | 精排（CrossEncoder） | 568M | **~2.3GB** | ~1.1GB | — | ~2.3GB |
| Qwen2.5-0.5B-Instruct | LLM 生成 | 0.5B | ~2GB | **~1GB** | ~0.4GB | ~1GB |
| FunASR paraformer-large | 语音识别 | 220M | ~880MB | ~440MB | ~110MB | ~900MB |
| 意图分类微调头 | 4 分类 | 24M+ | ~100MB | ~50MB | — | ~100MB |
| **合计** | | | **~5.4GB** | **~3.7GB** | **~2.5GB** | **~4.5GB** |

> 当前 `config.py` 实际配置：LLM 用 fp16（`LOAD_4BIT=False`）、ASR 用 fp32（`ASR_MODEL_QUANTIZE=False`）、重排与 embedding 用 fp32 —— 权重合计约 **4.3GB**。

## 二、运行时显存需求（GPU 峰值）

权重只是底数，还需叠加运行时开销（重排激活、LLM KV cache、音频特征、中间张量）：

| 配置 | 权重 | 运行时峰值 | 合计 |
|---|---|---|---|
| **当前配置**（LLM fp16，其余 fp32） | ~4.3GB | ~2.5GB | **≈ 7~8GB 显存** |
| 低配裁剪（LLM 4bit + 重排 fp16 + ASR int8） | ~2.5GB | ~1.5GB | **≈ 3~4GB** |

服务器（23.5G 显卡）实测进程峰值占用约 **9.6GB**，24G 卡跑当前配置绰绰有余。

## 三、内存（RAM）与磁盘

- **系统内存**：模型先加载到 RAM 再上 GPU/统一内存，加上 Chroma 向量库与系统开销 → **建议 16GB**（8GB 需低配裁剪配置）
- **磁盘**：模型约 4.5GB + 项目数据（MinerU 图片/PDF）+ 系统 → **预留 10GB 以上**
- 向量库本身很小：2134 条 × 384 维 ≈ 3MB，忽略不计

## 四、车机硬件建议（三档）

| 档位 | 硬件示例 | 配置 | 预期体验 |
|---|---|---|---|
| **入门** | Jetson Orin Nano 8GB / NX 8GB | LLM 4bit + 跳过重排（`retrieve_by_rrf`）+ ASR int8 | 能跑，问答延迟 3~5s，启动慢 |
| **推荐 ⭐** | Jetson Orin NX 16GB / AGX 16GB / 8G 显卡车机 | 当前配置（fp16 LLM + 全量重排） | 问答 1~3s，体验良好 |
| **舒适** | Orin AGX 32/64GB / 24G 显卡 | 全 fp16 + Flash Attention | 延迟 <1s，可上 1.8B 模型 |

> 车机选型核心指标：**统一内存 ≥16GB**（Jetson CPU/GPU 共享内存，最重要）、**GPU 算力 ≥100 TOPS**（Orin NX 级别）、**功耗/散热**（Orin NX 15~25W，符合车规）、**模型加载启动时间**（4 个模型顺序加载约 30s~2min，可用 `start_service.sh` 后台预热）。

## 五、显存不足时的裁剪优先级（按省显存效果排序）

1. **砍掉重排器**（省 ~2.3GB，效果最大）：`rerank.py` 已有现成流水线 `retrieve_by_rrf`，只做 RRF 融合不跑 CrossEncoder
2. **LLM 4bit 量化**（省 ~0.7GB）：`config.py` 开 `LOAD_4BIT=True`
3. **ASR int8 量化**（省 ~0.8GB）：`ASR_MODEL_QUANTIZE=True`（注意需配合 `main_model.float()` 修复版使用，规避 dtype 问题）
4. **调小检索量**：`TOP_K_DENSE/BM25=5`、`TOP_K_RRF=8`，减少重排批次数

四步全做可从 7~8GB 压到 **3~4GB**，8GB 车机即可跑完整链路。
