import json
import random
from pathlib import Path

from transformers import AutoConfig

# 這裡以 Llama-3.2-1B-Instruct 為例
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

print(f"總層數 (Layers): {config.num_hidden_layers}")          # 16
print(f"每層 Q-Heads: {config.num_attention_heads}")         # 16
print(f"每層 KV-Heads: {config.num_key_value_heads}")        # 4

# base_dir = Path(__file__).resolve().parent
# json_path = base_dir / "type1_results" / "all239" / "result239_with_source.json"

# random.seed(42)

# # 1. 讀資料
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# rows = data.get("baseline_rows", [])

# # 2. 分類
# false_rows = [row for row in rows if row.get("correct") is False]
# true_rows = [row for row in rows if row.get("correct") is True]

# # 3. 目標數量
# n = len(false_rows)

# # 4. 隨機抽 true
# sample_true_rows = random.sample(true_rows, min(n, len(true_rows)))

# # 5. 合併
# output_rows = false_rows + sample_true_rows

# # （可選）打亂順序，避免 model bias
# random.shuffle(output_rows)

# # 6. 輸出 JSON
# output = {
#     "false_count": len(false_rows),
#     "sample_true_count": len(sample_true_rows),
#     "total": len(output_rows),
#     "rows": output_rows
# }

# out_path = base_dir / "balanced_sample.json"
# with open(out_path, "w", encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False, indent=2)

# print(f"Saved to: {out_path}")
# print(f"False: {len(false_rows)}, Sample True: {len(sample_true_rows)}")

# # 條件 B：source_correct 是 True 且 correct 也是 True (使用 and 串聯)
# both_true_count = sum(
#     1
#     for row in rows
#     if row.get("source_correct") is True and row.get("correct") is True
# )
# # 條件 A：只有 source_correct 是 True
# source_false_count = sum(1 for row in rows if row.get("source_correct") is False)

# # 條件 B：source_correct 是 True 且 correct 也是 True (使用 and 串聯)
# both_false_count = sum(
#     1
#     for row in rows
#     if row.get("source_correct") is False and row.get("correct") is True
# )
# print(f"1. source_correct 為 True 的總筆數: {source_true_count}")
# print(f"2. source_correct 為 True 且 correct 亦為 True 的筆數: {both_true_count}")

# print(f"1. source_correct 為 False 的總筆數: {source_false_count}")
# print(f"2. source_correct 為 False 且 correct 亦為 True 的筆數: {both_false_count}")

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
# base_model = "meta-llama/Llama-3.2-1b-Instruct"

# # load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_model)

# # bnb_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=torch.bfloat16
# # )

# # load and quantize the model 
# base_model = AutoModelForCausalLM.from_pretrained(base_model, device_map = 'auto')
# # base_model_bnb_4b = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map = 'auto')
# print(base_model)
# print(base_model.config)




# LlamaForCausalLM(
#   (model): LlamaModel(
#     (embed_tokens): Embedding(128256, 2048)
#     (layers): ModuleList(
#       (0-15): 16 x LlamaDecoderLayer(
#         (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=2048, out_features=512, bias=False)
#           (v_proj): Linear(in_features=2048, out_features=512, bias=False)
#           (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
#         (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
#       )
#     )
#     (norm): LlamaRMSNorm((2048,), eps=1e-05)
#     (rotary_emb): LlamaRotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
# )

# from transformers import AutoModelForCausalLM

# MODEL_NAME = "meta-llama/Llama-3.2-1b-Instruct"  # 換成你的 model id
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
# cfg = model.config

# print("Config summary:")
# print(" num_hidden_layers:", getattr(cfg, "num_hidden_layers", None))
# print(" hidden_size:", getattr(cfg, "hidden_size", None))
# print(" num_attention_heads:", getattr(cfg, "num_attention_heads", None))
# print(" other config keys available:", list(cfg.to_dict().keys())[:30])

# # 找出模型中哪個屬性是 layer stack（常見 name: model.layers / model.decoder.layers）
# layer_containers = []
# for attr in ["model.layers", "model.decoder.layers", "model.transformer.h", "model.model.layers"]:
#     try:
#         obj = eval("model." + attr)
#         layer_containers.append((attr, obj))
#     except Exception:
#         pass

# print("\nDetected layer containers:")
# for name, obj in layer_containers:
#     try:
#         n = len(obj)
#     except Exception:
#         n = None
#     print(f" {name}  (len={n})  type={type(obj)}")

# # If we found a container, inspect the first block's children
# if layer_containers:
#     name, container = layer_containers[0]
#     print(f"\\nInspecting one block from {name}:")
#     block = container[0]
#     for child_name, child in block.named_children():
#         print(f"  {child_name}: {type(child)}")
#     # show deeper keys for typical MLP/Attention modules
#     print("\nNamed modules inside block (partial):")
#     for n, m in list(block.named_modules())[:60]:
#         print(n)
# else:
#     print("\nNo obvious layer container found; use model.named_modules() to search.")
#     for n, m in model.named_modules():
#         if 'layer' in n or 'block' in n or 'mlp' in n or 'attn' in n or 'feed_forward' in n:
#             print(n, type(m))