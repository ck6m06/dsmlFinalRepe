import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
base_model = "meta-llama/Llama-3.2-1b-Instruct"

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# load and quantize the model 
base_model = AutoModelForCausalLM.from_pretrained(base_model, device_map = 'auto')
# base_model_bnb_4b = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map = 'auto')
print(base_model)
print(base_model.config)




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