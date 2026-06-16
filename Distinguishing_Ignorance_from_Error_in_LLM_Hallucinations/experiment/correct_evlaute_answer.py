import re
import json
import unicodedata
from word2number import w2n
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 1. NUMBER NORMALIZATION
# =========================
def normalize_number(text):
    try:
        return str(w2n.word_to_num(text.lower()))
    except:
        return text


# =========================
# 2. FULL TEXT NORMALIZATION
# =========================
def normalize(text):
    if text is None:
        return ""

    text = text.lower().strip()

    # remove accents (Nîmes → Nimes)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # number normalization (four → 4)
    text = normalize_number(text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # normalize whitespace
    text = " ".join(text.split())

    return text


# =========================
# 3. EXACT MATCH CHECK
# =========================
def exact_match(ref, pred):
    return normalize(ref) == normalize(pred)


# =========================
# 4. LLM JUDGE PROMPT
# =========================
SYSTEM_PROMPT = """
You are a strict factual correctness evaluator.
Decide if the prediction refers to the SAME real-world entity as the reference.

Rules:
- Same entity = True
- Different entity = False
- Spelling/number/diacritics/word order differences are OK
- Abbreviations, full names, and aliases (e.g., USA vs. The United States) are OK
- Category similarity is NOT enough

Return ONLY True or False.

Examples:
Reference: 26th December
Prediction: December 26th
Answer: True

Reference: USA
Prediction: The United States
Answer: True

Reference: {reference}
Prediction: {prediction}

Answer:
"""
JUDGE_PROMPT = """Reference: {reference}
Prediction: {prediction}

Answer:"""

# =========================
# 5. LOAD MODEL
# =========================
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    )

    model.eval()
    return model, tokenizer


# =========================
# 6. LLM JUDGE (fallback only)
# =========================
def llm_judge(model, tokenizer, ref, pred):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": JUDGE_PROMPT.format(reference=ref, prediction=pred)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

    true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
    false_id = tokenizer.encode(" False", add_special_tokens=False)[0]

    probs = torch.softmax(logits, dim=-1)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=0
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"decoded: {decoded}")

    input_len = inputs["input_ids"].shape[1]

    answer = tokenizer.decode(
        output[0][input_len:],
        skip_special_tokens=True
    ).strip()

    print("Answer only:", answer)
    

    return probs[0, true_id] > probs[0, false_id]


# =========================
# 7. HYBRID EVALUATOR
# =========================
def is_correct(model, tokenizer, ref, pred):

    # Step 1: rule-based exact match
    if exact_match(ref, pred):
        return True

    # Step 2: short-circuit obvious mismatch
            print(f"[{i}] Ref: {ref}")
        print(f"     Pred: {pred}")
        print(f"     Correct: {result}")
        print("-" * 50)
    normalized_ref = normalize(ref)
    normalized_pred = normalize(pred)
    print(f"Normalized Ref: {normalized_ref}"
          f"\nNormalized Pred: {normalized_pred}")
    if normalized_ref and normalized_pred and normalized_ref != normalized_pred:
        print("Obvious mismatch detected, marking as incorrect.")
        print("-" * 50)
        
        return False
    # if normalize(ref) != normalize(pred):
    #     # Step 3: LLM fallback only for ambiguous cases
    #     return llm_judge(model, tokenizer, ref, pred)

    return True


# =========================
# 8. MAIN EVAL LOOP
# =========================
def evaluate(dataset_path, model_name):

    model, tokenizer = load_model(model_name)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data["results"])
    correct = 0

    for i, r in enumerate(data["results"]):
        
        ref = r.get("reference_answer", "")
        pred = r.get("prediction", "").split("\n")[0].strip()

        result = is_correct(model, tokenizer, ref, pred)

        r["final_correct"] = result

        if result:
            correct += 1



    acc = correct / total

    print("\n====================")
    print("Final accuracy:", acc)
    print("====================")

    return data


# =========================
# 9. RUN
# =========================
dataset_path = r"D:\dsmlFinalRepe\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations\experiment\withoutInsturct_open_book_eval_3bresults.json"

model_name = "meta-llama/Llama-3.2-3B-Instruct"

result_data = evaluate(dataset_path, model_name)

with open(dataset_path.replace(".json", "_final_eval.json"), "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False, indent=2)

# from __future__ import annotations

# import argparse
# import json
# import re
# from pathlib import Path
# from typing import Any

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer


# def load_model_and_tokenizer(model_name: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"

#     if torch.cuda.is_available():
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             # dtype=torch.float16,
#             low_cpu_mem_usage=True,
#         )
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             low_cpu_mem_usage=True,
#         )

#     model.eval()
#     return model, tokenizer

# path = r"D:\dsmlFinalRepe\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations\experiment\withoutInsturct_open_book_eval_3bresults.json"

# with open(path, "r", encoding="utf-8") as f:
#     data = json.load(f)


# # 挑出是 false 的結果去 叫模型判斷一下 有沒有對
# for result in data["results"]:
#     if result.get("correct") == False:
#         reference = result.get("reference_answer", "")
#         prediction = result.get("prediction", "").split("\n")[0].strip()
#         print("reference:", reference)
#         print("prediction:", prediction)
#         print("-"*50)


# print(data["model_name"])
# print(data["accuracy"])
# print(len(data["results"]))