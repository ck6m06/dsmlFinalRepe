"""Generate open-book context paragraphs for hallucination QA records.

This script reads the readable hallucination dataset JSON, uses a language model
to generate a short Wikipedia-style paragraph that supports the answer, and
writes a new JSON file with the generated context added to each record.

Example:
    python experiment/generate_open_book_contexts.py --input datasets/HallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.1-8b-Instruct_readable.json --output datasets/HallucinateTrivia_qa_with_context_meta-llama_Llama-3.1-8b-Instruct.json --model_name meta-llama/Llama-3.1-8b-Instruct --limit 1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8b-Instruct"


def load_records(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(data)}")
    if not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected each item in {input_path} to be an object/dict")
    return data


def extract_question(prompt_text: str) -> str:
    if "\nanswer:" in prompt_text:
        return prompt_text.split("\nanswer:", 1)[0].strip()
    return prompt_text.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def load_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16
    # ).to("cuda")

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    
    model.eval()
    return model, tokenizer


def get_model_input_device(model) -> torch.device:
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def build_generation_prompt(question: str, answer: str) -> str:
    return (
        "You are writing a Wikipedia-style paragraph for an open-book QA context.\n\n"
        "You MUST:\n"
        "- include factual information that directly supports the answer\n"
        "- include the exact answer string somewhere in the paragraph (not as a standalone sentence)\n"
        "- avoid adding any unrelated or distracting information\n"
        "- avoid phrases like 'the answer is' or 'answer:'\n"
        "- write 2 to 3 sentences\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Wikipedia-style paragraph:"
    )


def build_without_instruct_prompt(context_lines: list[str], few_shots: str, question: str) -> str:
    parts: list[str] = []
    parts.extend(line for line in context_lines if line.strip())
    if few_shots.strip():
        parts.append(few_shots.strip())
    parts.append(f"{question}\nAnswer:")
    return "\n\n".join(parts)


def format_prompt(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You write short factual background paragraphs for evaluation prompts.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Some tokenizers expose apply_chat_template but aren't configured with
            # a chat template (e.g., simple GPT-2 tokenizers). Fall back to the
            # raw user prompt in that case to avoid crashing generation.
            return user_prompt
    return user_prompt


def generate_context(
    model,
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    generation_prompt = build_generation_prompt(question, answer)
    formatted_prompt = format_prompt(tokenizer, generation_prompt)
    device = get_model_input_device(model)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return generated


def answer_mentioned(context: str, answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_context = normalize_text(context)
    return bool(normalized_answer) and normalized_answer in normalized_context


def augment_records(
    records: list[dict[str, Any]],
    model,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **_kwargs):
            return iterable

    augmented: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Generating contexts"):
        prompt_text = str(record.get("prompt", ""))
        few_shots = str(record.get("few_shot_prompt", ""))[:str(record.get("few_shot_prompt", "")).rfind(prompt_text)]
        answer = str(record.get("reference_answer", "")).strip()
        question = extract_question(prompt_text)

        context = generate_context(
            model=model,
            tokenizer=tokenizer,
            question=question,
            answer=answer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # If the model omitted the answer entirely, do one stronger retry.
        if not answer_mentioned(context, answer):
            stronger_context = generate_context(
                model=model,
                tokenizer=tokenizer,
                question=question,
                answer=answer,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            if answer_mentioned(stronger_context, answer):
                context = stronger_context

        augmented_record = dict(record)
        augmented_record["generated_context"] = context
        augmented_record["generated_context_model"] = model.config._name_or_path if hasattr(model, "config") else "unknown"
        # Build an evaluation-style prompt that wraps the generated context into numbered evidence
        # and asks the downstream model to answer using only that context.
        # Split the generated context into 2-3 sentence context items.
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
        if not sentences:
            sentences = [context.strip()] if context.strip() else [""]
        # take up to 1 context sentence (prefer a sentence that contains the answer)
        max_ctx = 1  # start with 1 sentence; can try 2-3 later
        ctx_lines = []
        if max_ctx == 1:
            # prefer a sentence that contains the answer; fall back to the first sentence
            chosen_idx = None
            for i, s in enumerate(sentences):
                if answer_mentioned(s, answer):
                    chosen_idx = i
                    break
            if chosen_idx is None:
                chosen_idx = 0
            s = sentences[chosen_idx]
            s_clean = s.replace('\n', ' ').strip()
            ctx_lines.append(f"[1] {s_clean}")
        else:
            for i, s in enumerate(sentences[:max_ctx]):
                # ensure no stray newlines
                s_clean = s.replace('\n', ' ').strip()
                ctx_lines.append(f"[{i+1}] {s_clean}")

        eval_prompt = (
            "Instruction:\n"
            "Answer the final question using only the provided context. Do NOT use outside knowledge.\n"
            "If the answer cannot be determined from the context, reply exactly: \"I don't know\".\n\n"
            "### Context:\n"
            + "\n".join(ctx_lines)
            + "\n\n"
            "### Examples:\n"
            f"{few_shots}\n\n"
            "### Question:\n"
            f"{question}\n\n"
            "### Answer:"
        )

        without_instruct_prompt = build_without_instruct_prompt(
            context_lines=ctx_lines,
            few_shots=few_shots,
            question=question,
        )

        augmented_record["generated_context_prompt"] = eval_prompt
        augmented_record["without_instruct_prompt"] = without_instruct_prompt
        augmented_record["generated_context_contains_answer"] = answer_mentioned(context, answer)
        augmented.append(augmented_record)

    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Wikipedia-style open-book contexts for hallucination QA records."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input readable hallucination JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file that will include generated_context for each record.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model used to generate context paragraphs.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick testing. Use 0 to process the whole file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    print(f"Loading records from {input_path}...\n")
    output_path = Path(args.output).resolve()
    print(f"Will write augmented records to {output_path}...\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    print(f"Loaded {len(records)} records from {input_path}")
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    augmented = augment_records(
        records=records,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(augmented, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(augmented)} records to {output_path}")


if __name__ == "__main__":
    main()