import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_PROMPT_FILE = Path(__file__).with_name("prompts.txt")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("responses.jsonl")


def load_prompts(prompt_file: Path) -> list[str]:
    if not prompt_file.exists():
        return []

    raw_text = prompt_file.read_text(encoding="utf-8")
    prompts = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    return prompts


def format_prompt(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{user_prompt}\n<|assistant|>\n"


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    formatted_prompt = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    response_tokens = generated_ids[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompts against meta-llama/Llama-3.2-1B-Instruct.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt_file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument("--output_file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output_file)
    prompts = load_prompts(prompt_path)

    if not prompts:
        raise SystemExit(f"No prompts found in {prompt_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, start=1):
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            record = {
                "index": index,
                "prompt": prompt,
                "response": response,
                "model_name": args.model_name,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            print(f"[{index}] {response}\n")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
