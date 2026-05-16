# Experiment

This folder is for quick prompt tests with `meta-llama/Llama-3.2-1B-Instruct`.

## Files

- `prompts.txt`: write one prompt block per paragraph, separated by a blank line.
- `run_llama3_prompt_test.py`: loads the model and prints/saves responses.
- `responses.jsonl`: generated output file created when you run the script.

## Run

```bash
python experiment/run_llama3_prompt_test.py
```

You can also override the default settings:

```bash
python experiment/run_llama3_prompt_test.py --prompt_file experiment/prompts.txt --output_file experiment/responses.jsonl --max_new_tokens 128 --temperature 0.7 --top_p 0.9
```

## Prompt format

Put prompts in `prompts.txt` like this:

```text
Write a short answer to this question.

Question: What is the capital of France?

Question: Explain hallucination in one paragraph.
```

Each blank-line-separated block is treated as one prompt.
