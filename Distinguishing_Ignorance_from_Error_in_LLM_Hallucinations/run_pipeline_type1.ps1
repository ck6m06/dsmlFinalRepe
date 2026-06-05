$ErrorActionPreference = "Stop"

Set-Location "d:\dsmlFinalRepe\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations"

# Write-Host "Step 1" -ForegroundColor Cyan
#     $falsePath = 'datasets\HallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json'
#     $n = (Get-Content -Raw $falsePath | ConvertFrom-Json).results.Count
#     python experiment\build_open_book_eval_results_from_readable_datasets.py `
#     --true_results datasets\NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct_readable.json `
#     --false_results $falsePath `
#     --limit_true $n --limit_false $n --sample_mode random --seed 42 `
#     --shuffle_merged --output_json experiment\type1_results\all210\balanced_sample210.json.json

# Write-Host "Step 1-1: extract hidden states (all)" -ForegroundColor Cyan
#   python experiment/extract_open_book_hidden_states.py `
#     --eval_results experiment/type1_results/all210/balanced_sample210.json `
#     --output_dir experiment/type1_results/all210 `
#     --model_name meta-llama/Llama-3.2-1B-Instruct

# Write-Host "Step 1-2: compute directions" -ForegroundColor Cyan
#   python experiment/compute_open_book_directions.py `
#     --input_dir experiment/type1_results/all210/meta-llama_Llama-3.2-1B-Instruct `
#     --method mean_diff

Write-Host "Step 1-3: all210 sweep" -ForegroundColor Cyan
  python experiment/run_open_book_intervention.py `
    --eval_results experiment/type1_results/all210/balanced_sample210.json `
    --directions_dir experiment/type1_results/all210/meta-llama_Llama-3.2-1B-Instruct/directions `
    --model_name meta-llama/Llama-3.2-1B-Instruct `
    --vector_type residual `
    --layer sweep `
    --output_json experiment/type1_results/all210/result210.json

# Write-Host "Step 1-3: all210 sweep" -ForegroundColor Cyan
#   python experiment/run_open_book_intervention.py `
#     --eval_results experiment/type1_results/all210/balanced_sample210.json.json `
#     --directions_dir experiment/type1_results/all210/meta-llama_Llama-3.2-1B-Instruct/directions `
#     --model_name meta-llama/Llama-3.2-1B-Instruct `
#     --vector_type residual `
#     --layer sweep `
#     --output_json experiment/type1_results/all210/result239.json

# Write-Host "Step 2-0: split dataset" -ForegroundColor Cyan
# python experiment2/split_open_book_eval_results.py `
#   --input experiment/type1_results/all210/balanced_sample210.json.json `
#   --output experiment/type1_results/trainTest239/test_50.json `
#   --train_output experiment/type1_results/trainTest239/train_50.json `
#   --target_total 50 --correct_mode random --seed 42

# Write-Host "Step 2-1: extract hidden states (train)" -ForegroundColor Cyan
# python experiment/extract_open_book_hidden_states.py `
#   --eval_results experiment/type1_results/trainTest239/train_50.json `
#   --output_dir experiment/type1_results/trainTest239 `
#   --model_name meta-llama/Llama-3.2-1B-Instruct

# Write-Host "Step 2-2: compute directions" -ForegroundColor Cyan
# python experiment/compute_open_book_directions.py `
#   --input_dir experiment/type1_results/trainTest239/meta-llama_Llama-3.2-1B-Instruct `
#   --method mean_diff

# Write-Host "Step 2-3: train sweep + extra test intervention" -ForegroundColor Cyan
# python experiment/run_open_book_intervention.py `
#   --eval_results experiment/type1_results/trainTest239/train_50.json `
#   --extra_test experiment/type1_results/trainTest239/test_50.json `
#   --directions_dir experiment/type1_results/trainTest239/meta-llama_Llama-3.2-1B-Instruct/directions `
#   --model_name meta-llama/Llama-3.2-1B-Instruct `
#   --vector_type residual `
#   --layer sweep `
#   --output_json experiment/type1_results/trainTest239/trainResult.json `
#   --extra_output_json experiment/type1_results/trainTest239/testResult.json

Write-Host "All steps done." -ForegroundColor Green