$ErrorActionPreference = "Stop"

Set-Location "d:\dsmlFinalRepe\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations"

# Write-Host "Step 1" -ForegroundColor Cyan
python experiment2/split_open_book_eval_results.py `
    --input experiment/withoutinsturct_open_book_eval_3bresults.json `
    --output experiment/3b_type2_results/73traintest/open_book_eval_results_balanced.json


Write-Host "Step 1-1: extract hidden states (all)" -ForegroundColor Cyan
  python experiment/extract_open_book_hidden_states.py `
    --eval_results experiment/3b_type2_results/73traintest/73traintest_train.json `
    --output_dir experiment/3b_type2_results/73traintest `
    --model_name meta-llama/Llama-3.2-3B-Instruct

Write-Host "Step 1-2: compute directions" -ForegroundColor Cyan
  python experiment/compute_open_book_directions.py `
    --input_dir experiment/3b_type2_results/73traintest/meta-llama_Llama-3.2-3B-Instruct `
    --method mean_diff

Write-Host "Step 1-3: all1520 sweep" -ForegroundColor Cyan
  python experiment/run_open_book_intervention.py `
    --eval_results experiment/3b_type2_results/73traintest/73traintest_train.json `
    --extra_test experiment/3b_type2_results/73traintest/73traintest_test.json `
    --directions_dir experiment/3b_type2_results/73traintest/meta-llama_Llama-3.2-3B-Instruct/directions `
    --model_name meta-llama/Llama-3.2-3B-Instruct `
    --vector_type residual `
    --layer sweep `
    --limit 1 `
    --output_json experiment/3b_type2_results/73traintest/trainresult.json `
    --extra_output_json experiment/3b_type2_results/73traintest/testResult.json

# Write-Host "Step 1-1: extract hidden states (all)" -ForegroundColor Cyan
# python experiment/extract_open_book_hidden_states.py `
#   --eval_results experiment/type2_results/all445/open_book_eval_results_445.json `
#   --model_name meta-llama/Llama-3.2-1B-Instruct

# Write-Host "Step 1-2: compute directions" -ForegroundColor Cyan
# python experiment/compute_open_book_directions.py `
#   --input_dir experiment/type2_results/all445/meta-llama_Llama-3.2-1B-Instruct `
#   --method mean_diff

# Write-Host "Step 1-3: all445 sweep" -ForegroundColor Cyan
# python experiment/run_open_book_intervention.py `
#   --eval_results experiment/type2_results/all445/open_book_eval_results_445.json `
#   --directions_dir experiment/type2_results/all445/meta-llama_Llama-3.2-1B-Instruct/directions `
#   --model_name meta-llama/Llama-3.2-1B-Instruct `
#   --vector_type residual `
#   --layer sweep `
#   --output_json experiment/type2_results/all445/result445.json 

# Write-Host "Step 2-1: extract hidden states (train)" -ForegroundColor Cyan
# python experiment/extract_open_book_hidden_states.py `
#   --eval_results experiment/type2_results/trainTest445/open_book_eval_results_50_train.json `
#   --output_dir experiment/type2_results/trainTest445 `
#   --model_name meta-llama/Llama-3.2-1B-Instruct

# Write-Host "Step 2-2: compute directions" -ForegroundColor Cyan
# python experiment/compute_open_book_directions.py `
#   --input_dir experiment/type2_results/trainTest445/meta-llama_Llama-3.2-1B-Instruct `
#   --method mean_diff

# Write-Host "Step 2-3: train sweep + extra test intervention" -ForegroundColor Cyan
# python experiment/run_open_book_intervention.py `
#   --eval_results experiment/type2_results/trainTest445/open_book_eval_results_50_train.json `
#   --extra_test experiment/type2_results/trainTest445/open_book_eval_results_50_test.json `
#   --directions_dir experiment/type2_results/trainTest445/meta-llama_Llama-3.2-1B-Instruct/directions `
#   --model_name meta-llama/Llama-3.2-1B-Instruct `
#   --vector_type residual `
#   --layer sweep `
#   --output_json experiment/type2_results/trainTest445/trainResult.json `
#   --extra_output_json experiment/type2_results/trainTest445/testResult.json

Write-Host "All steps done." -ForegroundColor Green