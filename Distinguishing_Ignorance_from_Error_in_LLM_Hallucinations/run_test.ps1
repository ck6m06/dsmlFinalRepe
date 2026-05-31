$ErrorActionPreference = "Stop"

Set-Location "d:\dsmlFinalRepe\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations"

# .\run_test.ps1 <-- run 這個檔案的指令

# 替換 eva_results 資料集位置 資料集長相可以參考 experiment/type1_results/all239/type1_dataset_239.json
# 替換 directions 向量位置
# oput_json 是輸出檔案位置 可以看一下 現在檔案輸出的範例，response 是要在哪去 ai judge 的 

Write-Host "open question type1 intervention" -ForegroundColor Cyan
python experiment\run_intervention_aIjudge.py `
  --eval_results experiment/type2_results/all445/open_book_eval_results_445.json `
  --directions_dir experiment/type2_results/all445/meta-llama_Llama-3.2-1B-Instruct/directions `
  --model_name meta-llama/Llama-3.2-1B-Instruct `
  --vector_type residual --method mean_diff `
  --layer 2 --alpha 1.0 `
  --output_json experiment\open_question\result1111.json `
  --save_only

Write-Host "open question type2 intervention" -ForegroundColor Cyan
python experiment\run_intervention_aIjudge.py `
  --eval_results experiment/type2_results/all445/open_book_eval_results_445.json `
  --directions_dir experiment/type2_results/all445/meta-llama_Llama-3.2-1B-Instruct/directions `
  --model_name meta-llama/Llama-3.2-1B-Instruct `
  --vector_type residual --method mean_diff `
  --layer 9 --alpha 1.0 `
  --output_json experiment\open_question\result2222.json `
  --save_only

Write-Host "All steps done." -ForegroundColor Green