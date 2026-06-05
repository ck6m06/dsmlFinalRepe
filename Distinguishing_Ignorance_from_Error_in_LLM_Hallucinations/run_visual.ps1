$ErrorActionPreference = "Stop"

Set-Location "d:\dsmlFinalRepe\Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations"

Write-Host "visualizing results type1" -ForegroundColor Cyan
python experiment/visualize_results.py --results_dir experiment/type1_results/all239/result239_layer_summaries --out_dir experiment/plots/result239

Write-Host "visualizing results type2" -ForegroundColor Cyan
python experiment/visualize_results.py --results_dir experiment/type2_results/all445/result445_layer_summaries --out_dir experiment/plots/result445

Write-Host "All steps done." -ForegroundColor Green