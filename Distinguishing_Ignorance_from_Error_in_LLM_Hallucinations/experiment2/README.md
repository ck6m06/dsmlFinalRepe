# experiment2

This folder contains the train/val/test split pipeline for the open-book CAA / hallucination probe experiments.

Files:
- `create_splits.py`: split the merged eval JSON into train/val/test.
- `slice_hidden_vectors_by_split.py`: slice the existing extracted hidden-state `.npy` files into split-specific folders.
- `train_caa_svm.py`: train a linear SVM on train split vectors, select the best layer on val, and report test metrics.

Typical flow:

```powershell
python experiment2\create_splits.py `
  --input experiment\type2_results\open_book_eval_results.json `
  --out_prefix experiment2\datasets\open_book_eval_results `
  --seed 42 --train_frac 0.7 --val_frac 0.15 --test_frac 0.15 `
  --stratify_key correct

python experiment2\slice_hidden_vectors_by_split.py `
  --split_json experiment2\datasets\open_book_eval_results_splits.json `
  --vector_dir experiment\type2_results\meta-llama_Llama-3.2-1B-Instruct `
  --output_dir experiment2\hidden_vectors\open_book_eval_results

python experiment2\train_caa_svm.py `
  --split_root experiment2\hidden_vectors\open_book_eval_results `
  --vector_type residual `
  --output_dir experiment2\caa_svm\open_book_eval_results
```
