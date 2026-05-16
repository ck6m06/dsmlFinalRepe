# 實驗說明

本資料夾用於針對 `meta-llama/Llama-3.2-1B-Instruct` 進行快速的 prompt 測試與回應收集。

## 內容檔案

- `prompts.txt`：每個 prompt 區塊以空行分隔（每段視為一個獨立 prompt）。
- `run_llama3_prompt_test.py`：載入模型並列印／儲存回應的執行腳本。
- `responses.jsonl`：執行腳本產生的輸出檔（JSONL 格式，每行為一筆回應）。

## 執行範例

預設執行：

```bash
python experiment/run_llama3_prompt_test.py
```

可透過命令列參數覆寫預設設定，例如：

```bash
python experiment/run_llama3_prompt_test.py --prompt_file experiment/prompts.txt --output_file experiment/responses.jsonl --max_new_tokens 128 --temperature 0.7 --top_p 0.9
```

## Prompt 格式範例

在 `prompts.txt` 中，每個以空行分隔的段落會被當作一個 prompt，例如：

```text
Write a short answer to this question.

Question: What is the capital of France?

Question: Explain hallucination in one paragraph.
```

## 注意事項

- 請確認已安裝並設定好所需的模型與執行環境（參考專案根目錄的 `environment.yml` 或 `README.md`）。
- `responses.jsonl` 為 JSONL 格式，便於後續分析與匯入。
- 若需不同生成參數，請使用命令列參數來覆寫腳本內的預設值。

如需我幫你把 README 翻成英文版或增加更詳細的執行範例（例如環境安裝步驟），告訴我即可。
