import json

# 1. 定義你希望轉換後的 JSON 欄位名稱
field_keys = [
    "prompt",
    "reference_answer",
    "alternative_answer",
    "reference_tokens",
    "alternative_tokens",
    "few_shot_prompt",
    "know_count",
    "rank_diff"
]

# 2. 讀取原始的 nested list JSON 檔案
# 請將 'input_nested_list.json' 替換成你實際的檔名
try:
    with open('datasets/NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-3B-Instruct.json', 'r', encoding='utf-8') as f:
        nested_list_data = json.load(f)
    
    print(f"成功讀取檔案，共有 {len(nested_list_data)} 筆資料開始轉換...")

    # 3. 進行結構轉換：將內層的 List 轉成帶有 Key 的 Dict
    readable_json_data = []
    for row in nested_list_data:
        # 使用 zip 將欄位名稱與資料打包成字典
        readable_json_data.append(dict(zip(field_keys, row)))

    # 4. 將轉換後的 Readable JSON 寫入新檔案
    output_filename = 'datasets/NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-3B-Instruct.json_readable.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        # indent=4 可以生成漂亮的縮進排版
        # ensure_ascii=False 確保裡面的文字與符號不會被轉成無意義的 \uXXXX 編碼
        json.dump(readable_json_data, f, indent=4, ensure_ascii=False)

    print(f"轉換完成！好讀版 JSON 已儲存至：{output_filename}")

except FileNotFoundError:
    print("找不到指定的 JSON 檔案，請檢查檔名或路徑是否正確。")
except Exception as e:
    print(f"轉換過程中發生錯誤: {e}")