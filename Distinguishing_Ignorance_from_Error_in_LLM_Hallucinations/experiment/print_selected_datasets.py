from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from analyze_hallucination_direction import find_dataset_file, default_dataset_dir

if __name__ == '__main__':
    dataset_dir = default_dataset_dir()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name = "trivia_qa_no_context"
    threshold = "1.0"
    for class_name in ["hallucinate", "nonhallucinate"]:
        try:
            p = find_dataset_file(dataset_dir, class_name, dataset_name, threshold, model_name)
            print(class_name, ":", p)
        except Exception as e:
            print(class_name, ": ERROR ->", e)
