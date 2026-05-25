import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


CLASS_NAMES = ["hallucinate", "nonhallucinate", "general"]
TASK_PAIRS = {
    "hall_vs_nonhall": ("hallucinate", "nonhallucinate"),
    "general_vs_nonhall": ("general", "nonhallucinate"),
}
TASK_TITLES = {
    "hall_vs_nonhall": "Hallucinate vs NonHallucinate",
    "general_vs_nonhall": "General(HK-) vs NonHallucinate",
}
HK_LABELS = {
    "hallucinate": "HK+",
    "general": "HK-",
    "nonhallucinate": "NON-HALL",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_results_dir() -> Path:
    return repo_root() / "results"


def default_dataset_dir() -> Path:
    return repo_root() / "datasets"


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def build_result_dir(
    results_dir: Path,
    model_name: str,
    dataset_name: str,
    threshold: str,
    concat_answer: bool,
    dataset_size: int,
    non_static: bool,
) -> Path:
    result_dir = (
        results_dir
        / model_slug(model_name)
        / dataset_name
        / threshold
        / f"concat_answer{concat_answer}_size{dataset_size}"
    )
    if non_static:
        result_dir = result_dir / "non_static"
    return result_dir


def vector_filename(vector_type: str, class_name: str) -> str:
    if vector_type == "mlp":
        return f"_all_mlp_vector_{class_name}.npy"
    if vector_type == "attention":
        return f"_all_attention_vector_{class_name}.npy"
    if vector_type == "residual":
        return f"_all_residual_vectors_{class_name}.npy"
    if vector_type == "heads":
        return f"_heads_vectors_{class_name}_no_projection.npy"
    raise ValueError(f"Unsupported vector type: {vector_type}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_three_class_arrays(result_dir: Path, vector_type: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    missing = []

    for class_name in CLASS_NAMES:
        file_path = result_dir / vector_filename(vector_type, class_name)
        if file_path.exists():
            arrays[class_name] = np.load(file_path, allow_pickle=True)
        else:
            missing.append(str(file_path))

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return arrays


def extract_layer_matrix(arr: np.ndarray, layer_idx: int) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, layer_idx, :]
    if arr.ndim == 4:
        return arr[:, layer_idx, :, :].reshape(arr.shape[0], -1)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def fit_direction_vector(
    positive: np.ndarray,
    negative: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    if method == "mean_diff":
        direction = positive.mean(axis=0) - negative.mean(axis=0)
    elif method == "pca":
        pca = PCA(n_components=1, random_state=seed)
        x = np.vstack([positive, negative])
        pca.fit(x)
        direction = pca.components_[0]

        pos_proj = positive @ direction
        neg_proj = negative @ direction
        if pos_proj.mean() < neg_proj.mean():
            direction = -direction
    else:
        raise ValueError(f"Unsupported direction method: {method}")

    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Direction vector has zero norm")
    return direction / norm


def compute_projection_scores(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return x @ direction


def find_best_layer_by_probe(
    arrays: dict[str, np.ndarray],
    task: str,
    max_samples_per_class: int,
    seed: int,
    n_splits: int = 5,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_name, negative_name = TASK_PAIRS[task]
    positive = arrays[positive_name]
    negative = arrays[negative_name]
    n_layers = positive.shape[1]

    idx_pos = sample_indices(len(positive), max_samples_per_class, seed)
    idx_neg = sample_indices(len(negative), max_samples_per_class, seed + 1)
    positive = positive[idx_pos]
    negative = negative[idx_neg]

    y = np.concatenate([
        np.ones(len(positive), dtype=np.int64),
        np.zeros(len(negative), dtype=np.int64),
    ])

    accuracy_per_layer = []
    f1_per_layer = []
    auc_per_layer = []
    accuracy_std = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for layer_idx in range(n_layers):
        x_pos = extract_layer_matrix(positive, layer_idx)
        x_neg = extract_layer_matrix(negative, layer_idx)
        x = np.vstack([x_pos, x_neg])

        acc_scores = []
        f1_scores = []
        auc_scores = []

        for train_idx, test_idx in skf.split(x, y):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            clf = LogisticRegression(
                max_iter=1000,
                random_state=seed,
                penalty="l2",
                C=0.1,
                class_weight="balanced",
            )
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            y_prob = clf.predict_proba(x_test)[:, 1]
            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            try:
                auc_scores.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                auc_scores.append(float("nan"))

        accuracy_per_layer.append(np.mean(acc_scores))
        f1_per_layer.append(np.mean(f1_scores))
        auc_per_layer.append(np.nanmean(auc_scores))
        accuracy_std.append(np.std(acc_scores))

    accuracy_per_layer = np.asarray(accuracy_per_layer)
    f1_per_layer = np.asarray(f1_per_layer)
    auc_per_layer = np.asarray(auc_per_layer)
    accuracy_std = np.asarray(accuracy_std)
    best_layer = int(np.nanargmax(accuracy_per_layer))
    return best_layer, accuracy_per_layer, f1_per_layer, auc_per_layer, accuracy_std


def train_direction_from_layer(
    arrays: dict[str, np.ndarray],
    task: str,
    layer_idx: int,
    direction_method: str,
    max_samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive_name, negative_name = TASK_PAIRS[task]
    positive = arrays[positive_name]
    negative = arrays[negative_name]

    idx_pos = sample_indices(len(positive), max_samples_per_class, seed)
    idx_neg = sample_indices(len(negative), max_samples_per_class, seed + 1)
    positive = positive[idx_pos]
    negative = negative[idx_neg]

    x_pos = extract_layer_matrix(positive, layer_idx)
    x_neg = extract_layer_matrix(negative, layer_idx)
    x = np.vstack([x_pos, x_neg])
    y = np.concatenate([
        np.ones(len(x_pos), dtype=np.int64),
        np.zeros(len(x_neg), dtype=np.int64),
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    direction = fit_direction_vector(
        positive=x_train[y_train == 1],
        negative=x_train[y_train == 0],
        method=direction_method,
        seed=seed,
    )

    train_scores = compute_projection_scores(x_train, direction)
    test_scores = compute_projection_scores(x_test, direction)

    train_auc = roc_auc_score(y_train, train_scores)
    test_auc = roc_auc_score(y_test, test_scores)
    train_acc = accuracy_score(y_train, train_scores >= 0.0)
    test_acc = accuracy_score(y_test, test_scores >= 0.0)

    return direction, x_test, y_test, np.array([train_auc, test_auc, train_acc, test_acc])


def save_probe_plot(
    accuracy: np.ndarray,
    f1: np.ndarray,
    auc: np.ndarray,
    accuracy_std: np.ndarray,
    task_title: str,
    out_file: Path,
) -> None:
    layers = np.arange(len(accuracy))
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, accuracy, yerr=accuracy_std, marker="o", linewidth=2, capsize=3, label="Accuracy")
    plt.plot(layers, f1, marker="s", linewidth=2, label="F1")
    plt.plot(layers, auc, marker="^", linewidth=2, label="AUC")
    plt.xlabel("Layer Index")
    plt.ylabel("Score")
    plt.title(f"Layer probing - {task_title}")
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def find_dataset_file(
    dataset_dir: Path,
    class_name: str,
    dataset_name: str,
    threshold: str,
    model_name: str,
) -> tuple[Path, int]:
    target_terms = [
        class_name.lower(),
        dataset_name.lower(),
        f"threshold{threshold.lower()}",
        model_slug(model_name).lower(),
    ]
    candidates = []

    for path in dataset_dir.rglob("*.json"):
        lower_name = path.name.lower()
        score = sum(1 for term in target_terms if term in lower_name)
        if score >= 3:
            candidates.append((score, len(lower_name), path))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find dataset file for {class_name} under {dataset_dir}. "
            f"Try passing explicit --dataset_dir or rename files to include {target_terms}."
        )

    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    best = candidates[0]
    return best[2], best[0]


def normalize_task_name(task: str) -> str:
    if task == "gen_vs_hall":
        return "general_vs_nonhall"
    return task


def resolve_tasks(task: str) -> list[str]:
    task = normalize_task_name(task)
    if task == "both":
        return ["hall_vs_nonhall", "general_vs_nonhall"]
    return [task]


def load_records(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data)}")
    return data


def build_prompt_from_record(
    record: list[Any],
    concat_answer: bool,
    use_badshot_prompt: bool,
) -> tuple[str, str]:
    prompt_index = 5 if use_badshot_prompt and len(record) > 5 else 0
    prompt = str(record[prompt_index])
    appended_answer = ""
    if concat_answer and len(record) > 2:
        appended_answer = str(record[2])
        prompt = prompt + appended_answer
    return prompt, appended_answer


def load_eval_records(
    dataset_dir: Path,
    dataset_name: str,
    threshold: str,
    model_name: str,
    class_name: str,
    limit: int,
    seed: int,
    explicit_dataset_file: str | None = None,
) -> tuple[Path, int, str, list[int], list[list[Any]]]:
    if explicit_dataset_file:
        candidate = Path(explicit_dataset_file)
        file_path = candidate if candidate.is_absolute() else (dataset_dir / candidate)
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Explicit dataset file not found for class '{class_name}': {file_path}")
        match_score = -1
    else:
        file_path, match_score = find_dataset_file(
            dataset_dir=dataset_dir,
            class_name=class_name,
            dataset_name=dataset_name,
            threshold=threshold,
            model_name=model_name,
        )
    records = load_records(file_path)
    idx = sample_indices(len(records), limit, seed)

    fname = file_path.name.lower()
    original_label = "unknown"
    for name in CLASS_NAMES:
        if name in fname:
            original_label = name
            break

    return file_path, match_score, original_label, idx.tolist(), [records[i] for i in idx]


def print_selected_samples(
    task: str,
    batch_name: str,
    source_class: str,
    dataset_file: str,
    dataset_match_score: int,
    dataset_original_label: str,
    sampled_indices: list[int],
    records: list[list[Any]],
    concat_answer: bool,
    use_badshot_prompt: bool,
    preview_max_chars: int,
) -> None:
    print("\n--- Preview Selected Samples ---")
    print(f"task={task}, batch={batch_name}, source_class={source_class}")
    print(
        f"dataset_file={dataset_file}, match_score={dataset_match_score}, "
        f"dataset_original_label={dataset_original_label}"
    )
    for i, (sample_idx, record) in enumerate(zip(sampled_indices, records), start=1):
        prompt, appended_answer = build_prompt_from_record(
            record,
            concat_answer=concat_answer,
            use_badshot_prompt=use_badshot_prompt,
        )
        gold = str(record[1]) if len(record) > 1 else ""
        wrong = str(record[2]) if len(record) > 2 else ""
        clipped_prompt = prompt if len(prompt) <= preview_max_chars else (prompt[:preview_max_chars] + " ...")
        print(
            json.dumps(
                {
                    "index": i,
                    "sample_idx": sample_idx,
                    "source_class": source_class,
                    "dataset_original_label": dataset_original_label,
                    "gold": gold,
                    "wrong": wrong,
                    "appended_answer": appended_answer,
                    "prompt": clipped_prompt,
                },
                ensure_ascii=False,
            )
        )


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tokenizer


def get_model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def add_direction_hook(
    model,
    layer_idx: int,
    direction: np.ndarray,
    alpha: float,
    token_position: str,
):
    device = get_model_input_device(model)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    layer_module = model.model.layers[layer_idx]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        hidden = hidden.clone()
        if token_position == "last":
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * direction_t.to(hidden.dtype)
        elif token_position == "all":
            hidden = hidden + alpha * direction_t.to(hidden.dtype).view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported token_position: {token_position}")

        if rest is None:
            return hidden
        return (hidden, *rest)

    return layer_module.register_forward_hook(hook)


def response_contains(text: str, target: str) -> bool:
    return target.strip().lower() in text.strip().lower()


def evaluate_generation(
    model,
    tokenizer,
    records: list[list[Any]],
    concat_answer: bool,
    use_badshot_prompt: bool,
    max_new_tokens: int,
    batch_name: str,
    mode: str,
    task_positive_class: str,
    task_negative_class: str,
    source_class: str,
    dataset_file: str,
    dataset_match_score: int,
    dataset_original_label: str,
) -> list[dict[str, Any]]:
    outputs = []
    for index, record in enumerate(records, start=1):
        prompt, appended_answer = build_prompt_from_record(
            record,
            concat_answer=concat_answer,
            use_badshot_prompt=use_badshot_prompt,
        )
        gold = str(record[1]) if len(record) > 1 else ""
        wrong = str(record[2]) if len(record) > 2 else ""

        response = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        wrong_hit = response_contains(response, wrong) if not (concat_answer and appended_answer) else False
        outputs.append(
            {
                "index": index,
                "batch": batch_name,
                "mode": mode,
                "task_positive_class": task_positive_class,
                "task_negative_class": task_negative_class,
                "source_class": source_class,
                "hk_label": HK_LABELS.get(source_class, "UNKNOWN"),
                "dataset_file": dataset_file,
                "dataset_match_score": int(dataset_match_score),
                "dataset_original_label": dataset_original_label,
                "prompt": prompt,
                "gold": gold,
                "wrong": wrong,
                "appended_answer": appended_answer,
                "response": response,
                "gold_hit": response_contains(response, gold),
                "wrong_hit": wrong_hit,
            }
        )
    return outputs


def summarize_generation(outputs: list[dict[str, Any]]) -> dict[str, float]:
    if not outputs:
        return {
            "gold_hit_rate": float("nan"),
            "wrong_hit_rate": float("nan"),
        }
    return {
        "gold_hit_rate": float(np.mean([row["gold_hit"] for row in outputs])),
        "wrong_hit_rate": float(np.mean([row["wrong_hit"] for row in outputs])),
    }


def analyze_per_class_metrics(
    baseline_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    task_title: str,
) -> None:
    """Analyze gold-hit and wrong-hit rates grouped by source_class and mode."""
    from collections import defaultdict

    metrics = defaultdict(lambda: {"baseline": {"gold_hit": [], "wrong_hit": []}, "intervention": {"gold_hit": [], "wrong_hit": []}})

    for row in baseline_rows:
        source_class = row.get("source_class", "unknown")
        metrics[source_class]["baseline"]["gold_hit"].append(row.get("gold_hit", False))
        metrics[source_class]["baseline"]["wrong_hit"].append(row.get("wrong_hit", False))

    for row in intervention_rows:
        source_class = row.get("source_class", "unknown")
        metrics[source_class]["intervention"]["gold_hit"].append(row.get("gold_hit", False))
        metrics[source_class]["intervention"]["wrong_hit"].append(row.get("wrong_hit", False))

    print(f"\n{task_title} - Per-Class Metrics:")
    print("=" * 120)
    print(f"{'Source Class':<20} {'Mode':<15} {'Count':<10} {'Gold-Hit Rate':<18} {'Wrong-Hit Rate':<18}")
    print("=" * 120)

    for source_class in sorted(metrics.keys()):
        for mode in ["baseline", "intervention"]:
            data = metrics[source_class][mode]
            count = len(data["gold_hit"])
            gold_hit_rate = sum(data["gold_hit"]) / count if count > 0 else float("nan")
            wrong_hit_rate = sum(data["wrong_hit"]) / count if count > 0 else float("nan")
            print(f"{source_class:<20} {mode:<15} {count:<10} {gold_hit_rate:<18.4f} {wrong_hit_rate:<18.4f}")

    print("\n" + "=" * 120)
    print("Intervention Effect (baseline → intervention):")
    print("=" * 120)
    print(f"{'Source Class':<20} {'Δ Gold-Hit':<20} {'Δ Wrong-Hit':<20}")
    print("=" * 120)

    for source_class in sorted(metrics.keys()):
        baseline_data = metrics[source_class]["baseline"]
        intervention_data = metrics[source_class]["intervention"]

        b_count = len(baseline_data["gold_hit"])
        i_count = len(intervention_data["gold_hit"])

        if b_count > 0 and i_count > 0:
            b_gold_rate = sum(baseline_data["gold_hit"]) / b_count
            i_gold_rate = sum(intervention_data["gold_hit"]) / i_count
            delta_gold = i_gold_rate - b_gold_rate

            b_wrong_rate = sum(baseline_data["wrong_hit"]) / b_count
            i_wrong_rate = sum(intervention_data["wrong_hit"]) / i_count
            delta_wrong = i_wrong_rate - b_wrong_rate

            print(f"{source_class:<20} {delta_gold:<20.4f} {delta_wrong:<20.4f}")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn a hallucination direction from saved hidden states and test activation intervention."
    )
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--dataset_dir", type=str, default=str(default_dataset_dir()))
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="trivia_qa_no_context")
    parser.add_argument("--threshold", type=str, default="1.0")
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--concat_answer", action="store_true")
    parser.add_argument("--non_static", action="store_true")
    parser.add_argument(
        "--vector_type",
        type=str,
        choices=["residual", "mlp", "attention", "heads"],
        default="residual",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["hall_vs_nonhall", "general_vs_nonhall", "gen_vs_hall", "both"],
        default="both",
    )
    parser.add_argument(
        "--direction_method",
        type=str,
        choices=["mean_diff", "pca"],
        default="mean_diff",
    )
    parser.add_argument("--max_samples_per_class", type=int, default=500)
    parser.add_argument("--probe_splits", type=int, default=5)
    parser.add_argument("--direction_train_frac", type=float, default=0.8)
    parser.add_argument("--eval_samples_per_class", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--token_position", type=str, choices=["last", "all"], default="last")
    parser.add_argument(
        "--raw_prompt",
        action="store_true",
        help="Use prompt[0] instead of the bad-shot prompt field (index 5).",
    )
    parser.add_argument("--output_dir", type=str, default=str(repo_root() / "experiment" / "direction_outputs"))
    parser.add_argument("--save_model_outputs", action="store_true")
    parser.add_argument(
        "--eval_dataset_hallucinate",
        type=str,
        default=str(default_dataset_dir() / "HallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct.json"),
        help="Explicit dataset JSON file path for hallucinate evaluation samples.",
    )
    parser.add_argument(
        "--eval_dataset_nonhallucinate",
        type=str,
        default=str(default_dataset_dir() / "NonHallucinateTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct.json"),
        help="Explicit dataset JSON file path for nonhallucinate evaluation samples.",
    )
    parser.add_argument(
        "--eval_dataset_general",
        type=str,
        default=str(default_dataset_dir() / "GeneralTrivia_qa_no_contextWithThreshold1.0_meta-llama_Llama-3.2-1B-Instruct.json"),
        help="Explicit dataset JSON file path for general evaluation samples.",
    )
    parser.add_argument(
        "--preview_eval_samples_only",
        action="store_true",
        help="Print selected eval samples only and skip model loading/generation.",
    )
    parser.add_argument(
        "--preview_max_chars",
        type=int,
        default=260,
        help="Max prompt characters to print for each previewed sample.",
    )
    parser.add_argument(
        "--fail_on_identical_eval_datasets",
        action="store_true",
        help="Raise an error if positive and negative eval dataset files are byte-identical.",
    )

    args = parser.parse_args()

    explicit_eval_datasets = {
        "hallucinate": args.eval_dataset_hallucinate,
        "nonhallucinate": args.eval_dataset_nonhallucinate,
        "general": args.eval_dataset_general,
    }

    results_dir = Path(args.results_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    target_dir = build_result_dir(
        results_dir=results_dir,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        threshold=args.threshold,
        concat_answer=args.concat_answer,
        dataset_size=args.dataset_size,
        non_static=args.non_static,
    )
    if not target_dir.exists():
        raise FileNotFoundError(f"Target result directory not found: {target_dir}")

    arrays = load_three_class_arrays(target_dir, args.vector_type)
    print(f"target_dir: {target_dir}")
    print(f"vector_type: {args.vector_type}")
    for class_name, arr in arrays.items():
        print(f"{class_name}: shape={arr.shape}, dtype={arr.dtype}")

    use_badshot_prompt = not args.raw_prompt
    tasks = resolve_tasks(args.task)
    best_layers: dict[str, int] = {}

    print("\n=== Layer probing ===")
    for task in tasks:
        best_layer, acc, f1, auc, acc_std = find_best_layer_by_probe(
            arrays=arrays,
            task=task,
            max_samples_per_class=args.max_samples_per_class,
            seed=42,
            n_splits=args.probe_splits,
        )
        best_layers[task] = best_layer
        print(f"\nTask: {TASK_TITLES[task]}")
        print(f"Best layer: {best_layer}")
        print(
            f"Best layer score: acc={acc[best_layer]:.4f} ± {acc_std[best_layer]:.4f}, "
            f"f1={f1[best_layer]:.4f}, auc={auc[best_layer]:.4f}"
        )

        plot_file = output_dir / f"probe_{task}_{args.vector_type}_concat{args.concat_answer}_nonstatic{args.non_static}.png"
        save_probe_plot(acc, f1, auc, acc_std, TASK_TITLES[task], plot_file)
        print(f"Saved probing plot: {plot_file}")

    learned: dict[str, dict[str, Any]] = {}
    print("\n=== Direction learning ===")
    for task in tasks:
        best_layer = best_layers[task]
        direction, x_test, y_test, train_eval = train_direction_from_layer(
            arrays=arrays,
            task=task,
            layer_idx=best_layer,
            direction_method=args.direction_method,
            max_samples_per_class=args.max_samples_per_class,
            seed=42,
        )
        learned[task] = {
            "layer": best_layer,
            "direction": direction,
            "train_auc": float(train_eval[0]),
            "test_auc": float(train_eval[1]),
            "train_acc": float(train_eval[2]),
            "test_acc": float(train_eval[3]),
        }

        direction_file = output_dir / f"direction_{task}_{args.vector_type}_{args.direction_method}_layer{best_layer}.npy"
        np.save(direction_file, direction)
        print(f"Task: {TASK_TITLES[task]}")
        print(f"Saved direction: {direction_file}")
        print(
            f"Projection split metrics: train_auc={train_eval[0]:.4f}, test_auc={train_eval[1]:.4f}, "
            f"train_acc={train_eval[2]:.4f}, test_acc={train_eval[3]:.4f}"
        )

    print("\n=== Generation evaluation ===")
    model = None
    tokenizer = None
    all_report_rows: list[dict[str, Any]] = []

    for task in tasks:
        positive_name, negative_name = TASK_PAIRS[task]
        best_layer = learned[task]["layer"]
        direction = learned[task]["direction"]

        positive_dataset_file, positive_match_score, positive_original_label, positive_sample_indices, eval_positive = load_eval_records(
            dataset_dir=dataset_dir,
            dataset_name=args.dataset_name,
            threshold=args.threshold,
            model_name=args.model_name,
            class_name=positive_name,
            limit=args.eval_samples_per_class,
            seed=42,
            explicit_dataset_file=explicit_eval_datasets.get(positive_name),
        )
        negative_dataset_file, negative_match_score, negative_original_label, negative_sample_indices, eval_negative = load_eval_records(
            dataset_dir=dataset_dir,
            dataset_name=args.dataset_name,
            threshold=args.threshold,
            model_name=args.model_name,
            class_name=negative_name,
            limit=args.eval_samples_per_class,
            seed=43,
            explicit_dataset_file=explicit_eval_datasets.get(negative_name),
        )

        print(f"\nTask: {TASK_TITLES[task]}")
        print(f"Best layer: {best_layer}")
        print(f"Positive dataset ({positive_name}): {positive_dataset_file} (match_score={positive_match_score}, original_label={positive_original_label})")
        print(f"Negative dataset ({negative_name}): {negative_dataset_file} (match_score={negative_match_score}, original_label={negative_original_label})")

        pos_hash = file_sha256(positive_dataset_file)
        neg_hash = file_sha256(negative_dataset_file)
        if pos_hash == neg_hash:
            msg = (
                f"[WARNING] Positive and negative eval datasets are byte-identical for task={task}. "
                f"This makes class-level eval unreliable. "
                f"file={positive_dataset_file}, sha256={pos_hash}"
            )
            if args.fail_on_identical_eval_datasets:
                raise ValueError(msg)
            print(msg)

        batch_specs = [
            (
                f"{task}_positive",
                positive_name,
                str(positive_dataset_file),
                eval_positive,
                positive_match_score,
                positive_original_label,
                positive_sample_indices,
            ),
            (
                f"{task}_negative",
                negative_name,
                str(negative_dataset_file),
                eval_negative,
                negative_match_score,
                negative_original_label,
                negative_sample_indices,
            ),
        ]

        if args.preview_eval_samples_only:
            for batch_name, source_class, dataset_file, records, match_score, original_label, sampled_indices in batch_specs:
                print_selected_samples(
                    task=task,
                    batch_name=batch_name,
                    source_class=source_class,
                    dataset_file=dataset_file,
                    dataset_match_score=match_score,
                    dataset_original_label=original_label,
                    sampled_indices=sampled_indices,
                    records=records,
                    concat_answer=args.concat_answer,
                    use_badshot_prompt=use_badshot_prompt,
                    preview_max_chars=args.preview_max_chars,
                )
            continue

        baseline_rows: list[dict[str, Any]] = []
        intervention_rows: list[dict[str, Any]] = []

        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(args.model_name)

        for batch_name, source_class, dataset_file, records, match_score, original_label, _ in batch_specs:
            baseline_rows.extend(
                evaluate_generation(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    concat_answer=args.concat_answer,
                    use_badshot_prompt=use_badshot_prompt,
                    max_new_tokens=args.max_new_tokens,
                    batch_name=batch_name,
                    mode="baseline",
                    task_positive_class=positive_name,
                    task_negative_class=negative_name,
                    source_class=source_class,
                    dataset_file=dataset_file,
                    dataset_match_score=match_score,
                    dataset_original_label=original_label,
                )
            )

            handle = add_direction_hook(
                model=model,
                layer_idx=best_layer,
                direction=direction,
                alpha=args.alpha,
                token_position=args.token_position,
            )
            try:
                intervention_rows.extend(
                    evaluate_generation(
                        model=model,
                        tokenizer=tokenizer,
                        records=records,
                        concat_answer=args.concat_answer,
                        use_badshot_prompt=use_badshot_prompt,
                        max_new_tokens=args.max_new_tokens,
                        batch_name=batch_name,
                        mode="intervention",
                        task_positive_class=positive_name,
                        task_negative_class=negative_name,
                        source_class=source_class,
                        dataset_file=dataset_file,
                        dataset_match_score=match_score,
                        dataset_original_label=original_label,
                    )
                )
            finally:
                handle.remove()

        baseline_summary = summarize_generation(baseline_rows)
        intervention_summary = summarize_generation(intervention_rows)

        print(f"Baseline gold-hit rate: {baseline_summary['gold_hit_rate']:.4f}")
        print(f"Intervention gold-hit rate: {intervention_summary['gold_hit_rate']:.4f}")
        print(f"Baseline wrong-hit rate: {baseline_summary['wrong_hit_rate']:.4f}")
        print(f"Intervention wrong-hit rate: {intervention_summary['wrong_hit_rate']:.4f}")

        analyze_per_class_metrics(baseline_rows, intervention_rows, TASK_TITLES[task])

        task_rows = baseline_rows + intervention_rows
        if args.save_model_outputs:
            jsonl_path = output_dir / f"generation_{task}_concat{args.concat_answer}_nonstatic{args.non_static}.jsonl"
            save_jsonl(jsonl_path, task_rows)
            print(f"Saved generation records: {jsonl_path}")

        all_report_rows.extend([{"task": task, **row} for row in task_rows])

    if args.preview_eval_samples_only:
        print("\nPreview mode completed. Model loading and generation were skipped.")
        return

    summary_path = output_dir / f"direction_summary_concat{args.concat_answer}_nonstatic{args.non_static}.jsonl"
    save_jsonl(summary_path, all_report_rows)
    print(f"\nSaved combined summary: {summary_path}")


if __name__ == "__main__":
    main()
