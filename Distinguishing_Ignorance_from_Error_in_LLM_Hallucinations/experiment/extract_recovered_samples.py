import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def key_of(row: Dict[str, Any]) -> Tuple[str, int]:
    # Use batch and index to match baseline/intervention pairs
    return (row.get("batch", ""), int(row.get("index", -1)))


def extract_recovered(rows):
    grouped: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        k = key_of(row)
        grouped.setdefault(k, {})[row.get("mode", "")] = row

    recovered = []
    for k, pair in grouped.items():
        baseline = pair.get("baseline")
        intervention = pair.get("intervention")
        if not baseline or not intervention:
            continue
        # Condition: originally hallucinated (wrong_hit True) and later correct (gold_hit True)
        if baseline.get("wrong_hit") and intervention.get("gold_hit"):
            recovered.append({"key": k, "baseline": baseline, "intervention": intervention})
    return recovered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to generation or summary JSONL")
    parser.add_argument("--out", type=str, default=None, help="Path to write recovered JSONL (optional)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    rows = list(load_jsonl(inp))
    recovered = extract_recovered(rows)

    print(f"Loaded rows: {len(rows)}")
    print(f"Recovered count: {len(recovered)}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            for item in recovered:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote recovered cases to: {outp}")
    else:
        # Print first few examples
        for i, item in enumerate(recovered[:5], start=1):
            print(f"\n=== Recovered {i} ===")
            print("Batch,Index:", item["key"]) 
            print("Baseline prompt (clipped):", item["baseline"]["prompt"][:200].replace('\n','\\n'))
            print("Baseline response (clipped):", item["baseline"]["response"][:300].replace('\n','\\n'))
            print("Intervention response (clipped):", item["intervention"]["response"][:300].replace('\n','\\n'))


if __name__ == "__main__":
    main()
