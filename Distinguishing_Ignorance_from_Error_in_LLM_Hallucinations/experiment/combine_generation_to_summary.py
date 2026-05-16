#!/usr/bin/env python3
"""Combine generation files into direction_summary by adding task field."""

import json
import sys
from pathlib import Path

output_dir = Path("experiment/direction_outputs")

summary_rows = []

# Process hall_vs_nonhall
gen_file = output_dir / "generation_hall_vs_nonhall_concatFalse_nonstaticFalse.jsonl"
print(f"Reading {gen_file}...", file=sys.stderr)
with open(gen_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        row = json.loads(line)
        row["task"] = "hall_vs_nonhall"
        summary_rows.append(row)
print(f"  Added {i+1} rows from hall_vs_nonhall", file=sys.stderr)

# Process general_vs_nonhall
gen_file = output_dir / "generation_general_vs_nonhall_concatFalse_nonstaticFalse.jsonl"
print(f"Reading {gen_file}...", file=sys.stderr)
with open(gen_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        row = json.loads(line)
        row["task"] = "general_vs_nonhall"
        summary_rows.append(row)
print(f"  Added {i+1} rows from general_vs_nonhall", file=sys.stderr)

# Write summary
summary_path = output_dir / "direction_summary_concatFalse_nonstaticFalse.jsonl"
print(f"Writing {summary_path}...", file=sys.stderr)
with open(summary_path, "w", encoding="utf-8") as f:
    for row in summary_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Combined {len(summary_rows)} rows into {summary_path}")
print(f"First record task: {summary_rows[0].get('task')}")
print(f"Record 129 task: {summary_rows[128].get('task')}")
