import argparse
from pathlib import Path

import numpy as np


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_results_dir() -> Path:
    return resolve_repo_root() / "results"


def list_npy_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*.npy"))


def summarize_array(arr: np.ndarray, preview_items: int = 10) -> str:
    lines = []
    lines.append(f"shape: {arr.shape}")
    lines.append(f"dtype: {arr.dtype}")
    lines.append(f"size: {arr.size}")

    if arr.size == 0:
        lines.append("preview: <empty array>")
        return "\n".join(lines)

    flat = arr.reshape(-1)
    n = min(preview_items, flat.size)
    preview = flat[:n]
    lines.append(f"preview_first_{n}: {preview}")

    # Only compute numeric stats for numeric arrays.
    if np.issubdtype(arr.dtype, np.number):
        lines.append(f"min: {np.min(arr)}")
        lines.append(f"max: {np.max(arr)}")
        lines.append(f"mean: {np.mean(arr)}")

    return "\n".join(lines)


def print_file_list(files: list[Path], base_dir: Path) -> None:
    if not files:
        print(f"No .npy files found under: {base_dir}")
        return

    print(f"Found {len(files)} .npy files under: {base_dir}")
    for i, f in enumerate(files, start=1):
        rel = f.relative_to(base_dir)
        print(f"[{i:03d}] {rel}")


def select_file(files: list[Path], index: int | None, name_contains: str | None) -> Path:
    if not files:
        raise FileNotFoundError("No .npy files available to select.")

    if index is not None:
        if index < 1 or index > len(files):
            raise ValueError(f"--index must be between 1 and {len(files)}")
        return files[index - 1]

    if name_contains:
        matched = [f for f in files if name_contains in str(f)]
        if len(matched) == 0:
            raise FileNotFoundError(f"No file matched --contains '{name_contains}'")
        if len(matched) > 1:
            raise ValueError(
                "Multiple files matched --contains. Use a more specific keyword or --index.\n"
                + "\n".join(str(m) for m in matched[:20])
            )
        return matched[0]

    return files[0]


def infer_dataset_class(file_path: Path) -> str:
    """Infer class label from result filename and parent path."""
    name = file_path.name.lower()
    path_text = str(file_path).lower()

    if "hallucinate" in name or "with_hall" in name:
        return "hallucinate"
    if "nonhallucinate" in name or "without_hall" in name:
        return "nonhallucinate"
    if "general" in name:
        return "general"

    if "static" in path_text:
        return "static_or_mixed"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read and inspect .npy result files under the results folder."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(default_results_dir()),
        help="Path to results directory (default: repository/results)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all .npy files and exit",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="1-based index from --list output",
    )
    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Select file by substring in path",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Direct file path to a .npy file (overrides --index/--contains)",
    )
    parser.add_argument(
        "--preview_items",
        type=int,
        default=10,
        help="How many flattened elements to preview",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full array content (can be very large)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files = list_npy_files(results_dir)

    if args.list:
        print_file_list(files, results_dir)
        return

    if args.file:
        target = Path(args.file).resolve()
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
    else:
        target = select_file(files, args.index, args.contains)

    arr = np.load(target, allow_pickle=True)

    print(f"Selected file: {target}")
    print(f"inferred_class: {infer_dataset_class(target)}")
    print(summarize_array(arr, preview_items=args.preview_items))
    if args.full:
        with np.printoptions(threshold=np.inf):
            print("full_array:")
            print(arr)


if __name__ == "__main__":
    main()
