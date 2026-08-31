from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_name: str
    subset: str | None
    split: str
    question_key: str
    answer_key: str


DATASET_SPECS = {
    "gsm8k": DatasetSpec(
        name="gsm8k",
        source_name="openai/gsm8k",
        subset="main",
        split="test",
        question_key="question",
        answer_key="answer",
    ),
    "math500": DatasetSpec(
        name="math500",
        source_name="HuggingFaceH4/MATH-500",
        subset=None,
        split="test",
        question_key="problem",
        answer_key="answer",
    ),
    "aime25": DatasetSpec(
        name="aime25",
        source_name="yentinglin/aime_2025",
        subset="part1",
        split="train",
        question_key="problem",
        answer_key="answer",
    ),
}


def build_turns(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def iter_records(dataset: Dataset, spec: DatasetSpec, limit: int | None) -> Iterable[dict]:
    total = len(dataset) if limit is None else min(len(dataset), limit)
    for index in range(total):
        row = dataset[index]
        question = str(row[spec.question_key]).strip()
        answer = str(row[spec.answer_key]).strip()
        yield {
            "dataset": spec.name,
            "sample_id": index,
            "turns": build_turns(question),
            "ground_truth": answer,
        }


def save_jsonl(records: Iterable[dict], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def save_preview(jsonl_path: Path, preview_path: Path, limit: int) -> None:
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()[:limit]
    with preview_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            record = json.loads(line)
            handle.write(f"[{record['dataset']} #{record['sample_id']}]\n")
            handle.write(record["turns"][-1]["content"] + "\n\n")


def prepare_dataset(output_dir: Path, spec: DatasetSpec, limit: int | None, preview_limit: int) -> dict:
    dataset = load_dataset(spec.source_name, spec.subset, split=spec.split)
    output_path = output_dir / f"{spec.name}.jsonl"
    count = save_jsonl(iter_records(dataset, spec, limit), output_path)
    preview_path = output_dir / f"{spec.name}_preview.txt"
    save_preview(output_path, preview_path, preview_limit)
    return {
        "dataset": spec.name,
        "source_name": spec.source_name,
        "subset": spec.subset,
        "split": spec.split,
        "rows_written": count,
        "output_path": str(output_path),
        "preview_path": str(preview_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert eval datasets into DeepSpec-compatible JSONL."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "math500", "aime25"],
        choices=sorted(DATASET_SPECS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval_datasets"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap per dataset for quick smoke runs.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=5,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for dataset_name in args.datasets:
        manifest.append(
            prepare_dataset(
                output_dir=args.output_dir,
                spec=DATASET_SPECS[dataset_name],
                limit=args.limit,
                preview_limit=args.preview_limit,
            )
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "datasets": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
