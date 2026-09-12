#!/usr/bin/env python3
"""Build a deterministic, multi-domain prompt manifest for MISA router training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable


AIME_TEMPLATE = """Solve the following AIME problem step by step. The last line
of your response should be of the form Answer: $ANSWER.

{question}"""

MATH_TEMPLATE = """Solve the following mathematics problem. Explain your
reasoning step by step and put the final answer in \\boxed{{}}.

{problem}"""

GPQA_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)


def stable_key(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def split_for(source_id: str) -> str:
    bucket = int.from_bytes(stable_key(source_id)[:4], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def round_robin(groups: dict[str, list[dict]]) -> Iterable[dict]:
    queues = {
        key: deque(sorted(rows, key=lambda row: stable_key(row["source_id"])))
        for key, rows in sorted(groups.items())
    }
    while queues:
        for key in list(queues):
            if queues[key]:
                yield queues[key].popleft()
            if not queues[key]:
                del queues[key]


def take_grouped(rows: Iterable[dict], count: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["task"])].append(row)
    return list(round_robin(groups))[:count]


def load_ruler(root: Path, count: int, min_tokens: int) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/validation.jsonl")):
        for offset, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            raw = json.loads(line)
            length = int(raw.get("length_w_model_temp", raw.get("length", 0)))
            if length < min_tokens:
                continue
            task = path.parent.name
            rows.append(
                {
                    "dataset": "ruler",
                    "task": task,
                    "source_id": f"ruler:{task}:{offset}",
                    "declared_tokens": length,
                    "text": raw["input"],
                }
            )
    selected = take_grouped(rows, count)
    if len(selected) < count:
        raise RuntimeError(f"RULER has only {len(selected)} qualifying prompts")
    return selected


def longbench_prompt(row: dict) -> str:
    choices = [row.get(f"choice_{x}", row.get(x, "")) for x in "ABCD"]
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{row.get('context', '').strip()}\n</text>\n\n"
        f"What is the correct answer to this question: {row['question'].strip()}\n"
        "Choices:\n"
        + "\n".join(f"({letter}) {choice}" for letter, choice in zip("ABCD", choices))
        + '\n\nFormat your response as: "The correct answer is (A/B/C/D)".'
    )


def load_longbench(source: Path, count: int) -> list[dict]:
    dataset = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for offset, raw in enumerate(dataset):
        row = dict(raw)
        source_id = f"longbench_v2:{row.get('_id', offset)}"
        rows.append(
            {
                "dataset": "longbench_v2",
                "task": str(row.get("domain", row.get("sub_domain", "unknown"))),
                "source_id": source_id,
                "declared_length": row.get("length"),
                "text": longbench_prompt(row),
            }
        )
    selected = take_grouped(rows, count)
    if len(selected) < count:
        raise RuntimeError(f"LongBench-v2 has only {len(selected)} prompts")
    return selected


def load_aime(sources: list[Path], count: int) -> list[dict]:
    rows = []
    for source in sources:
        config = source.stem
        records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for offset, raw in enumerate(records):
            rows.append(
                {
                    "dataset": "aime_2025",
                    "task": config.lower(),
                    "source_id": f"aime_2025:{config}:{offset}",
                    "text": AIME_TEMPLATE.format(question=raw["question"]),
                }
            )
    return sorted(rows, key=lambda row: stable_key(row["source_id"]))[:count]


def read_csv(source: str) -> list[dict]:
    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(source, timeout=120) as response:
            text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def load_gpqa(source: str, count: int) -> list[dict]:
    rows = []
    for offset, raw in enumerate(read_csv(source)):
        choices = [
            raw["Correct Answer"],
            raw["Incorrect Answer 1"],
            raw["Incorrect Answer 2"],
            raw["Incorrect Answer 3"],
        ]
        # A stable permutation prevents the correct option from always being A.
        choices.sort(key=lambda value: stable_key(f"{offset}:{value}"))
        text = (
            "Answer the following graduate-level multiple-choice question. "
            "Explain your reasoning, then state the chosen letter.\n\n"
            f"{raw['Question']}\n\n"
            + "\n".join(
                f"({letter}) {choice}" for letter, choice in zip("ABCD", choices)
            )
        )
        rows.append(
            {
                "dataset": "gpqa_diamond",
                "task": str(raw.get("High-level domain", "gpqa")),
                "source_id": f"gpqa_diamond:{offset}",
                "text": text,
            }
        )
    return take_grouped(rows, count)


def load_math500(source: Path, count: int) -> list[dict]:
    rows = []
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for offset, raw in enumerate(records):
        rows.append(
            {
                "dataset": "math_500",
                "task": str(raw.get("subject", raw.get("type", "math"))),
                "source_id": f"math_500:{offset}",
                "text": MATH_TEMPLATE.format(problem=raw["problem"]),
            }
        )
    return take_grouped(rows, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--ruler-count", type=int, default=256)
    parser.add_argument("--ruler-min-tokens", type=int, default=120000)
    parser.add_argument("--longbench-source", type=Path, required=True)
    parser.add_argument("--longbench-count", type=int, default=256)
    parser.add_argument(
        "--aime-sources", type=Path, nargs="+", required=True
    )
    parser.add_argument("--aime-count", type=int, default=30)
    parser.add_argument("--gpqa-count", type=int, default=448)
    parser.add_argument("--gpqa-source", default=GPQA_URL)
    parser.add_argument("--math-source", type=Path, required=True)
    parser.add_argument("--math-count", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(load_ruler(args.ruler_root, args.ruler_count, args.ruler_min_tokens))
    rows.extend(load_longbench(args.longbench_source, args.longbench_count))
    rows.extend(load_aime(args.aime_sources, args.aime_count))
    rows.extend(load_gpqa(args.gpqa_source, args.gpqa_count))
    rows.extend(load_math500(args.math_source, args.math_count))

    for prompt_id, row in enumerate(rows):
        row["prompt_id"] = prompt_id
        row["split"] = split_for(row["source_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["dataset"]][row["split"]] += 1
    summary = {
        "total": len(rows),
        "counts": {key: dict(value) for key, value in counts.items()},
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
