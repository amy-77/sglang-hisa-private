#!/usr/bin/env python3
"""Build tokenized long and packed-short prompts for Router trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collect_longbench_v2 import (  # noqa: E402
    JsonTokenizer,
    encode_prompt,
    select_stratified,
)
from collect_ruler import prompt_text, selected_records  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def held_out_longbench(records: list[dict], excluded: set[str]) -> list[dict]:
    groups: dict[tuple[str, str], deque] = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        if str(row["_id"]) not in excluded:
            grouped[(row["difficulty"], row["length"])].append(row)
    for key, values in grouped.items():
        values.sort(
            key=lambda row: hashlib.sha256(str(row["_id"]).encode()).digest()
        )
        groups[key] = deque(values)
    selected = []
    while groups and len(selected) < 64:
        for key in list(sorted(groups)):
            selected.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
            if len(selected) == 64:
                break
    return selected


def pack_short_prompts(
    tokenizer: JsonTokenizer,
    manifest: Path,
    max_context_tokens: int,
) -> list[dict]:
    """Pack short reasoning tasks past the sparse transition without labels.

    Distractors come only from the target's own split. Their answers are never
    included, so validation/test supervision cannot leak into training.
    """
    with manifest.open(encoding="utf-8") as input_file:
        source_rows = [json.loads(line) for line in input_file if line.strip()]
    short_datasets = {"aime_2025", "gpqa_diamond", "math_500"}
    source_rows = [
        row
        for row in source_rows
        if row["dataset"] in short_datasets
        and row["split"] in ("train", "validation")
    ]
    by_split = {
        split: [row for row in source_rows if row["split"] == split]
        for split in ("train", "validation")
    }
    practice_ids = {
        row["source_id"]: tokenizer.encode(
            "\n[Unanswered practice question]\n" + row["text"] + "\n",
            add_special_tokens=False,
        )
        for row in source_rows
    }
    target_lengths = (4096, 8192, 16384, 32768)
    packed = []
    for row in source_rows:
        digest = hashlib.sha256(row["source_id"].encode()).digest()
        target_length = min(
            target_lengths[int.from_bytes(digest[:2], "big") % len(target_lengths)],
            max_context_tokens,
        )
        prefix = (
            "<｜User｜>Several unrelated practice questions are provided as "
            "context. Ignore them and solve only the final TARGET question.\n\n"
            "<PRACTICE_CONTEXT>\n"
        )
        suffix = (
            "\n</PRACTICE_CONTEXT>\n\n<TARGET>\n"
            + row["text"]
            + "\n</TARGET><｜Assistant｜></think>"
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        filler_budget = target_length - len(prefix_ids) - len(suffix_ids)
        if filler_budget < 0:
            raise RuntimeError(
                f"{row['source_id']} target exceeds packed length {target_length}"
            )
        peers = [
            peer
            for peer in by_split[row["split"]]
            if peer["source_id"] != row["source_id"]
        ]
        start = int.from_bytes(digest[2:6], "big") % len(peers)
        filler_ids: list[int] = []
        offset = 0
        while len(filler_ids) < filler_budget:
            peer = peers[(start + offset) % len(peers)]
            filler_ids.extend(practice_ids[peer["source_id"]])
            offset += 1
        input_ids = prefix_ids + filler_ids[:filler_budget] + suffix_ids
        packed.append(
            {
                **{key: value for key, value in row.items() if key != "text"},
                "prompt_input_ids": input_ids,
                "packed_context_tokens": target_length,
                "output_ids": [],
            }
        )
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--longbench-data", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--short-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=131_070)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--longbench-test-out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = JsonTokenizer(args.model)
    longbench = json.loads(args.longbench_data.read_text(encoding="utf-8"))
    rows = []
    excluded_longbench_ids = set()

    for split in ("train", "validation"):
        selected = select_stratified(longbench, split)
        for record in selected:
            excluded_longbench_ids.add(str(record["_id"]))
            rows.append(
                {
                    "dataset": "longbench_v2",
                    "task": record["domain"],
                    "source_id": str(record["_id"]),
                    "split": split,
                    "prompt_input_ids": encode_prompt(
                        tokenizer,
                        record,
                        args.max_context_tokens,
                        chat_template=True,
                    ),
                    "output_ids": [],
                }
            )
    longbench_test = held_out_longbench(longbench, excluded_longbench_ids)
    args.longbench_test_out.parent.mkdir(parents=True, exist_ok=True)
    args.longbench_test_out.write_text(
        json.dumps(longbench_test, ensure_ascii=False), encoding="utf-8"
    )

    for split in ("train", "validation"):
        for task, length, index, record in selected_records(
            args.ruler_root, split
        ):
            text = (
                "<｜User｜>"
                + prompt_text(record)
                + "<｜Assistant｜></think>"
            )
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            if len(input_ids) > args.max_context_tokens:
                head = args.max_context_tokens // 2
                input_ids = (
                    input_ids[:head]
                    + input_ids[-(args.max_context_tokens - head) :]
                )
            rows.append(
                {
                    "dataset": "ruler",
                    "task": task,
                    "length": length,
                    "source_id": f"ruler:{task}:{length}:{index}",
                    "split": split,
                    "prompt_input_ids": input_ids,
                    "output_ids": [],
                }
            )

    rows.extend(
        pack_short_prompts(
            tokenizer,
            args.short_manifest,
            args.max_context_tokens,
        )
    )
    for prompt_id, row in enumerate(rows):
        row["prompt_id"] = prompt_id
    write_jsonl(args.out, rows)
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    print(json.dumps({"total": len(rows), "splits": counts}, indent=2))


if __name__ == "__main__":
    main()
