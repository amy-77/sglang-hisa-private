#!/usr/bin/env python3
"""Build the fixed 26-example RULER test prompt manifest (index 7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collect_ruler import prompt_text, selected_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        nargs="+",
        choices=["32k", "128k"],
        default=["32k", "128k"],
    )
    args = parser.parse_args()

    selected = selected_records(args.data_root, "test")
    selected = [item for item in selected if item[1] in args.lengths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for prompt_id, (task, length, index, record) in enumerate(selected):
            row = {
                "prompt_id": prompt_id,
                "dataset": "ruler",
                "task": task,
                "length": length,
                "source_id": f"ruler:{task}:{length}:{index}",
                "split": "test",
                "kind": "long",
                "declared_tokens": int(
                    record.get("length_w_model_temp", record.get("length", 0))
                ),
                "text": prompt_text(record),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} RULER test prompts to {args.out}", flush=True)


if __name__ == "__main__":
    main()
