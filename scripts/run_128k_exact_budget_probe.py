#!/usr/bin/env python3
"""Send diverse real RULER and LongBench-v2 128K prompts to the DSA probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collect_longbench_v2 import JsonTokenizer, encode_prompt, generate  # noqa: E402


def post_text(server: str, text: str, max_new_tokens: int) -> dict:
    payload = json.dumps(
        {
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(
        f"{server.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def round_robin(groups: dict[str, list[dict]]):
    queues = {key: deque(value) for key, value in sorted(groups.items())}
    while queues:
        for key in list(queues):
            if queues[key]:
                yield queues[key].popleft()
            if not queues[key]:
                del queues[key]


def choose_ruler(root: Path, count: int, min_tokens: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for path in sorted(root.glob("*/validation.jsonl")):
        rows = []
        for offset, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            length = int(row.get("length_w_model_temp", row.get("length", 0)))
            if length >= min_tokens:
                rows.append(
                    {
                        "dataset": "ruler",
                        "task": path.parent.name,
                        "source_id": f"ruler:{path.parent.name}:{offset}",
                        "declared_tokens": length,
                        "text": row["input"],
                    }
                )
        if rows:
            groups[path.parent.name] = rows
    chosen = list(round_robin(groups))[:count]
    if len(chosen) != count:
        raise RuntimeError(f"found only {len(chosen)} qualifying RULER prompts")
    return chosen


def choose_longbench(
    path: Path,
    tokenizer: JsonTokenizer,
    count: int,
    min_tokens: int,
    max_context_tokens: int,
) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row.get("length") == "long":
            groups[str(row.get("domain", row.get("sub_domain", "unknown")))].append(row)

    chosen = []
    for row in round_robin(groups):
        input_ids = encode_prompt(
            tokenizer, row, max_context_tokens, chat_template=True
        )
        if len(input_ids) < min_tokens:
            continue
        chosen.append(
            {
                "dataset": "longbench_v2",
                "task": str(row.get("domain", row.get("sub_domain", "unknown"))),
                "source_id": str(row["_id"]),
                "declared_tokens": len(input_ids),
                "input_ids": input_ids,
            }
        )
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"found only {len(chosen)} qualifying LongBench-v2 prompts")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:31555")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--longbench-data", type=Path, required=True)
    parser.add_argument("--ruler-count", type=int, default=8)
    parser.add_argument("--longbench-count", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=120000)
    parser.add_argument("--max-context-tokens", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = JsonTokenizer(str(args.model))
    ruler = choose_ruler(args.ruler_root, args.ruler_count, args.min_tokens)
    longbench = choose_longbench(
        args.longbench_data,
        tokenizer,
        args.longbench_count,
        args.min_tokens,
        args.max_context_tokens,
    )
    prompts = [item for pair in zip(ruler, longbench) for item in pair]
    prompts.extend(ruler[len(longbench) :])
    prompts.extend(longbench[len(ruler) :])

    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        records = [
            {
                **{k: v for k, v in prompt.items() if k not in ("text", "input_ids")},
                "prompt_id": prompt_id,
                "probe_sample_start": prompt_id * args.max_new_tokens,
                "probe_sample_end": (prompt_id + 1) * args.max_new_tokens - 1,
            }
            for prompt_id, prompt in enumerate(prompts)
        ]
        args.output.write_text(
            json.dumps(
                {"max_new_tokens": args.max_new_tokens, "records": records},
                indent=2,
                ensure_ascii=False,
            )
        )
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return

    for prompt_id, prompt in enumerate(prompts):
        start = time.monotonic()
        if prompt["dataset"] == "ruler":
            result = post_text(
                args.server, prompt.pop("text"), args.max_new_tokens
            )
        else:
            result = generate(
                args.server, prompt.pop("input_ids"), args.max_new_tokens
            )
        record = {
            **prompt,
            "prompt_id": prompt_id,
            "probe_sample_start": prompt_id * args.max_new_tokens,
            "probe_sample_end": (prompt_id + 1) * args.max_new_tokens - 1,
            "elapsed_seconds": time.monotonic() - start,
            "response": result,
        }
        records.append(record)
        args.output.write_text(
            json.dumps(
                {
                    "max_new_tokens": args.max_new_tokens,
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(
            f"[{prompt_id + 1}/{len(prompts)}] {record['dataset']} "
            f"{record['task']} tokens={record['declared_tokens']} "
            f"time={record['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
