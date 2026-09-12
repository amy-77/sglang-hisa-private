#!/usr/bin/env python3
"""Send diverse real 128K RULER prompts to a probe-enabled SGLang server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def choose_samples(data_root: Path, count: int, min_tokens: int) -> list[dict]:
    by_task: dict[str, list[dict]] = {}
    for path in sorted(data_root.glob("*/validation.jsonl")):
        rows = []
        with path.open(encoding="utf-8") as f:
            for offset, line in enumerate(f):
                row = json.loads(line)
                length = int(row.get("length_w_model_temp", row.get("length", 0)))
                if length >= min_tokens:
                    rows.append(
                        {
                            "task": path.parent.name,
                            "offset": offset,
                            "path": str(path),
                            "length": length,
                            "input": row["input"],
                            "outputs": row.get("outputs"),
                        }
                    )
        if rows:
            by_task[path.parent.name] = rows

    chosen: list[dict] = []
    round_id = 0
    while len(chosen) < count:
        added = False
        for task in sorted(by_task):
            rows = by_task[task]
            if round_id < len(rows):
                chosen.append(rows[round_id])
                added = True
                if len(chosen) == count:
                    break
        if not added:
            break
        round_id += 1
    if len(chosen) < count:
        raise RuntimeError(
            f"only found {len(chosen)} prompts >= {min_tokens} tokens under {data_root}"
        )
    return chosen


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_manifest(path: Path, records: list[dict], args: argparse.Namespace) -> None:
    path.write_text(
        json.dumps(
            {
                "server": args.server,
                "data_root": str(args.data_root),
                "requested_samples": args.count,
                "min_tokens": args.min_tokens,
                "max_new_tokens": args.max_new_tokens,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:31555")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/DATA/disk0/qyl/data/ruler_deepseek_v3_2_qwen20_128k/128k"
        ),
    )
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=128000)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--start-prompt",
        type=int,
        default=0,
        help="Resume from this 0-based prompt index; probe sample offset must match.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = choose_samples(args.data_root, args.count, args.min_tokens)
    if args.start_prompt < 0 or args.start_prompt > len(samples):
        parser.error("--start-prompt is outside the selected prompt range")
    records: list[dict] = []
    if args.resume and args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        records = list(previous.get("records", []))
    write_manifest(args.out, records, args)

    completed = {
        int(record["sample_id"])
        for record in records
        if record.get("status") == "ok"
    }
    for sample_id, sample in enumerate(samples):
        if sample_id < args.start_prompt or sample_id in completed:
            continue
        metadata = {k: v for k, v in sample.items() if k != "input"}
        record = {"sample_id": sample_id, **metadata, "status": "running"}
        records = [old for old in records if int(old.get("sample_id", -1)) != sample_id]
        records.append(record)
        records.sort(key=lambda item: int(item["sample_id"]))
        write_manifest(args.out, records, args)
        print(
            f"[{sample_id + 1}/{len(samples)}] {sample['task']} "
            f"offset={sample['offset']} length={sample['length']}",
            flush=True,
        )
        payload = {
            "text": sample["input"],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "ignore_eos": True,
            },
        }
        start = time.monotonic()
        error = None
        for attempt in range(args.retries + 1):
            try:
                result = post_json(f"{args.server}/generate", payload, args.timeout)
                record.update(
                    {
                        "status": "ok",
                        "seconds": time.monotonic() - start,
                        "response": result,
                    }
                )
                error = None
                break
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                ConnectionError,
            ) as exc:
                error = repr(exc)
                print(f"  attempt {attempt + 1} failed: {error}", flush=True)
                if attempt < args.retries:
                    time.sleep(5)
        if error is not None:
            record.update(
                {
                    "status": "error",
                    "seconds": time.monotonic() - start,
                    "error": error,
                }
            )
            write_manifest(args.out, records, args)
            raise RuntimeError(f"sample {sample_id} failed: {error}")
        write_manifest(args.out, records, args)
        print(f"  completed in {record['seconds']:.1f}s", flush=True)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
