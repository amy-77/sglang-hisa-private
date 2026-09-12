#!/usr/bin/env python3
"""Generate trajectories, then replay identified decode queries into a probe.

I/O:
  generate: prompts.jsonl[text,...] -> trajectories.jsonl[prompt_input_ids,output_ids,...]
  replay:   trajectories.jsonl -> samples.jsonl[request_id,source_id,split,...]
            The probe server writes one [64 candidates, 16 local heads] record
            per TP rank; training later aggregates it into utility [8,64].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_generate(
    server: str,
    input_ids: list[int],
    max_new_tokens: int,
    timeout: int,
    retries: int,
    ignore_eos: bool = False,
    return_token_ids: bool = False,
    request_id: str | None = None,
) -> dict:
    request_payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": ignore_eos,
        },
    }
    if request_id is not None:
        request_payload["rid"] = request_id
    if return_token_ids:
        # The native /generate response does not expose ``output_ids`` unless
        # logprobs are requested.  Each output logprob tuple carries its token
        # id at index 1, which lets replay preserve exact generated prefixes.
        request_payload.update(
            {
                "return_logprob": True,
                "top_logprobs_num": 0,
            }
        )
    payload = json.dumps(request_payload).encode("utf-8")
    error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                f"{server.rstrip('/')}/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = exc
            if attempt < retries:
                time.sleep(5)
    raise RuntimeError(f"generation failed after {retries + 1} attempts: {error}")


def prompt_ids(tokenizer, text: str, max_tokens: int) -> list[int]:
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        # DeepSeek-V3.2 checkpoints do not ship a tokenizer chat_template.
        # Match evaluate_ruler_e2e.py and the router corpus builder exactly.
        ids = tokenizer.encode(
            f"<｜User｜>{text}<｜Assistant｜></think>",
            add_special_tokens=True,
        )
    if len(ids) > max_tokens:
        head = max_tokens // 2
        ids = ids[:head] + ids[-(max_tokens - head) :]
    return ids


def output_ids(tokenizer, response: dict) -> list[int]:
    meta = response.get("meta_info", {})
    for key in ("output_token_ids", "output_ids"):
        if key in meta:
            return [int(token) for token in meta[key]]
        if key in response:
            return [int(token) for token in response[key]]
    if "output_token_logprobs" in meta:
        return [int(item[1]) for item in meta["output_token_logprobs"]]
    return tokenizer.encode(response.get("text", ""), add_special_tokens=False)


def generate_trajectories(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    completed = set()
    if args.resume and args.out.exists():
        # Old pilot files may contain prompt-only rows with empty output_ids.
        # Keep one valid trajectory per source and regenerate every invalid one.
        valid_by_source = {
            row["source_id"]: row
            for row in read_jsonl(args.out)
            if row.get("output_ids")
        }
        write_jsonl(args.out, list(valid_by_source.values()))
        completed = set(valid_by_source)
    if args.out.exists() and not args.resume:
        args.out.unlink()
    prompts = read_jsonl(args.prompts)
    def generate_one(index: int, row: dict) -> tuple[int, dict]:
        long_context = is_long_context(row)
        max_new_tokens = (
            args.long_max_new_tokens
            if long_context
            else args.reasoning_max_new_tokens
        )
        if "prompt_input_ids" in row:
            ids = [int(token) for token in row["prompt_input_ids"]]
            if len(ids) > args.max_prompt_tokens:
                head = args.max_prompt_tokens // 2
                ids = ids[:head] + ids[-(args.max_prompt_tokens - head) :]
        else:
            ids = prompt_ids(tokenizer, row["text"], args.max_prompt_tokens)
        start = time.monotonic()
        response = post_generate(
            args.server,
            ids,
            max_new_tokens,
            args.timeout,
            args.retries,
            return_token_ids=True,
        )
        generated = output_ids(tokenizer, response)
        if max_new_tokens and not generated:
            raise RuntimeError(
                f"server returned no output token ids for {row['source_id']}"
            )
        return (
            index,
            {
                **{key: value for key, value in row.items() if key != "text"},
                "prompt_input_ids": ids,
                "output_ids": generated,
                "generated_text": response.get("text", ""),
                "elapsed_seconds": time.monotonic() - start,
            },
        )

    pending = [
        (index, row)
        for index, row in enumerate(prompts)
        if row["source_id"] not in completed
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(generate_one, index, row): row
            for index, row in pending
        }
        for future in as_completed(futures):
            index, result = future.result()
            append_jsonl(args.out, result)
            print(
                f"[{index + 1}/{len(prompts)}] {result['dataset']} "
                f"prompt={len(result['prompt_input_ids'])} "
                f"output={len(result['output_ids'])}",
                flush=True,
            )


def is_long_context(row: dict) -> bool:
    """Long-context rows decode briefly; reasoning rows decode a long answer.

    New manifests carry an explicit ``kind``; the pilot manifests are
    recognised by dataset name.
    """
    kind = row.get("kind")
    if kind is not None:
        return kind == "long"
    return row["dataset"] in ("ruler", "longbench_v2")


def replay_positions(row: dict, output_length: int) -> list[int]:
    requested = (0, 4, 16, 64) if is_long_context(row) else (0, 32, 128, 512, 2048)
    return [position for position in requested if position <= output_length]


def replay_plan(
    trajectories: list[dict],
    *,
    max_replay_tokens: int,
    min_query_seq_len: int,
) -> list[dict]:
    """Build stable, uniquely identified decode queries.

    A request asks for two output tokens. The first output token becomes the
    query token for the one required decode step, so its visible length is
    ``len(input_ids) + 1``.
    """
    if max_replay_tokens < min_query_seq_len:
        raise ValueError("max_replay_tokens must be at least min_query_seq_len")
    plan = []
    for row in trajectories:
        output_length = len(row["output_ids"])
        positions = set(replay_positions(row, output_length))
        retained_prompt_len = min(
            len(row["prompt_input_ids"]), max_replay_tokens
        )
        first_sparse_position = max(
            0, min_query_seq_len - retained_prompt_len - 1
        )
        if first_sparse_position <= output_length:
            positions.add(first_sparse_position)
        source_hash = hashlib.sha256(
            json.dumps(
                row["prompt_input_ids"], separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        for position in sorted(positions):
            ids = row["prompt_input_ids"] + row["output_ids"][:position]
            ids = ids[-max_replay_tokens:]
            expected_seq_len = len(ids) + 1
            if expected_seq_len < min_query_seq_len:
                continue
            digest = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            sample_id = len(plan)
            plan.append(
                {
                    "sample_id": sample_id,
                    "request_id": (
                        f"group16-probe-{sample_id}-{expected_seq_len}-{digest}"
                    ),
                    "prefix_hash": digest,
                    "source_hash": source_hash,
                    "expected_seq_len": expected_seq_len,
                    "input_ids": ids,
                    "position": position,
                    "trajectory": row,
                }
            )
    return plan


def print_replay_stats(
    trajectories: list[dict], plan: list[dict], *, file=None
) -> None:
    groups: dict[tuple[str, str, str], dict[str, set | int]] = {}
    for row in trajectories:
        key = (
            str(row.get("dataset", "unknown")),
            str(row.get("split", "unknown")),
            str(row.get("length") or "unknown"),
        )
        groups.setdefault(key, {"sources": set(), "kept": set(), "samples": 0})
        groups[key]["sources"].add(str(row["source_id"]))
    for item in plan:
        row = item["trajectory"]
        key = (
            str(row.get("dataset", "unknown")),
            str(row.get("split", "unknown")),
            str(row.get("length") or "unknown"),
        )
        groups[key]["kept"].add(str(row["source_id"]))
        groups[key]["samples"] += 1
    for key, values in sorted(groups.items()):
        total = len(values["sources"])
        kept = len(values["kept"])
        print(
            f"coverage dataset={key[0]} split={key[1]} length={key[2]} "
            f"trajectories={total} kept={kept} dropped={total-kept} "
            f"samples={values['samples']}",
            file=file,
        )


def replay(args: argparse.Namespace) -> None:
    trajectories = read_jsonl(args.trajectories)
    empty = [row["source_id"] for row in trajectories if not row.get("output_ids")]
    if empty and not args.allow_empty_output:
        raise RuntimeError(
            f"{len(empty)}/{len(trajectories)} trajectories have no output_ids; "
            "run the generate stage before replay, or use "
            "--allow-empty-output for an explicit prompt-end-only probe"
        )
    replay_rows = replay_plan(
        trajectories,
        max_replay_tokens=args.max_replay_tokens,
        min_query_seq_len=args.min_query_seq_len,
    )
    print_replay_stats(trajectories, replay_rows)

    existing = read_jsonl(args.out) if args.resume and args.out.exists() else []
    completed = {
        str(row["request_id"]) for row in existing if row["status"] == "ok"
    }
    if args.out.exists() and not args.resume:
        args.out.unlink()
    trajectory_mismatches = 0
    for item in replay_rows:
        sample_id = item["sample_id"]
        request_id = item["request_id"]
        if request_id in completed:
            continue
        row = item["trajectory"]
        ids = item["input_ids"]
        position = item["position"]
        start = time.monotonic()
        response = post_generate(
            args.server,
            ids,
            2,
            args.timeout,
            args.retries,
            ignore_eos=True,
            return_token_ids=True,
            request_id=request_id,
        )
        generated_ids = output_ids(None, response)
        if len(generated_ids) < 2:
            raise RuntimeError(
                f"{request_id} did not execute the required decode step"
            )
        expected_query_token = (
            int(row["output_ids"][position])
            if position < len(row["output_ids"])
            else None
        )
        trajectory_match = (
            expected_query_token is None
            or generated_ids[0] == expected_query_token
        )
        trajectory_mismatches += int(not trajectory_match)
        append_jsonl(
            args.out,
            {
                "sample_id": sample_id,
                "request_id": request_id,
                "prefix_hash": item["prefix_hash"],
                "source_hash": item["source_hash"],
                "expected_seq_len": item["expected_seq_len"],
                "query_token_id": generated_ids[0],
                "trajectory_query_token_id": expected_query_token,
                "trajectory_token_match": trajectory_match,
                "prompt_id": row["prompt_id"],
                "source_id": row["source_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "length": row.get("length"),
                "split": row["split"],
                "decode_position": position,
                "replay_tokens": len(ids),
                "status": "ok",
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        print(
            f"[{sample_id + 1}/{len(replay_rows)}] {row['dataset']} "
            f"position={position} tokens={len(ids)}",
            flush=True,
        )
    print(
        f"trajectory query-token mismatches: "
        f"{trajectory_mismatches}/{len(replay_rows)}",
        flush=True,
    )


def count_replay(args: argparse.Namespace) -> None:
    trajectories = read_jsonl(args.trajectories)
    plan = replay_plan(
        trajectories,
        max_replay_tokens=args.max_replay_tokens,
        min_query_seq_len=args.min_query_seq_len,
    )
    print_replay_stats(trajectories, plan, file=sys.stderr)
    print(len(plan))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--prompts", type=Path, required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--server", default="http://127.0.0.1:31720")
    generate.add_argument("--long-max-new-tokens", type=int, default=96)
    generate.add_argument("--reasoning-max-new-tokens", type=int, default=4096)
    generate.add_argument("--concurrency", type=int, default=1)
    generate.add_argument("--max-prompt-tokens", type=int, default=131000)
    generate.add_argument("--timeout", type=int, default=3600)
    generate.add_argument("--retries", type=int, default=2)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--out", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--trajectories", type=Path, required=True)
    replay_parser.add_argument("--server", default="http://127.0.0.1:31720")
    replay_parser.add_argument("--max-replay-tokens", type=int, default=131070)
    replay_parser.add_argument("--min-query-seq-len", type=int, default=4096)
    replay_parser.add_argument("--timeout", type=int, default=3600)
    replay_parser.add_argument("--retries", type=int, default=2)
    replay_parser.add_argument("--resume", action="store_true")
    replay_parser.add_argument("--allow-empty-output", action="store_true")
    replay_parser.add_argument("--out", type=Path, required=True)

    count_parser = subparsers.add_parser("count")
    count_parser.add_argument("--trajectories", type=Path, required=True)
    count_parser.add_argument("--max-replay-tokens", type=int, default=131070)
    count_parser.add_argument("--min-query-seq-len", type=int, default=4096)

    args = parser.parse_args()
    if args.command == "generate":
        generate_trajectories(args)
    elif args.command == "replay":
        replay(args)
    else:
        count_replay(args)


if __name__ == "__main__":
    main()
