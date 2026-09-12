"""Drive every prefill regime of the unified per-head scheme against a server.

The sparse threshold is a per-query property, so a prompt shorter than it has an
entirely dense prefill, a prompt slightly longer straddles the boundary inside a
single chunk, and a long prompt spans several chunks with a growing prefix.  Each
case is sent on its own so a failure names the regime that broke.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer

# (label, prompt tokens, why this length matters)
CASES = [
    ("all_dense", 2048, "every row below the 4096 threshold"),
    ("straddle", 6144, "dense rows and sparse rows inside one chunk"),
    ("multi_chunk", 20000, "several prefill chunks, all-sparse tail"),
    ("long", 65536, "long context, coarse pruning under load"),
]


def post(server: str, ids: list[int], max_new_tokens: int, timeout: int) -> dict:
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    request = urllib.request.Request(
        f"{server}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    # A repeated natural sentence keeps tokenisation predictable; the point of
    # the smoke test is the index path, not the content.
    unit = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog near the quiet riverbank. ",
        add_special_tokens=False,
    )

    results = []
    for label, target, why in CASES:
        ids = (unit * (target // len(unit) + 2))[:target]
        started = time.monotonic()
        record = {"case": label, "prompt_tokens": len(ids), "why": why}
        try:
            response = post(args.server, ids, args.max_new_tokens, args.timeout)
            record["status"] = "ok"
            record["text"] = response.get("text", "")[:120]
            record["completion_tokens"] = response.get("meta_info", {}).get(
                "completion_tokens"
            )
        except urllib.error.HTTPError as error:
            record["status"] = "http_error"
            record["error"] = error.read().decode(errors="replace")[:1500]
        except Exception as error:  # noqa: BLE001 - the driver reports, not raises
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"[:1500]
        record["elapsed_seconds"] = round(time.monotonic() - started, 1)
        results.append(record)
        print(
            f"[{label}] tokens={len(ids)} status={record['status']} "
            f"{record['elapsed_seconds']}s",
            flush=True,
        )
        if record["status"] != "ok":
            print(record.get("error", "")[:1500], flush=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    failed = [r["case"] for r in results if r["status"] != "ok"]
    print(f"\nfailed cases: {failed or 'none'}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
