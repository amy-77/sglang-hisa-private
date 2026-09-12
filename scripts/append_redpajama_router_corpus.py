#!/usr/bin/env python3
"""Append deterministic RedPajama prompts to a zero-shot Router manifest."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_zero_shot_router_corpus import JsonTokenizer, PromptBuilder


def stable_records(path: Path, count: int) -> list[tuple[str, str]]:
    """Keep the records with the smallest hashes without loading a shard."""
    heap: list[tuple[int, str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for offset, line in enumerate(source):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # HTTP range downloads end with one partial JSON line.
                continue
            text = row.get("text", "")
            if len(text) < 2_000:
                continue
            meta = row.get("meta", {})
            identity = str(
                meta.get("url")
                or meta.get("id")
                or meta.get("content_hash")
                or offset
            )
            title = str(
                meta.get("title")
                or meta.get("path")
                or meta.get("repo_name")
                or identity
            )
            score = int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:8], "big"
            )
            item = (-score, title, text)
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    return [(title, text) for _, title, text in sorted(heap, reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count-per-source", type=int, default=100)
    parser.add_argument("--max-context-tokens", type=int, default=131_000)
    args = parser.parse_args()

    existing = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Idempotent reruns keep all original rows and replace only this appendix.
    existing = [
        row
        for row in existing
        if not row["source_id"].startswith("redpajama:")
    ]
    builder = PromptBuilder(JsonTokenizer(args.model))
    sources = {
        "arxiv": args.raw / "arxiv_shard0.jsonl",
        "c4": args.raw / "c4_shard0.jsonl",
        "github": args.raw / "github_shard0.jsonl",
        "stackexchange": args.raw / "stackexchange_head2g.jsonl",
        "wikipedia": args.raw / "wikipedia_head2g.jsonl",
    }
    appended: list[dict] = []
    for subset, path in sources.items():
        pool = stable_records(path, args.count_per_source * 20)
        if subset == "arxiv":
            for index, (title, text) in enumerate(
                pool[: args.count_per_source]
            ):
                appended.append(
                    builder.single_document(
                        "redpajama",
                        subset,
                        f"redpajama:{subset}:{index}:{title}",
                        text,
                        args.max_context_tokens,
                    )
                )
        else:
            for index in range(args.count_per_source):
                source_id = f"redpajama:{subset}:collection:{index}"
                documents = (
                    pool[(index * 17 + offset) % len(pool)]
                    for offset in range(len(pool))
                )
                row = builder.collection(
                    "redpajama",
                    subset,
                    source_id,
                    documents,
                    args.max_context_tokens,
                    "Summarize Document {index} ({title}).",
                )
                if row is not None:
                    appended.append(row)
        print(f"{subset}: {len(appended)} cumulative prompts", flush=True)

    rows = existing + appended
    for prompt_id, row in enumerate(rows):
        row["prompt_id"] = prompt_id
    temporary = args.manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.manifest)
    print(
        json.dumps(
            {
                "original": len(existing),
                "redpajama": len(appended),
                "total": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
