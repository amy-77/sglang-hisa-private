#!/usr/bin/env python3
"""Build a benchmark-free prompt manifest for zero-shot Router training.

The Assignment Router is supervised by dense-attention utility, which is an
intrinsic quantity of the frozen model.  It therefore needs no labels, so the
training corpus can be arbitrary long text.  This builder draws from generic
sources (books, papers, Wikipedia, code, web math, long-instruction data) and
from reasoning problem banks that are disjoint from every evaluation set
(MATH train, AIME 1983-2024, MMLU-Pro).  LongBench-v2, RULER, AIME-2025,
GPQA-Diamond and MATH-500 are never read here.

Output rows follow the pilot manifest contract consumed by
collect_misa_router_samples.py:
  {dataset, task, source_id, split, kind, target_tokens,
   prompt_input_ids, output_ids: []}
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collect_longbench_v2 import JsonTokenizer  # noqa: E402

USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜></think>"

# Long-context targets and their sampling weights (percent).  The mass sits in
# 16K-64K, where the sparse decode path is active and the evaluation suites
# concentrate, with tails at the 4K threshold and at the 128K context limit.
LONG_TARGETS = ((4096, 10), (8192, 15), (16384, 20), (32768, 25), (65536, 20), (131072, 10))
REASONING_TARGETS = (4096, 8192, 16384, 32768)

LONG_INSTRUCTIONS = (
    "Summarize the following text in a few paragraphs.",
    "Describe the main ideas, people, and events in the following text.",
    "Continue the following text for a few more paragraphs in the same style.",
    "List the most important claims made in the following text and explain them.",
)


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def bucket(value: str, modulus: int, offset: int = 0) -> int:
    return int.from_bytes(digest(value)[offset : offset + 4], "big") % modulus


def split_for(source_id: str) -> str:
    return "validation" if bucket(source_id, 10, 8) == 0 else "train"


def pick_long_target(source_id: str, max_tokens: int) -> int:
    total = sum(weight for _, weight in LONG_TARGETS)
    roll = bucket(source_id, total, 4)
    for target, weight in LONG_TARGETS:
        if roll < weight:
            return min(target, max_tokens)
        roll -= weight
    return min(LONG_TARGETS[-1][0], max_tokens)


def stable_sample(items: list, count: int, key) -> list:
    return sorted(items, key=lambda item: digest(key(item)))[:count]


class PromptBuilder:
    def __init__(self, tokenizer: JsonTokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, special: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=special)

    def wrap(self, instruction: str) -> tuple[list[int], list[int]]:
        prefix = self.encode(f"{USER}{instruction}\n\n<text>\n", special=True)
        suffix = self.encode(f"\n</text>\n{ASSISTANT}")
        return prefix, suffix

    def single_document(
        self, dataset: str, task: str, source_id: str, text: str, max_tokens: int
    ) -> dict:
        """One long document trimmed to a hashed target length."""
        target = pick_long_target(source_id, max_tokens)
        instruction = LONG_INSTRUCTIONS[bucket(source_id, len(LONG_INSTRUCTIONS), 12)]
        prefix, suffix = self.wrap(instruction)
        budget = target - len(prefix) - len(suffix)
        # ~4.5 chars/token upper bound keeps the tokenizer call bounded.
        body = self.encode(text[: budget * 5])[:budget]
        return self.row(dataset, task, source_id, "long", target, prefix + body + suffix)

    def collection(
        self,
        dataset: str,
        task: str,
        source_id: str,
        docs: Iterable[tuple[str, str]],
        max_tokens: int,
        ask: str,
    ) -> dict | None:
        """Concatenate many titled documents; ask about one named document.

        The question names a document by its title/path so the instruction is a
        natural retrieval task with no external label.
        """
        target = pick_long_target(source_id, max_tokens)
        prefix = self.encode(
            f"{USER}Several documents are provided below, each starting with a "
            f"'### Document' header.\n\n",
            special=True,
        )
        titles: list[str] = []
        body: list[int] = []
        reserve = 256  # room for the trailing question and long source titles
        for title, text in docs:
            remaining = target - len(prefix) - len(body) - reserve
            if remaining <= 256:
                break
            chunk = self.encode(f"### Document {len(titles) + 1}: {title}\n{text[: remaining * 5]}\n\n")
            body.extend(chunk[:remaining])
            titles.append(title)
        if len(titles) < 2:
            return None
        chosen = bucket(source_id, len(titles), 16)
        suffix = self.encode(
            f"\n{ask.format(index=chosen + 1, title=titles[chosen])}{ASSISTANT}"
        )
        ids = (prefix + body + suffix)[:target]
        return self.row(dataset, task, source_id, "long", target, ids)

    def packed_problem(
        self,
        dataset: str,
        task: str,
        source_id: str,
        text: str,
        peers: list[list[int]],
        max_tokens: int,
    ) -> dict:
        """Pack a short problem past the sparse threshold with unanswered peers."""
        target = min(REASONING_TARGETS[bucket(source_id, len(REASONING_TARGETS), 4)], max_tokens)
        prefix = self.encode(
            f"{USER}Several unrelated practice questions are provided as context. "
            "Ignore them and solve only the final TARGET question.\n\n<PRACTICE_CONTEXT>\n",
            special=True,
        )
        suffix = self.encode(f"\n</PRACTICE_CONTEXT>\n\n<TARGET>\n{text}\n</TARGET>{ASSISTANT}")
        budget = target - len(prefix) - len(suffix)
        filler: list[int] = []
        start = bucket(source_id, len(peers), 2)
        offset = 0
        while len(filler) < budget:
            filler.extend(peers[(start + offset) % len(peers)])
            offset += 1
        return self.row(dataset, task, source_id, "reasoning", target, prefix + filler[:budget] + suffix)

    @staticmethod
    def row(dataset, task, source_id, kind, target, ids) -> dict:
        return {
            "dataset": dataset,
            "task": task,
            "source_id": source_id,
            "split": split_for(source_id),
            "kind": kind,
            "target_tokens": target,
            "prompt_input_ids": ids,
            "output_ids": [],
        }


def load_pg19(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(raw / "emozilla__pg19/data/*.parquet")))]
    df = pd.concat(frames, ignore_index=True)
    books = [(str(r.short_book_title), str(r.text)) for r in df.itertuples() if len(r.text) > 20_000]
    rows = []
    for title, text in stable_sample(books, count, key=lambda b: "pg19:" + b[0]):
        sid = f"pg19:{title}"
        # Start at a hashed offset so prompts are not all book openings.
        start = bucket(sid, max(1, len(text) - 600_000), 20)
        rows.append(builder.single_document("pg19", "book", sid, text[start:], max_tokens))
    return rows


def load_arxiv(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    df = pd.read_parquet(glob.glob(str(raw / "ccdv__arxiv-summarization/document/*.parquet"))[0])
    papers = [str(a) for a in df["article"] if len(a) > 8_000]
    papers = stable_sample(papers, min(len(papers), count * 12), key=lambda a: a[:200])
    rows = []
    for i in range(count):
        sid = f"arxiv:collection:{i}"
        if bucket(sid, 2, 24) == 0:
            rows.append(builder.single_document("arxiv", "paper", sid, papers[i], max_tokens))
        else:
            docs = ((f"paper {j}", papers[(i * 7 + j) % len(papers)]) for j in range(1, 40))
            row = builder.collection(
                "arxiv", "paper_collection", sid, docs, max_tokens,
                "Summarize Document {index} ({title}) and state its main contribution.",
            )
            if row:
                rows.append(row)
    return rows


def load_wikipedia(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    df = pd.read_parquet(glob.glob(str(raw / "wikimedia__wikipedia/20231101.en/*.parquet"))[0])
    df = df[df["text"].str.len() > 3_000]
    articles = list(zip(df["title"].astype(str), df["text"].astype(str)))
    articles = stable_sample(articles, min(len(articles), count * 60), key=lambda a: "wiki:" + a[0])
    rows = []
    for i in range(count):
        sid = f"wikipedia:collection:{i}"
        docs = (articles[(i * 53 + j) % len(articles)] for j in range(400))
        row = builder.collection(
            "wikipedia", "article_collection", sid, docs, max_tokens,
            "Summarize the document titled '{title}' (Document {index}) in one paragraph.",
        )
        if row:
            rows.append(row)
    return rows


def load_code(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    df = pd.read_parquet(glob.glob(str(raw / "codeparrot__github-code-clean/data/*.parquet"))[0])
    df = df[df["language"].isin(["Python", "C++", "Java", "GO", "Rust", "TypeScript", "JavaScript", "C"])]
    by_repo: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in df.itertuples():
        by_repo[str(r.repo_name)].append((str(r.path), str(r.code)))
    repos = [(name, files) for name, files in by_repo.items() if sum(len(c) for _, c in files) > 40_000]
    repos = stable_sample(repos, min(len(repos), count), key=lambda r: "code:" + r[0])
    rows = []
    for name, files in repos:
        sid = f"code:{name}"
        files = sorted(files, key=lambda f: digest(sid + f[0]))
        row = builder.collection(
            "code", "repository", sid, files, max_tokens,
            "Explain what the file '{title}' (Document {index}) does and how it relates to the other files.",
        )
        if row:
            rows.append(row)
    return rows


def load_openwebmath(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    df = pd.read_parquet(glob.glob(str(raw / "open-web-math__open-web-math/data/*.parquet"))[0])
    df = df[df["text"].str.len() > 4_000]
    docs = list(zip(df["url"].astype(str), df["text"].astype(str)))
    docs = stable_sample(docs, min(len(docs), count * 40), key=lambda d: "owm:" + d[0])
    rows = []
    for i in range(count):
        sid = f"openwebmath:collection:{i}"
        row = builder.collection(
            "openwebmath", "math_web_collection", sid,
            (docs[(i * 41 + j) % len(docs)] for j in range(300)), max_tokens,
            "Explain the main mathematical result discussed in Document {index} ({title}).",
        )
        if row:
            rows.append(row)
    return rows


def load_longalign(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    rows = []
    with (raw / "zai-org__LongAlign-10k/long.jsonl").open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    records = [r for r in records if r["messages"] and r["messages"][0]["role"] == "user"]
    for r in stable_sample(records, count, key=lambda r: f"longalign:{r['id']}"):
        sid = f"longalign:{r['id']}"
        ids = builder.encode(USER + r["messages"][0]["content"] + ASSISTANT, special=True)
        if len(ids) > max_tokens:
            head = max_tokens // 2
            ids = ids[:head] + ids[-(max_tokens - head):]
        if len(ids) < 4096:
            continue
        rows.append(builder.row("longalign", str(r.get("dataset", "longalign")), sid, "long", len(ids), ids))
    return rows


def load_longalpaca(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    records = json.loads((raw / "Yukang__LongAlpaca-12k/LongAlpaca-12k.json").read_text(encoding="utf-8"))
    records = [(i, r) for i, r in enumerate(records) if len(r["instruction"]) > 16_000]
    rows = []
    for i, r in stable_sample(records, count, key=lambda x: f"longalpaca:{x[0]}"):
        sid = f"longalpaca:{i}"
        ids = builder.encode(USER + r["instruction"] + ASSISTANT, special=True)
        if len(ids) > max_tokens:
            head = max_tokens // 2
            ids = ids[:head] + ids[-(max_tokens - head):]
        if len(ids) < 4096:
            continue
        rows.append(builder.row("longalpaca", "long_instruction", sid, "long", len(ids), ids))
    return rows


MATH_TEMPLATE = (
    "Solve the following mathematics problem. Explain your reasoning step by step "
    "and put the final answer in \\boxed{{}}.\n\n{problem}"
)
AIME_TEMPLATE = (
    "Solve the following AIME problem step by step. The last line of your response "
    "should be of the form Answer: $ANSWER.\n\n{problem}"
)


def pack_bank(
    builder: PromptBuilder, dataset: str, items: list[tuple[str, str, str]], count: int, max_tokens: int
) -> list[dict]:
    """items: (source_id, task, prompt_text).  Distractors come from the same split."""
    chosen = stable_sample(items, count, key=lambda x: x[0])
    by_split: dict[str, list[list[int]]] = defaultdict(list)
    for sid, _, text in chosen:
        by_split[split_for(sid)].append(builder.encode(f"\n[Unanswered practice question]\n{text}\n"))
    rows = []
    for sid, task, text in chosen:
        peers = by_split[split_for(sid)]
        if len(peers) < 2:
            continue
        rows.append(builder.packed_problem(dataset, task, sid, text, peers, max_tokens))
    return rows


def load_math_train(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(raw / "EleutherAI__hendrycks_math/*/train-*.parquet"))):
        df = pd.read_parquet(f)
        for i, r in enumerate(df.itertuples()):
            items.append((f"math_train:{Path(f).parent.name}:{i}", str(r.type), MATH_TEMPLATE.format(problem=r.problem)))
    return pack_bank(builder, "math_train", items, count, max_tokens)


def load_aime_hist(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    df = pd.read_csv(raw / "di-zhang-fdu__AIME_1983_2024/AIME_Dataset_1983_2024.csv")
    df = df.rename(columns={"Problem Number": "Number"})
    assert int(df["Year"].max()) <= 2024, "AIME-2025 must stay held out"
    items = [
        (f"aime_hist:{r.Year}:{r.Part}:{r.Number}", f"aime_{r.Year}", AIME_TEMPLATE.format(problem=r.Question))
        for r in df.itertuples()
    ]
    return pack_bank(builder, "aime_hist", items, count, max_tokens)


def load_mmlu_pro(raw: Path, builder: PromptBuilder, count: int, max_tokens: int) -> list[dict]:
    df = pd.read_parquet(glob.glob(str(raw / "TIGER-Lab__MMLU-Pro/data/*.parquet"))[0])
    df = df[df["category"].isin(["physics", "chemistry", "biology", "computer science", "engineering", "math"])]
    items = []
    for r in df.itertuples():
        options = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(r.options))
        text = (
            "Answer the following graduate-level multiple-choice question. Explain your "
            f"reasoning, then state the chosen letter.\n\n{r.question}\n\n{options}"
        )
        items.append((f"mmlu_pro:{r.question_id}", str(r.category), text))
    return pack_bank(builder, "mmlu_pro", items, count, max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=131_000)
    parser.add_argument("--out", type=Path, required=True)
    for name, default in (
        ("pg19", 200), ("arxiv", 200), ("wikipedia", 150), ("code", 150), ("openwebmath", 100),
        ("longalign", 200), ("longalpaca", 100), ("math-train", 250), ("aime-hist", 150), ("mmlu-pro", 200),
    ):
        parser.add_argument(f"--{name}-count", type=int, default=default)
    args = parser.parse_args()

    builder = PromptBuilder(JsonTokenizer(args.model))
    loaders = (
        ("pg19", load_pg19), ("arxiv", load_arxiv), ("wikipedia", load_wikipedia), ("code", load_code),
        ("openwebmath", load_openwebmath), ("longalign", load_longalign), ("longalpaca", load_longalpaca),
        ("math_train", load_math_train), ("aime_hist", load_aime_hist), ("mmlu_pro", load_mmlu_pro),
    )
    rows: list[dict] = []
    for name, loader in loaders:
        count = getattr(args, f"{name}_count")
        produced = loader(args.raw, builder, count, args.max_context_tokens)
        rows.extend(produced)
        print(f"{name}: {len(produced)} prompts", flush=True)

    for prompt_id, row in enumerate(rows):
        row["prompt_id"] = prompt_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: dict = {"total": len(rows), "datasets": {}, "length_buckets": defaultdict(int), "splits": defaultdict(int)}
    for row in rows:
        d = summary["datasets"].setdefault(row["dataset"], {"train": 0, "validation": 0})
        d[row["split"]] += 1
        summary["splits"][row["split"]] += 1
        summary["length_buckets"][str(row["target_tokens"])] += 1
    summary["length_buckets"] = dict(sorted(summary["length_buckets"].items(), key=lambda kv: int(kv[0])))
    summary["splits"] = dict(summary["splits"])
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
