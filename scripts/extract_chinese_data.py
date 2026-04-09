#!/usr/bin/env python3
"""Extract Chinese conversations from nemotron-v2-jsonl dataset.

Scans all JSONL files in the source directory and outputs conversations
containing Chinese characters to a single output JSONL file.
"""
import argparse
import json
import os
import re
import sys

ZH_PATTERN = re.compile(r'[\u4e00-\u9fff]')

# Japanese-only files to exclude (contain kanji but not Chinese conversations)
JA_FILES = {"multilingual_ja.jsonl"}


def has_chinese(conversations: list[dict]) -> bool:
    """Check if any user/assistant message contains Chinese characters."""
    for msg in conversations:
        if msg.get("role") in ("user", "assistant"):
            if ZH_PATTERN.search(msg.get("content", "")):
                return True
    return False


def extract_chinese(src_dir: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    jsonl_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jsonl") and f not in JA_FILES)
    total_zh = 0
    total_all = 0

    with open(output_path, "w") as out:
        for fname in jsonl_files:
            path = os.path.join(src_dir, fname)
            file_zh = 0
            file_total = 0

            with open(path) as f:
                for line in f:
                    file_total += 1
                    entry = json.loads(line)
                    convs = entry.get("conversations", entry.get("messages", []))
                    if has_chinese(convs):
                        # Normalize to conversations key
                        out.write(json.dumps({"conversations": convs}, ensure_ascii=False) + "\n")
                        file_zh += 1

            total_zh += file_zh
            total_all += file_total
            pct = 100 * file_zh / file_total if file_total > 0 else 0
            print(f"  {fname}: {file_zh}/{file_total} Chinese ({pct:.1f}%)")

    print(f"\nTotal: {total_zh}/{total_all} Chinese entries ({100*total_zh/total_all:.1f}%)")
    print(f"Output: {output_path}")
    return total_zh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", default="/data/tengwan/datasets/nemotron-v2-jsonl/")
    parser.add_argument("--output", default="/data/tengwan/datasets/nemotron-v2-chinese/conversations.jsonl")
    args = parser.parse_args()

    print(f"Extracting Chinese conversations from {args.src_dir}")
    n = extract_chinese(args.src_dir, args.output)
    print(f"\nDone. {n} Chinese conversations extracted.")
