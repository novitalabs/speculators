#!/usr/bin/env python3
"""Merge multiple Novita conversation datasets with session-level dedup.

Turn dropout creates many training samples from one conversation, but they all
share identical prefixes (system prompt + issue description + early turns). When
the same task appears hundreds of times, the model overfits to those specific
token sequences rather than learning general draft prediction.

This script:
1. Loads multiple conversations.jsonl files
2. Fingerprints each conversation by its first user message (hash of first N chars)
3. Groups by fingerprint to identify same-session duplicates
4. Caps each group to --max-per-task samples (default 5), preferring longer conversations
5. Applies --min-turns filter
6. Shuffles and writes merged output with a diversity report

Usage:
    python k8s/merge_novita_datasets.py \
        --inputs /data/datasets/novita20260309/conversations_full_no_filter.jsonl \
                 /data/datasets/novita20260320/conversations.jsonl \
                 /data/datasets/novita20260327/conversations.jsonl \
        --output /data/datasets/novita_merged_exp18/conversations.jsonl \
        --max-per-task 5 \
        --min-turns 2 \
        --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict


def fingerprint(conv: dict, chars: int = 500) -> str:
    """Hash the first user message to identify the task/session."""
    msgs = conv.get("conversations", conv.get("messages", []))
    user_msg = ""
    for m in msgs:
        if m.get("role") == "user":
            user_msg = m["content"][:chars]
            break
    return hashlib.md5(user_msg.encode()).hexdigest()


def count_turns(conv: dict) -> int:
    msgs = conv.get("conversations", conv.get("messages", []))
    return len(msgs)


def count_assistant_turns(conv: dict) -> int:
    msgs = conv.get("conversations", conv.get("messages", []))
    return sum(1 for m in msgs if m.get("role") == "assistant")


def main():
    parser = argparse.ArgumentParser(description="Merge Novita datasets with session-level dedup")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-per-task", type=int, default=5,
                        help="Max conversations per unique task (default: 5)")
    parser.add_argument("--min-turns", type=int, default=2,
                        help="Min turns per conversation (default: 2)")
    parser.add_argument("--fingerprint-chars", type=int, default=500,
                        help="Characters of first user message to use for fingerprinting (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all conversations
    all_convs = []
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}, skipping")
            continue
        with open(path) as f:
            convs = [json.loads(line) for line in f]
        print(f"[load] {path}: {len(convs)} conversations")
        # Tag source for reporting
        source = os.path.basename(os.path.dirname(path))
        for c in convs:
            c["_source"] = source
        all_convs.extend(convs)

    print(f"\n[total] {len(all_convs)} conversations loaded from {len(args.inputs)} files")

    # Filter by min turns
    before = len(all_convs)
    all_convs = [c for c in all_convs if count_turns(c) >= args.min_turns
                 and count_assistant_turns(c) >= 1]
    print(f"[filter] min_turns={args.min_turns}: {before} → {len(all_convs)}")

    # Group by task fingerprint
    groups = defaultdict(list)
    for c in all_convs:
        fp = fingerprint(c, args.fingerprint_chars)
        groups[fp].append(c)

    print(f"[dedup] {len(all_convs)} conversations → {len(groups)} unique tasks")

    # Per-task sampling: keep at most max_per_task, prefer longer conversations
    sampled = []
    group_sizes_before = []
    group_sizes_after = []

    for fp, convs in groups.items():
        group_sizes_before.append(len(convs))
        # Sort by number of turns descending (prefer more complete conversations)
        convs.sort(key=lambda c: count_turns(c), reverse=True)
        selected = convs[:args.max_per_task]
        group_sizes_after.append(len(selected))
        sampled.extend(selected)

    print(f"[sample] max_per_task={args.max_per_task}: {len(all_convs)} → {len(sampled)}")

    # Shuffle
    random.shuffle(sampled)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for c in sampled:
            # Remove internal tag before writing
            c.pop("_source", None)
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n[output] {len(sampled)} conversations written to {args.output}")

    # Diversity report
    print(f"\n{'='*60}")
    print(f"  DIVERSITY REPORT")
    print(f"{'='*60}")

    # Per-source breakdown
    source_counter = Counter()
    for c in sampled:
        source_counter[c.get("_source", "unknown")] += 1

    # System prompt diversity
    sys_counter = Counter()
    for c in sampled:
        msgs = c.get("conversations", c.get("messages", []))
        sys_msg = next((m["content"][:100] for m in msgs if m.get("role") == "system"), "NONE")
        sys_counter[sys_msg] += 1

    # Re-check task uniqueness
    task_fps = set()
    for c in sampled:
        task_fps.add(fingerprint(c, args.fingerprint_chars))

    print(f"\nTotal conversations: {len(sampled)}")
    print(f"Unique tasks: {len(task_fps)}")
    print(f"Unique system prompts (100 chars): {len(sys_counter)}")
    print(f"Avg samples per task: {len(sampled)/len(task_fps):.1f}")

    print(f"\nGroup size distribution (before → after cap):")
    for label, sizes in [("before", group_sizes_before), ("after", group_sizes_after)]:
        dist = Counter()
        for s in sizes:
            if s == 1: dist["1x"] += 1
            elif s <= 3: dist["2-3x"] += 1
            elif s <= 5: dist["4-5x"] += 1
            elif s <= 10: dist["6-10x"] += 1
            else: dist[">10x"] += 1
        print(f"  {label}: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))

    print(f"\nTop 10 system prompts:")
    for sp, cnt in sys_counter.most_common(10):
        print(f"  [{cnt:>6d}] {sp[:80]}")


if __name__ == "__main__":
    main()
