"""Preprocess Novita API logs into training JSONL for speculators datagen.

Extracts OpenAI-format chat messages from request_body fields in Novita API logs.
Outputs JSONL with {"conversations": [{"role": ..., "content": ...}, ...]} format.

Usage:
    python preprocess_novita_logs.py \
        --input /data/datasets/novita20260309/.../export-*.json \
        --output /data/datasets/novita20260309/conversations.jsonl \
        --max-samples 10000
"""

import argparse
import json
import sys
from pathlib import Path


def extract_conversations(input_path: str, output_path: str, max_samples: int = 0,
                          min_turns: int = 2, min_assistant_turns: int = 1) -> dict:
    """Extract conversations from Novita API logs.

    Args:
        input_path: Path to the JSONL API log file.
        output_path: Path to write the output JSONL file.
        max_samples: Maximum number of conversations to extract (0 = unlimited).
        min_turns: Minimum number of turns (messages) per conversation.
        min_assistant_turns: Minimum number of assistant turns required.

    Returns:
        dict with extraction statistics.
    """
    stats = {
        "total_records": 0,
        "records_with_body": 0,
        "records_with_messages": 0,
        "skipped_too_few_turns": 0,
        "skipped_no_assistant": 0,
        "skipped_parse_error": 0,
        "conversations_written": 0,
        "total_turns": 0,
    }

    seen_traces = set()  # deduplicate by trace_id

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            stats["total_records"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_parse_error"] += 1
                continue

            request_body = record.get("request_body", "")
            if not request_body or len(str(request_body)) < 100:
                continue
            stats["records_with_body"] += 1

            # Deduplicate by trace_id
            trace_id = record.get("trace_id", "")
            if trace_id and trace_id in seen_traces:
                continue
            if trace_id:
                seen_traces.add(trace_id)

            # Parse request_body
            try:
                if isinstance(request_body, str):
                    body = json.loads(request_body)
                else:
                    body = request_body
            except json.JSONDecodeError:
                stats["skipped_parse_error"] += 1
                continue

            messages = body.get("messages", [])
            if not messages:
                continue
            stats["records_with_messages"] += 1

            # Filter: need minimum turns
            if len(messages) < min_turns:
                stats["skipped_too_few_turns"] += 1
                continue

            # Filter: need at least one assistant turn
            assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
            if assistant_count < min_assistant_turns:
                stats["skipped_no_assistant"] += 1
                continue

            # Normalize messages to conversations format
            conversations = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Handle content that might be a list (OpenAI multimodal format)
                if isinstance(content, list):
                    # Extract text parts only
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts)

                if not content or not isinstance(content, str):
                    continue

                # Only keep standard roles
                if role not in ("system", "user", "assistant"):
                    continue

                conversations.append({"role": role, "content": content})

            if len(conversations) < min_turns:
                stats["skipped_too_few_turns"] += 1
                continue

            # Write conversation
            fout.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")
            stats["conversations_written"] += 1
            stats["total_turns"] += len(conversations)

            if max_samples > 0 and stats["conversations_written"] >= max_samples:
                break

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Novita API logs for Eagle3 training")
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL API log file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max-samples", type=int, default=0, help="Max conversations (0=all)")
    parser.add_argument("--min-turns", type=int, default=2, help="Min turns per conversation")
    parser.add_argument("--min-assistant-turns", type=int, default=1, help="Min assistant turns")
    args = parser.parse_args()

    print(f"[PREPROCESS] Input: {args.input}")
    print(f"[PREPROCESS] Output: {args.output}")
    print(f"[PREPROCESS] Max samples: {args.max_samples or 'unlimited'}")

    stats = extract_conversations(
        args.input, args.output,
        max_samples=args.max_samples,
        min_turns=args.min_turns,
        min_assistant_turns=args.min_assistant_turns,
    )

    print(f"\n[PREPROCESS] Results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["conversations_written"] > 0:
        avg_turns = stats["total_turns"] / stats["conversations_written"]
        print(f"  avg_turns_per_conversation: {avg_turns:.1f}")
