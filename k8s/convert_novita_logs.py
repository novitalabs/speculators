#!/usr/bin/env python3
"""Convert Novita API logs (MiniMax-M2.5) to ShareGPT JSONL for Eagle3 training.

Handles:
- Extracts messages from request_body field
- Converts list-type content (multimodal) to plain text
- Keeps tool messages as-is (they're part of the conversation context)
- Filters out conversations with no assistant response
"""

import json
import sys
from pathlib import Path


def flatten_content(content):
    """Convert list-type content to plain string."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    # skip image_url etc
                    parts.append(f"[{item.get('type', 'unknown')}]")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def convert_messages(messages):
    """Convert API messages to ShareGPT conversations format."""
    conversations = []
    for msg in messages:
        role = msg.get("role", "")
        content = flatten_content(msg.get("content", ""))

        # Map roles
        if role == "system":
            conversations.append({"role": "system", "content": content})
        elif role == "user":
            conversations.append({"role": "user", "content": content})
        elif role == "assistant":
            # assistant may have tool_calls but also content
            if not content and "tool_calls" in msg:
                # Serialize tool calls as content so the model sees them
                tc = msg["tool_calls"]
                content = json.dumps(tc, ensure_ascii=False)
            conversations.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Convert tool response to user message (model needs to see it)
            conversations.append({"role": "user", "content": content})
        else:
            # skip unknown roles
            continue
    return conversations


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_json> <output_jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    skipped_no_body = 0
    skipped_no_assistant = 0
    skipped_too_short = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            total += 1
            rec = json.loads(line)

            if "request_body" not in rec:
                skipped_no_body += 1
                continue

            body = rec["request_body"]
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    skipped_no_body += 1
                    continue

            messages = body.get("messages", [])
            if not messages:
                skipped_no_body += 1
                continue

            conversations = convert_messages(messages)

            # Must have at least one assistant turn
            has_assistant = any(c["role"] == "assistant" for c in conversations)
            if not has_assistant:
                skipped_no_assistant += 1
                continue

            # Must have at least 2 turns (user + assistant)
            if len(conversations) < 2:
                skipped_too_short += 1
                continue

            record = {"conversations": conversations}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Total records:        {total}")
    print(f"Converted:            {converted}")
    print(f"Skipped (no body):    {skipped_no_body}")
    print(f"Skipped (no asst):    {skipped_no_assistant}")
    print(f"Skipped (too short):  {skipped_too_short}")


if __name__ == "__main__":
    main()
