KB_PATH = "kb_data.jsonl"

with open(KB_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        print(f"Line {i}: {repr(line)}")
