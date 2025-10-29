import json

KB_PATH = "kb_data.jsonl"

docs = []
with open(KB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:  # skip blank lines
            continue
        docs.append(json.loads(line))

print(f"✅ Loaded {len(docs)} documents from {KB_PATH}")
