import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Paths
KB_PATH = "kb_data.jsonl"
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load KB documents
docs = []
with open(KB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))

# Load FAISS index and embedding model
index = faiss.read_index(FAISS_INDEX_PATH)
model = SentenceTransformer(EMBEDDING_MODEL)

# Query loop
while True:
    query = input("\nAsk something (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), k=2)

    print("\n🔎 Top results:")
    for idx, dist in zip(I[0], D[0]):
        doc = docs[idx]
        print(f"\n📘 Title: {doc['title']}")
        print(f"📖 Source: {doc['source']}")
        print(f"💬 Text: {doc['text'][:400]}...")
        print(f"🧩 Distance: {dist:.4f}")

