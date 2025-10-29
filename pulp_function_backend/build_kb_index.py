import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
KB_PATH = "kb_data.jsonl"
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # lightweight, works well

# Load knowledge base
docs = []
with open(KB_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))

print(f"✅ Loaded {len(docs)} documents from {KB_PATH}")

# Prepare texts
texts = [d["text"] for d in docs]
titles = [d["title"] for d in docs]

# Generate embeddings
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, show_progress_bar=True)
print(f"🔍 Generated embeddings: {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

# Save index
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")
