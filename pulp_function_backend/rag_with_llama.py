"""
rag_with_llama.py
Integrates FAISS-based retrieval (kb_data.jsonl + faiss_index.bin) with a local LLaMA GGUF model
using llama_cpp. The LLM is forced to use only retrieved KB passages and cite sources.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ------------------------
# Configuration (edit)
# ------------------------
KB_JSONL = "kb_data.jsonl"
FAISS_INDEX = "faiss_index.bin"
EMBED_MODEL = "all-MiniLM-L6-v2"            # embedding model for queries
LLAMA_GGUF_PATH = "models/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"  # path to your GGUF file
N_CTX = 4096
N_THREADS = 8
TOP_K = 3          # number of KB passages to retrieve

# ------------------------
# Sanity checks
# ------------------------
if not os.path.exists(KB_JSONL):
    raise FileNotFoundError(f"KB JSONL not found: {KB_JSONL}")
if not os.path.exists(FAISS_INDEX):
    raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX}")
if not os.path.exists(LLAMA_GGUF_PATH):
    raise FileNotFoundError(f"LLaMA GGUF file not found: {LLAMA_GGUF_PATH}")

# ------------------------
# Load KB documents
# ------------------------
docs = []
with open(KB_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))
print(f"✅ Loaded {len(docs)} KB docs")

# ------------------------
# Load FAISS index and embedding model
# ------------------------
index = faiss.read_index(FAISS_INDEX)
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ FAISS and embedder loaded")

# ------------------------
# Load local LLaMA via llama_cpp
# ------------------------
print("🦙 Loading LLaMA model (this may take a while)...")
llm = Llama.from_pretrained(
    model_path=LLAMA_GGUF_PATH,   # use model_path if you have local file
    n_ctx=N_CTX,
    n_threads=N_THREADS
)
print("✅ LLaMA loaded")

# ------------------------
# Helpers
# ------------------------
def retrieve(query_text, top_k=TOP_K):
    q_emb = embedder.encode([query_text], convert_to_numpy=True)
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)
    results = []
    for i_score, idx in zip(D[0], I[0]):
        if idx < 0: 
            continue
        doc = docs[idx]
        results.append({"idx": idx, "distance": float(i_score), "title": doc.get("title"), "source": doc.get("source"), "text": doc.get("text"), "url": doc.get("url", "")})
    return results

# Limit snippet length to avoid huge prompts
def shorten(text, max_chars=900):
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " ..."

# Build a safe RAG prompt instructing the model to only use retrieved docs
def build_prompt(query, retrieved):
    system = (
        "You are an agricultural assistant. ONLY use the retrieved documents below to answer the user's question. "
        "Cite sources inline as [source 1], [source 2], etc. If the information is not in the retrieved documents, say 'Not found in retrieved sources'. "
        "If recommending chemicals, include active ingredients and a clear safety note: 'Follow label/regulation and wear PPE'. Do NOT invent facts."
    )

    # Format retrieved docs numbered
    docs_text = ""
    for i, d in enumerate(retrieved, start=1):
        docs_text += f"[source {i}] Title: {d['title']} (Source: {d.get('source','unknown')})\n{shorten(d['text'])}\nURL: {d.get('url','')}\n\n"

    user = f"User question: {query}\n\nUse only the sources above to answer. Provide:\n1) Short confirmed label or summary\n2) Actionable steps (Immediate -> Treatment -> Prevention)\n3) Safety/regulatory notes\n4) List citations used (numbers)\n\nAnswer concisely and in plain language."
    # We will send system + user as a single message
    prompt = system + "\n\nRETRIEVED DOCUMENTS:\n\n" + docs_text + "\n\n" + user
    return prompt

# ------------------------
# Generate with LLaMA
# ------------------------
def generate_answer(prompt, max_tokens=512, temperature=0.0):
    # Use chat-like API for llama_cpp
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    # The returned structure might contain generated text under choices
    # For many GGUFs llama_cpp returns choices with message.content
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        # fallback to raw text if different shape
        return resp.get("text", str(resp))

# ------------------------
# Interactive loop
# ------------------------
if __name__ == "__main__":
    print("\n🔎 RAG + LLaMA ready. Type a question (or 'exit')\n")
    while True:
        query = input("Question> ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        # 1) Retrieve
        retrieved = retrieve(query, top_k=TOP_K)
        if not retrieved:
            print("⚠️ No retrieved documents found.")
            continue

        # 2) Build prompt
        prompt = build_prompt(query, retrieved)

        # 3) Generate answer from LLaMA
        print("\n🦙 Generating answer (may take a few seconds)...\n")
        answer = generate_answer(prompt, max_tokens=600, temperature=0.0)
        print("=== ANSWER ===\n")
        print(answer)
        print("\n=== SOURCES USED ===")
        for i, d in enumerate(retrieved, start=1):
            print(f"[{i}] {d['title']} — {d.get('source')} — {d.get('url','')}")
        print("\n----------------------------\n")
