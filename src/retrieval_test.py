import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_PATH = "vectorstore/case_index.faiss"
META_PATH = "vectorstore/meta.json"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

query = "land acquisition compensation dispute"

q_emb = model.encode([query])
D, I = index.search(np.array(q_emb), 5)

print("Top similar cases:")
for i in I[0]:
    print(meta[i])
