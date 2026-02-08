import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os

INPUT_FILE = "data/processed/cases.json"
VECTOR_PATH = "vectorstore/case_index.faiss"
META_PATH = "vectorstore/meta.json"

CHUNK_SIZE = 500

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, size=500):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

with open(INPUT_FILE) as f:
    cases = json.load(f)

texts = []
metadata = []

for case in tqdm(cases):
    for chunk in chunk_text(case["text"]):
        texts.append(chunk)
        metadata.append({"case_id": case["case_id"]})

embeddings = model.encode(texts, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(index, VECTOR_PATH)

with open(META_PATH, "w") as f:
    json.dump(metadata, f)

print("Vector DB built:", len(texts))
