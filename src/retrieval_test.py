import os
import json
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------- CONFIG ----------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_FILE = "../data/embeddings.json"

# ----------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_TOKEN
)

db = json.load(open(EMB_FILE))

def cosine(a,b):
    a,b=np.array(a),np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

query = input("Enter case facts: ")

q_emb = client.feature_extraction(query)

scores=[]

for item in db:
    s = cosine(q_emb,item["embedding"])
    scores.append((s,item["case_id"]))

scores.sort(reverse=True)

print("\nTop similar precedents:")
for s,c in scores[:5]:
    print(c, round(s,4))
