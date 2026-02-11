import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ---------- CONFIG ----------

INPUT_DIR = "../data/processed_cases/structured"
OUT_FILE = "../data/embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_TOKEN
)

embeddings_db = []

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

for file in tqdm(files):

    data = json.load(open(os.path.join(INPUT_DIR,file),encoding="utf8"))

    text = (
        str(data.get("key_facts","")) + " " +
        str(data.get("legal_issues","")) + " " +
        str(data.get("reasoning_summary",""))
    )

    emb = client.feature_extraction(text).tolist()

    embeddings_db.append({
        "case_id": file,
        "embedding": emb
    })

json.dump(embeddings_db, open(OUT_FILE,"w"))
print("Embeddings created (cloud).")