import os
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm
import json

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw_cases"
OUT_FILE = BASE_DIR / "data" / "processed" / "cases.json"

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

cases = []

for file in tqdm(os.listdir(RAW_DIR)):
    if file.endswith(".pdf"):
        text = extract_text(RAW_DIR / file)

        cases.append({
            "case_id": file.replace(".pdf",""),
            "text": text
        })

os.makedirs("data/processed", exist_ok=True)

with open(OUT_FILE, "w") as f:
    json.dump(cases, f)

print("Cases saved:", len(cases))
