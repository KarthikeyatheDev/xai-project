import os, json
import pdfplumber
from tqdm import tqdm

RAW_DIR = "../data/raw_cases"
OUT_DIR = "../data/processed_cases/raw_text"

os.makedirs(OUT_DIR, exist_ok=True)

def extract_text(pdf_path):
    text=""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

for file in tqdm(os.listdir(RAW_DIR)):
    if file.endswith(".pdf"):

        text = extract_text(os.path.join(RAW_DIR,file))

        out_path = os.path.join(OUT_DIR,file.replace(".pdf",".json"))

        json.dump({"text":text}, open(out_path,"w",encoding="utf8"), indent=2)

print("PDF ingestion completed.")