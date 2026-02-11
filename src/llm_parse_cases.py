import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ------------------ CONFIG ------------------

INPUT_DIR = "../data/processed_cases/raw_text"
OUT_DIR = "../data/processed_cases/structured"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_RETRIES = 3
TEXT_LIMIT = 12000

PROMPT = """
Extract structured legal information.

Return STRICT JSON with:
case_title
key_facts
legal_issues
decision
reasoning_summary
"""

# --------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_TOKEN
)

os.makedirs(OUT_DIR, exist_ok=True)


def call_llm(text):
    """Call HuggingFace LLM with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=800,
                temperature=0
            )
            return resp.choices[0].message["content"]

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise e
            time.sleep(2)


files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

for file in tqdm(files):

    in_path = os.path.join(INPUT_DIR, file)
    out_path = os.path.join(OUT_DIR, file)

    # skip already processed
    if os.path.exists(out_path):
        continue

    data = json.load(open(in_path, encoding="utf8"))

    text = data.get("text", "")[:TEXT_LIMIT]

    try:
        parsed = call_llm(text)

        with open(out_path, "w", encoding="utf8") as f:
            f.write(parsed)

    except Exception as e:
        print(f"Failed: {file} -> {e}")

print("LLM parsing completed.")
