import os
import json
import time
import re
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ------------------ CONFIG ------------------

INPUT_DIR = "../data/processed_cases/raw_text"
OUT_DIR = "data/processed_cases/structured"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_RETRIES = 3
TEXT_LIMIT = 12000

PROMPT = """
Extract structured legal information.

Return ONLY valid JSON.
Do NOT include markdown, backticks, or explanations.

Schema:
{
  "case_title": string,
  "key_facts": list of strings,
  "legal_issues": list of strings,
  "decision": string,
  "reasoning_summary": string,
  "outcome_label": one of [allowed, dismissed, partially_allowed, pending, unknown]
}
"""

# --------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

os.makedirs(OUT_DIR, exist_ok=True)


# -------- CLEAN JSON --------


def clean_json_output(output):

    output = output.strip()

    # remove ```json or ```
    output = re.sub(r"^```json", "", output)
    output = re.sub(r"^```", "", output)
    output = re.sub(r"```$", "", output)

    # extract JSON block
    match = re.search(r"\{.*\}", output, re.DOTALL)

    return match.group() if match else None


# -------- FIX STRUCTURE --------


def fix_lists(data):

    if isinstance(data.get("key_facts"), str):
        data["key_facts"] = [data["key_facts"]]

    if isinstance(data.get("legal_issues"), str):
        data["legal_issues"] = [data["legal_issues"]]

    # fallback if missing
    if not isinstance(data.get("key_facts"), list):
        data["key_facts"] = []

    if not isinstance(data.get("legal_issues"), list):
        data["legal_issues"] = []

    # normalize outcome
    allowed_labels = {
        "allowed",
        "dismissed",
        "partially_allowed",
        "pending",
        "unknown",
    }

    label = str(data.get("outcome_label", "unknown")).lower()

    if label not in ["allowed", "dismissed", "partially_allowed"]:
        label = "unknown"

    data["outcome_label"] = label

    return data


# -------- LLM CALL --------


def call_llm(text):

    for attempt in range(MAX_RETRIES):

        try:
            resp = client.chat_completion(
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=800,
                temperature=0,
            )

            return resp.choices[0].message["content"]

        except Exception as e:

            if attempt == MAX_RETRIES - 1:
                raise e

            time.sleep(2)


# -------- MAIN --------

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

for file in tqdm(files):

    in_path = os.path.join(INPUT_DIR, file)
    out_path = os.path.join(OUT_DIR, file)

    if os.path.exists(out_path):
        continue

    data = json.load(open(in_path, encoding="utf8"))

    text = data.get("text", "")[:TEXT_LIMIT]

    try:

        raw_output = call_llm(text)

        cleaned = clean_json_output(raw_output)

        if not cleaned:
            print(f"Failed cleaning: {file}")
            continue

        parsed = json.loads(cleaned)

        parsed = fix_lists(parsed)

        with open(out_path, "w", encoding="utf8") as f:
            json.dump(parsed, f, indent=2)

    except Exception as e:
        print(f"Failed: {file} -> {e}")

print("LLM parsing completed.")
