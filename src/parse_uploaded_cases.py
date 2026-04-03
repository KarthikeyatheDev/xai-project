import os
import json
import re
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=os.getenv("HF_TOKEN"))

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

MAX_RETRIES = 3


# -------- CLEAN JSON --------


def clean_json_output(output):

    output = output.strip()

    output = re.sub(r"^```json", "", output)
    output = re.sub(r"^```", "", output)
    output = re.sub(r"```$", "", output)

    match = re.search(r"\{.*\}", output, re.DOTALL)

    return match.group() if match else None


# -------- FIX STRUCTURE --------


def fix_lists(data):

    if isinstance(data.get("key_facts"), str):
        data["key_facts"] = [data["key_facts"]]

    if isinstance(data.get("legal_issues"), str):
        data["legal_issues"] = [data["legal_issues"]]

    if not isinstance(data.get("key_facts"), list):
        data["key_facts"] = []

    if not isinstance(data.get("legal_issues"), list):
        data["legal_issues"] = []

    allowed_labels = {
        "allowed",
        "dismissed",
        "partially_allowed",
        "pending",
        "unknown",
    }

    label = str(data.get("outcome_label", "unknown")).lower()

    if label not in allowed_labels:
        label = "unknown"

    data["outcome_label"] = label

    return data


# -------- MAIN FUNCTION --------


def parse_uploaded_case(text):

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text[:12000]},
                ],
                max_tokens=800,
                temperature=0,
            )

            output = response.choices[0].message["content"]

            cleaned = clean_json_output(output)

            if not cleaned:
                print("FAILED CLEANING\n", output)
                continue

            parsed = json.loads(cleaned)

            return fix_lists(parsed)

        except Exception as e:

            if attempt == MAX_RETRIES - 1:
                print("FINAL FAILURE:", e)
                return None

            time.sleep(2)
