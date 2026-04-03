import os
import json
from collections import defaultdict

STRUCTURED_DIR = "../data/processed_cases/structured"


# -------- LOAD CASE --------


def load_case(case_id):
    path = os.path.join(STRUCTURED_DIR, case_id)
    return json.load(open(path, encoding="utf8"))


# -------- 1. Influential Cases --------


def influential_cases(retrieved_cases):
    """
    Just returns ranked cases (already sorted)
    """
    return retrieved_cases


# -------- 2. Influential Factors (FIXED LOGIC) --------


def influential_factors_dynamic(query_data, retrieved_cases):

    qf = query_data.get("key_facts", [])
    qi = query_data.get("legal_issues", [])

    scores = defaultdict(int)

    for case, _ in retrieved_cases:

        data = load_case(case)

        facts = data.get("key_facts", [])
        issues = data.get("legal_issues", [])

        # 🔥 BETTER MATCHING (substring instead of exact)
        for q in qf:
            for f in facts:
                if q.lower() in f.lower() or f.lower() in q.lower():
                    scores[q] += 1

        for q in qi:
            for i in issues:
                if q.lower() in i.lower() or i.lower() in q.lower():
                    scores[q] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:5]


# -------- 3. Counterfactual (CLEAN VERSION) --------


def counterfactual_analysis(query_data, predict_function):

    facts = query_data.get("key_facts", [])

    if not facts:
        return None

    removed_fact = facts[0]

    modified = query_data.copy()
    modified["key_facts"] = facts[1:]

    # Save temp
    temp_path = os.path.join(STRUCTURED_DIR, "temp_cf.json")

    with open(temp_path, "w", encoding="utf8") as f:
        json.dump(modified, f, indent=2)

    pred, _, _ = predict_function("temp_cf.json")

    os.remove(temp_path)

    return removed_fact, pred
