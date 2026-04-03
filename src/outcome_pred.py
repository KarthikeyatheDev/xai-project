import os
import json
from collections import defaultdict

from hybrid_retrieval import hybrid

# -------- CONFIG --------

STRUCTURED_DIR = "data/processed_cases/structured"

# ------------------------


def load_decision(case_id):

    path = os.path.join(STRUCTURED_DIR, case_id)

    if not os.path.exists(path):
        return None

    data = json.load(open(path, encoding="utf8"))

    return data.get("outcome_label", "unknown").lower()


# -------- MAIN --------


def predict_outcome(query_case):

    results = hybrid(query_case)

    decision_scores = defaultdict(float)

    retrieved_info = []

    for case, score in results:

        if case == query_case:
            continue

        decision = load_decision(case)

        if decision is None:
            continue

        retrieved_info.append((case, decision, score))

        decision_scores[decision] += score

    total = sum(decision_scores.values())

    if total == 0:
        return "unknown", {}, []

    probabilities = {k: v / total for k, v in decision_scores.items()}

    predicted = max(probabilities, key=probabilities.get)

    return predicted, probabilities, retrieved_info


# -------- TEST --------

# if __name__ == "__main__":
#     pred, probs, results = predict_outcome("Jallikattu-Judgement.json")

#     print("\n--- Final Prediction ---\n")
#     print("Predicted Outcome:", pred)

#     print("\nProbabilities:")
#     for k, v in probs.items():
#         print(f"{k} : {round(v,3)}")

#     print("\nRetrieved Cases:")
#     for c, d, s in results:
#         print(f"{c} → {d} ({round(s,3)})")
