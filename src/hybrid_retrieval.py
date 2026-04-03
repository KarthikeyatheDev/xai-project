import os
import json
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from graph_retreival import graph_retrieve  # ✅ use clean graph file

# ------------ CONFIG ----------------

STRUCTURED_DIR = "data/processed_cases/structured"
EMBED_FILE = "../data/embeddings.json"

TOP_K = 5
VECTOR_WEIGHT = 0.5
GRAPH_WEIGHT = 0.5

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)


# -------- VECTOR --------


def embed_query(text):
    emb = client.feature_extraction(text, model=EMBED_MODEL)
    return np.array(emb).flatten()


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_retrieve(query_case):

    path = os.path.join(STRUCTURED_DIR, query_case)

    if not os.path.exists(path):
        print("Missing file:", path)
        return {}

    data = json.load(open(path, encoding="utf8"))

    text = (
        " ".join(data.get("key_facts", []))
        + " "
        + " ".join(data.get("legal_issues", []))
    )

    query_vec = embed_query(text)

    embeddings_db = json.load(open(EMBED_FILE))

    scores = {}

    for item in embeddings_db:
        case_id = item["case_id"]
        vec = np.array(item["embedding"])

        sim = cosine(query_vec, vec)
        scores[case_id] = sim

    return scores


# -------- HYBRID --------


def hybrid(query_case):

    vector_scores = vector_retrieve(query_case)
    graph_results = graph_retrieve(query_case)

    graph_scores = dict(graph_results)

    final_scores = {}

    max_graph = max(graph_scores.values()) if graph_scores else 1

    for case in vector_scores:

        v = vector_scores.get(case, 0)
        g = graph_scores.get(case, 0) / max_graph

        final_scores[case] = VECTOR_WEIGHT * v + GRAPH_WEIGHT * g

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:TOP_K]


# -------- TEST --------

# if __name__ == "__main__":
#     results = hybrid("Jallikattu-Judgement.json")

#     print("\nHybrid Retrieved Cases:\n")

#     for case, score in results:
#         print(case, "score:", round(score, 3))
