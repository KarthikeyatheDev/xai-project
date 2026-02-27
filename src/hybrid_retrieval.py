import os
import json
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from collections import defaultdict

# ------------ CONFIG ----------------

STRUCTURED_DIR = "../data/processed_cases/structured"

QUERY_CASE = "Jallikattu-Judgement.json"

EMBED_FILE = "../data/embeddings.json"

TOP_K = 5

VECTOR_WEIGHT = 0.5
GRAPH_WEIGHT = 0.5

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    token=HF_TOKEN
)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD)
)


# ---------- VECTOR PART --------------

def embed_query(text):

    emb = client.feature_extraction(
        text,
        model=EMBED_MODEL
    )

    return np.array(emb).flatten()


def cosine(a,b):

    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))


def vector_retrieve(query_case):

    embeddings_db = json.load(open(EMBED_FILE))

    data = json.load(open(
        os.path.join(STRUCTURED_DIR,query_case),
        encoding="utf8"
    ))

    text = (
        " ".join(data["key_facts"])
        + " "
        + " ".join(data["legal_issues"])
    )

    query_vec = embed_query(text)

    scores = {}

    for item in embeddings_db:

        case_id = item["case_id"]

        vec = np.array(item["embedding"])

        sim = cosine(query_vec,vec)

        scores[case_id] = sim


    return scores


# ---------- GRAPH PART ---------------


def graph_search(tx,facts,issues):

    results=[]

    for fact in facts:

        res = tx.run(
        """
        MATCH (c:Case)-[:HAS_FACT]->(f:Fact)
        WHERE toLower(f.text) CONTAINS toLower($fact)
        RETURN c.id AS case_id
        """,
        fact=fact
        )

        results += [r["case_id"] for r in res]


    for issue in issues:

        res = tx.run(
        """
        MATCH (c:Case)-[:HAS_ISSUE]->(i:Issue)
        WHERE toLower(i.text) CONTAINS toLower($issue)
        RETURN c.id AS case_id
        """,
        issue=issue
        )

        results += [r["case_id"] for r in res]


    return results


def graph_retrieve(query_case):

    data=json.load(open(
        os.path.join(STRUCTURED_DIR,query_case),
        encoding="utf8"
    ))

    facts=data["key_facts"]
    issues=data["legal_issues"]

    with driver.session() as session:

        res=session.execute_read(
            graph_search,
            facts,
            issues
        )

    scores=defaultdict(int)

    for c in res:
        scores[c]+=1

    return scores


# ---------- HYBRID COMBINE -----------


def hybrid(query_case):

    vector_scores = vector_retrieve(query_case)

    graph_scores = graph_retrieve(query_case)

    final_scores = {}

    max_graph = max(graph_scores.values()) if graph_scores else 1

    for case in vector_scores:

        v = vector_scores.get(case,0)

        g = graph_scores.get(case,0) / max_graph

        score = (
            VECTOR_WEIGHT*v
            +
            GRAPH_WEIGHT*g
        )

        final_scores[case] = score


    ranked = sorted(
        final_scores.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return ranked[:TOP_K]


# -------- RUN -----------------------


results = hybrid(QUERY_CASE)

print("\nHybrid Retrieved Cases:\n")

for case,score in results:

    if case != QUERY_CASE:

        print(case,"score:",round(score,3))


driver.close()