import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from collections import defaultdict

# ------------ CONFIG ----------------

STRUCTURED_DIR = "../data/processed_cases/structured"
TOP_K = 5

# ------------------------------------

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# -------- GRAPH SEARCH --------


def graph_search(tx, facts, issues):

    results = []

    for fact in facts:
        res = tx.run(
            """
            MATCH (c:Case)-[:HAS_FACT]->(f:Fact)
            WHERE toLower(f.text) CONTAINS toLower($fact)
            RETURN c.id AS case_id
            """,
            fact=fact,
        )
        results += [r["case_id"] for r in res]

    for issue in issues:
        res = tx.run(
            """
            MATCH (c:Case)-[:HAS_ISSUE]->(i:Issue)
            WHERE toLower(i.text) CONTAINS toLower($issue)
            RETURN c.id AS case_id
            """,
            issue=issue,
        )
        results += [r["case_id"] for r in res]

    return results


# -------- MAIN FUNCTION --------


def graph_retrieve(query_case):

    path = os.path.join(STRUCTURED_DIR, query_case)

    if not os.path.exists(path):
        print("Missing file:", path)
        return []

    data = json.load(open(path, encoding="utf8"))

    facts = data.get("key_facts", [])
    issues = data.get("legal_issues", [])

    with driver.session() as session:
        res = session.execute_read(graph_search, facts, issues)

    scores = defaultdict(int)

    for c in res:
        scores[c] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:TOP_K]


# -------- TEST ONLY --------

# if __name__ == "__main__":
#     results = graph_retrieve("Jallikattu-Judgement.json")

#     print("\nTop Graph Retrieved Cases:\n")

#     for case, score in results:
#         print(case, "score:", score)
