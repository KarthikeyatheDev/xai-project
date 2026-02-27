import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from collections import defaultdict

# ------------ CONFIG ----------------

STRUCTURED_DIR = "../data/processed_cases/structured"

# Choose any REAL case as query
QUERY_CASE = "Jallikattu-Judgement.json"

TOP_K = 5

# ------------------------------------


load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD)
)


def graph_search(tx, facts, issues):

    results = []

    # Search by Facts

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


    # Search by Issues

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


def rank_cases(case_list):

    scores = defaultdict(int)

    for c in case_list:
        scores[c] += 1

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:TOP_K]


def main():

    query_path = os.path.join(
        STRUCTURED_DIR,
        QUERY_CASE
    )

    data = json.load(open(query_path, encoding="utf8"))

    facts = data["key_facts"]
    issues = data["legal_issues"]

    print("\nQuery Case:", QUERY_CASE)

    print("\nFacts:")
    for f in facts:
        print("-", f)

    print("\nIssues:")
    for i in issues:
        print("-", i)


    with driver.session() as session:

        retrieved = session.execute_read(
            graph_search,
            facts,
            issues
        )


    ranked = rank_cases(retrieved)


    print("\nTop Graph Retrieved Cases:\n")

    for case,score in ranked:

        if case != QUERY_CASE:
            print(case,"score:",score)


main()

driver.close()