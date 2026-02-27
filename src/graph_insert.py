import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------- CONFIG ----------------

DATA_DIR = "../data/processed_cases/structured"

# ----------------------------------------

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    URI,
    auth=(USER, PASSWORD)
)


def insert_case(tx, case_id, title, facts, issues, decision, reasoning):

    # Create Case node
    tx.run(
        """
        MERGE (c:Case {id:$id})
        SET c.title=$title,
            c.decision=$decision,
            c.reasoning=$reasoning
        """,
        id=case_id,
        title=title,
        decision=decision,
        reasoning=reasoning
    )

    # Insert Facts
    for fact in facts:

        tx.run(
            """
            MERGE (f:Fact {text:$fact})
            WITH f
            MATCH (c:Case {id:$case_id})
            MERGE (c)-[:HAS_FACT]->(f)
            """,
            fact=fact,
            case_id=case_id
        )

    # Insert Legal Issues

    for issue in issues:

        tx.run(
            """
            MERGE (i:Issue {text:$issue})
            WITH i
            MATCH (c:Case {id:$case_id})
            MERGE (c)-[:HAS_ISSUE]->(i)
            """,
            issue=issue,
            case_id=case_id
        )


def process_file(file_path):

    data = json.load(open(file_path, encoding="utf8"))

    case_id = os.path.basename(file_path)

    title = data.get("case_title","")

    facts = data.get("key_facts",[])
    issues = data.get("legal_issues",[])

    decision = data.get("decision","")
    reasoning = data.get("reasoning_summary","")

    with driver.session() as session:

        session.execute_write(
            insert_case,
            case_id,
            title,
            facts,
            issues,
            decision,
            reasoning
        )


files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

print("Inserting cases into Neo4j...\n")

for file in tqdm(files):

    path = os.path.join(DATA_DIR,file)

    process_file(path)

print("\nGraph insertion complete.")

driver.close()