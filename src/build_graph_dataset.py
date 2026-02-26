import os
import json
from tqdm import tqdm

INPUT_DIR = "../data/processed_cases/structured"

NODE_FILE = "../data/graph/nodes.json"
EDGE_FILE = "../data/graph/edges.json"

os.makedirs("../data/graph", exist_ok=True)

nodes = []
edges = []

cases = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

for file in tqdm(cases):

    data = json.load(open(os.path.join(INPUT_DIR,file),encoding="utf8"))

    case_id = file.replace(".json","")

    # -------- Case Node --------

    nodes.append({
        "id": case_id,
        "type": "Case"
    })

    # -------- Issue Nodes --------

    issues = data.get("legal_issues","")

    if isinstance(issues,str):

        issues = issues.split(",")

    for issue in issues:

        issue = issue.strip()

        if len(issue) < 3:
            continue

        issue_id = "ISSUE_" + issue.replace(" ","_")

        nodes.append({
            "id": issue_id,
            "type":"Issue",
            "name":issue
        })

        edges.append({
            "source":case_id,
            "target":issue_id,
            "type":"HAS_ISSUE"
        })


json.dump(nodes,open(NODE_FILE,"w"),indent=2)
json.dump(edges,open(EDGE_FILE,"w"),indent=2)

print("Graph dataset created.")