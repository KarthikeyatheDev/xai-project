import os

print("\nSTEP 1: Extract text")
os.system("python ingest_cases.py")

print("\nSTEP 2: LLM parsing")
os.system("python llm_parse_cases.py")

print("\nSTEP 3: Build embeddings")
os.system("python build_embeddings.py")

print("\nSTEP 4: Insert graph")
os.system("python graph_insert.py")

print("\n✅ Knowledge Base Ready!")