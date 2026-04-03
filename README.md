# ⚖️ Legal Decision Support System (XAI + Graph RAG)

An AI-powered system that analyzes legal case PDFs, retrieves similar precedents using **hybrid retrieval (vector + graph)**, predicts outcomes, and provides **explainable insights**.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Neo4j](https://img.shields.io/badge/Neo4j-4793D2?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)

## 🚀 Features

- 📄 **Upload legal case PDFs** for instant analysis
- 🤖 **LLM-based structured parsing** into JSON format
- 🔍 **Hybrid Retrieval System**:
  - Vector similarity search (embeddings)
  - Graph-based reasoning (Neo4j)
- ⚖️ **Outcome Prediction** (allowed/dismissed/etc.)
- 🧠 **Explainable AI (XAI)**:
  - Most influential precedent cases
  - Key legal factors driving decisions
  - Counterfactual "what-if" analysis

## 🏗️ Project Workflow

```
Raw PDFs → Text Extraction → LLM Parsing (JSON) 
    ↓
Embeddings (Vector DB) + Graph (Neo4j)
    ↓
Hybrid Retrieval → Outcome Prediction → XAI Insights
```

## 📁 Project Structure

```
Xai/
├── data/
│   ├── raw_cases/          # Input PDFs
│   ├── processed_cases/
│   │   ├── raw_text/       # Extracted text
│   │   └── structured/     # Parsed JSON
│   ├── embeddings.json
│   └── graph/
│       ├── nodes.json
│       └── edges.json
├── src/
│   ├── app.py              # Streamlit frontend
│   ├── ingest_cases.py     # PDF → text extraction
│   ├── llm_parse_cases.py  # LLM parsing to JSON
│   ├── build_embeddings.py # Generate embeddings
│   ├── graph_insert.py     # Neo4j population
│   ├── graph_retrieval.py  # Graph-based retrieval
│   ├── hybrid_retrieval.py # Combined vector+graph
│   ├── outcome_pred.py     # Outcome prediction
│   ├── xai.py              # Explainability module
│   └── parse_uploaded_cases.py
└── requirements.txt
```

## ⚙️ Quick Start

### Prerequisites
- Python 3.8+
- Neo4j (local or AuraDB)
- Hugging Face account (for Inference API)

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd Xai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# Hugging Face
HF_TOKEN=your_huggingface_token

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 3. Build Knowledge Base (One-time)

```bash
# Start Neo4j first and make sure the instance is running!
python src/setup.py
```

### 4. Launch Application

```bash
streamlit run src/app.py
```

## 🌐 Deployment

**Streamlit Cloud** (Recommended):
1. Push to GitHub
2. Connect repo in Streamlit Cloud
3. Set `src/app.py` as main file
4. Add secrets (HF_TOKEN, Neo4j creds)

## 📊 Sample Output

- **Predicted Outcome**: "Dismissed" (85% confidence)
- **Top 3 Similar Cases**: Case A, B, C with reasoning
- **Key Factors**: Jurisdiction, Precedent Strength, Evidence Quality
- **Counterfactual**: "If evidence was stronger → 65% chance of approval"

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | HuggingFace Inference API |
| Graph DB | Neo4j |
| Embeddings | Sentence Transformers |
| Processing | NumPy, Python |

## ⚠️ Important Notes

- Ensure Neo4j is running before `setup.py`
- LLM responses may vary slightly between runs
- Graph retrieval provides better causal reasoning than vector-only
- Start with small PDF batch for testing

## 🔮 Future Improvements

- [ ] ML/DL-based outcome prediction model
- [ ] Interactive UI (charts, case highlighting)
- [ ] Scalable vector DB (FAISS/Pinecone)
- [ ] FastAPI backend + REST API
- [ ] Multi-jurisdiction support

## 👨‍💻 Author

**Karthikeya Mohan**  
Vellore, Tamil Nadu, India