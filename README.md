# XAI: Case Law Retrieval System

## Overview

This project implements a **semantic search and retrieval system for legal cases** using embeddings and vector similarity search. It enables intelligent case lookup by finding semantically similar cases to a given query, demonstrating **Explainable AI (XAI)** principles through case-based reasoning.

## Features

- **PDF Case Ingestion**: Automatically extracts and processes text from PDF case documents
- **Text Embedding**: Converts case documents into semantic embeddings using sentence transformers
- **Vector Database**: Utilizes FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Semantic Retrieval**: Finds the most relevant cases based on semantic similarity to queries
- **Scalable Architecture**: Chunked text processing for handling large documents

## Project Structure

```
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw_cases/                    # Input: PDF case files
│   └── processed/
│       └── cases.json                # Extracted case data in JSON format
├── src/
│   ├── ingest_cases.py              # Extract text from PDF files
│   ├── build_embeddings.py          # Generate embeddings and build vector index
│   └── retrieval_test.py            # Test semantic search functionality
└── vectorstore/
    ├── case_index.faiss             # FAISS vector index
    └── meta.json                    # Metadata mapping for retrieved cases
```

## Workflow

### 1. Data Ingestion (`ingest_cases.py`)
- Reads PDF files from `data/raw_cases/`
- Extracts text content from each PDF
- Stores extracted text and case IDs in `data/processed/cases.json`
- Output: JSON file containing case_id and full text for each case

### 2. Embedding Generation (`build_embeddings.py`)
- Loads processed case data from JSON
- Chunks case text into 500-word segments for efficient processing
- Generates semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` model
- Builds FAISS index with L2 distance metric for similarity search
- Saves vector index and metadata to `vectorstore/`
- Output: FAISS index file and metadata JSON

### 3. Retrieval Testing (`retrieval_test.py`)
- Loads the FAISS vector index and metadata
- Takes a query string (e.g., "land acquisition compensation dispute")
- Encodes the query into embeddings
- Retrieves top-k most similar case chunks using FAISS
- Displays metadata of matching cases

## Dependencies

- **faiss-cpu**: FAISS library for similarity search
- **sentence-transformers**: Pre-trained models for text embeddings
- **pypdf**: PDF text extraction
- **numpy/pandas**: Data processing
- **tqdm**: Progress bars

See `requirements.txt` for complete dependency list.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Usage

1. **Place PDF case files** in `data/raw_cases/`

2. **Ingest and process cases**:
   ```bash
   python src/ingest_cases.py
   ```

3. **Build the vector index**:
   ```bash
   python src/build_embeddings.py
   ```

4. **Test retrieval**:
   ```bash
   python src/retrieval_test.py
   ```

## Model Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Lightweight and efficient
  - 384-dimensional embeddings
  - Optimized for semantic similarity tasks
  
- **Vector Index**: FAISS IndexFlatL2
  - Uses L2 Euclidean distance metric
  - Suitable for exact similarity search on moderate data sizes

## Use Cases

- **Legal Research**: Find precedent cases relevant to current disputes
- **Case Analysis**: Identify similar cases for comparative analysis
- **Knowledge Retrieval**: Semantic search through large case law databases
- **Legal Decision Support**: Provide relevant case references for explainable recommendations

## Notes

- Text is chunked into 500-word segments to balance context preservation with computational efficiency
- The FAISS index uses exact search (IndexFlatL2); consider approximate methods (HNSW, IVF) for larger datasets
- Metadata is maintained separately to associate retrieved chunks back to original cases
