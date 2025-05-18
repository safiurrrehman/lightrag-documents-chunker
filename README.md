# LightRAG Documents Chunker

A pattern-aware RAG system that intelligently processes documents using NLP techniques. This pipeline extracts documents, identifies semantic patterns through topic modeling and clustering, creates optimized chunks, and uploads them to a LightRAG deployment for enhanced retrieval performance.

## Features

- Document extraction from various file formats
- Pattern identification using NLP techniques:
  - Topic modeling with LDA
  - Sentence clustering based on semantic similarity
  - Key entity extraction
- Intelligent chunking with configurable parameters
- Batch upserting to LightRAG deployment

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables (see `.env.example`)
4. Run the pipeline: `python main.py`

## Usage

```bash
python main.py --documents_directory /path/to/documents --lightrag_url https://your-lightrag-deployment-url
```

For more options, run:
```bash
python main.py --help
```
