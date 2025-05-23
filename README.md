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
- Interactive visualization interface:
  - Chunk visualization with size distribution and relationship graphs
  - Pattern explorer for topics, clusters, and entities
  - Interactive network graph showing relationships between documents, chunks, and patterns

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables (see `.env.example`)
4. Run the pipeline: `python main.py`

## Usage

### Start the LightRAG Server
```bash
python mock_lightrag_server.py
```

### Run the Pipeline
```bash
python main.py --documents_directory /path/to/documents --lightrag_url http://localhost:8000
```

For more options, run:
```bash
python main.py --help
```

### Launch the Visualization Interface
```bash
python run_visualization.py
```

This will start the visualization server on http://localhost:5001 and automatically open it in your browser. The visualization interface allows you to:

- Process documents directly from the web interface
- Visualize chunk distribution and relationships
- Explore patterns identified across documents
- Generate interactive network graphs showing relationships between documents, chunks, and patterns
