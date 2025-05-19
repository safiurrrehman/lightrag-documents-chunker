"""
Visualization Interface for RAG System Chunking Pipeline

This module provides a web-based interface for visualizing chunks and patterns
identified by the RAG system chunking pipeline.
"""

import os
import json
import logging
import time
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
import sys

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from the main project
from modules.document_extractor import DocumentExtractor
from modules.pattern_identifier import PatternIdentifier
from modules.chunker import Chunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Flask to use our custom JSON encoder
app.json_encoder = NumpyEncoder

# Directory to store visualization data
VISUALIZATION_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(VISUALIZATION_DATA_DIR, exist_ok=True)

@app.route('/')
def index():
    """Render the main visualization dashboard."""
    return render_template('index.html')

@app.route('/chunk-visualization')
def chunk_visualization():
    """Render the chunk visualization page."""
    return render_template('chunk_visualization.html')

@app.route('/pattern-explorer')
def pattern_explorer():
    """Render the pattern explorer page."""
    return render_template('pattern_explorer.html')

@app.route('/api/process-documents', methods=['POST'])
def process_documents():
    """
    Process documents and generate visualization data.
    
    Expected JSON payload:
    {
        "documents_directory": "/path/to/documents",
        "max_chunk_size": 1000,
        "min_chunk_size": 100,
        "overlap_size": 50
    }
    """
    try:
        data = request.json
        documents_directory = data.get('documents_directory')
        max_chunk_size = int(data.get('max_chunk_size', 1000))
        min_chunk_size = int(data.get('min_chunk_size', 100))
        overlap_size = int(data.get('overlap_size', 50))
        
        if not documents_directory:
            return jsonify({"error": "Documents directory not specified"}), 400
        
        # Step 1: Document Extraction
        document_extractor = DocumentExtractor(documents_directory)
        documents = document_extractor.extract_all_documents()
        
        # Step 2: Pattern Identification
        pattern_identifier = PatternIdentifier()
        patterns = pattern_identifier.identify_patterns(documents)
        
        # Step 3: Chunking Process
        chunker = Chunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_size=overlap_size
        )
        chunks = chunker.process_documents(documents, patterns)
        
        # Generate visualization data
        visualization_data = generate_visualization_data(documents, patterns, chunks)
        
        # Save visualization data
        timestamp = int(time.time())
        visualization_data_path = os.path.join(VISUALIZATION_DATA_DIR, f"visualization_data_{timestamp}.json")
        with open(visualization_data_path, 'w') as f:
            json.dump(visualization_data, f, cls=NumpyEncoder, indent=2)
        
        return jsonify({
            "status": "success",
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "visualization_data_path": visualization_data_path
        })
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-visualization-data', methods=['GET'])
def get_visualization_data():
    """Get the latest visualization data."""
    try:
        # Get the latest visualization data file
        visualization_data_files = [f for f in os.listdir(VISUALIZATION_DATA_DIR) if f.startswith('visualization_data_')]
        if not visualization_data_files:
            # If no visualization data exists, return sample data
            return jsonify(get_sample_visualization_data())
        
        latest_file = max(visualization_data_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        visualization_data_path = os.path.join(VISUALIZATION_DATA_DIR, latest_file)
        
        with open(visualization_data_path, 'r') as f:
            visualization_data = json.load(f)
        
        return jsonify(visualization_data)
        
    except Exception as e:
        logger.error(f"Error getting visualization data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def generate_visualization_data(documents, patterns, chunks):
    """
    Generate visualization data from documents, patterns, and chunks.
    
    Args:
        documents: Dictionary mapping document IDs to their content
        patterns: Dictionary containing all identified patterns
        chunks: List of chunks created from the documents
        
    Returns:
        Dictionary containing visualization data
    """
    # Extract topic data
    topic_data = []
    if 'topic_patterns' in patterns and 'topics' in patterns['topic_patterns']:
        for topic_idx, keywords in enumerate(patterns['topic_patterns']['topics']):
            topic_data.append({
                "topic_id": topic_idx,
                "keywords": keywords[:10],  # Top 10 keywords
                "weight": patterns['topic_patterns']['topic_weights'][topic_idx] if 'topic_weights' in patterns['topic_patterns'] else 1.0
            })
    
    # Extract cluster data
    cluster_data = []
    if 'cluster_patterns' in patterns and 'clusters' in patterns['cluster_patterns']:
        clusters = patterns['cluster_patterns']['clusters']
        # Handle different cluster formats
        if isinstance(clusters, list):
            for cluster_idx, sentences in enumerate(clusters):
                # Check if sentences is a list or another type
                if isinstance(sentences, list):
                    sample_sentences = sentences[:3] if len(sentences) > 0 else []
                    sentence_count = len(sentences)
                else:
                    # Handle case where sentences might be a count or other value
                    sample_sentences = []
                    sentence_count = 1 if sentences else 0
                    
                cluster_data.append({
                    "cluster_id": cluster_idx,
                    "sentence_count": sentence_count,
                    "sample_sentences": sample_sentences
                })
        elif isinstance(clusters, dict):
            # Alternative format where clusters might be a dictionary
            for cluster_id, cluster_info in clusters.items():
                if isinstance(cluster_info, dict) and 'sentences' in cluster_info:
                    sentences = cluster_info['sentences']
                    sample_sentences = sentences[:3] if isinstance(sentences, list) and len(sentences) > 0 else []
                    sentence_count = len(sentences) if isinstance(sentences, list) else 1
                else:
                    sample_sentences = []
                    sentence_count = 1
                    
                cluster_data.append({
                    "cluster_id": cluster_id,
                    "sentence_count": sentence_count,
                    "sample_sentences": sample_sentences
                })
    
    # Extract entity data
    entity_data = []
    if 'entity_patterns' in patterns:
        for doc_id, entities in patterns['entity_patterns'].items():
            # Handle different entity pattern formats
            if isinstance(entities, dict):
                # Original format: entities is a dict with entity_type as keys
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        entity_data.append({
                            "document_id": doc_id,
                            "entity_type": entity_type,
                            "entity_text": entity['text'],
                            "frequency": entity.get('count', 1)
                        })
            elif isinstance(entities, list):
                # Alternative format: entities is a list of entity objects
                for entity in entities:
                    entity_data.append({
                        "document_id": doc_id,
                        "entity_type": entity.get('label', 'ENTITY'),
                        "entity_text": entity.get('text', ''),
                        "frequency": entity.get('count', 1)
                    })
    
    # Extract chunk data
    chunk_data = []
    for chunk_idx, chunk in enumerate(chunks):
        # Create a chunk data object with safe access to fields
        chunk_obj = {
            "chunk_id": chunk_idx,
            "document_id": chunk.get('document_id', f'doc{chunk_idx}'),
            "content_preview": chunk.get('content', '')[:100] + "..." if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
        }
        
        # Add position information if available
        if 'start_position' in chunk and 'end_position' in chunk:
            chunk_obj["start_position"] = chunk['start_position']
            chunk_obj["end_position"] = chunk['end_position']
        else:
            # Estimate positions if not available
            chunk_obj["start_position"] = 0
            chunk_obj["end_position"] = len(chunk.get('content', ''))
        
        # Add pattern information
        chunk_obj["topics"] = chunk.get('topics', [])
        chunk_obj["entities"] = chunk.get('entities', [])
        chunk_obj["clusters"] = chunk.get('clusters', [])
            
        chunk_data.append(chunk_obj)
    
    # Create document-chunk relationships
    document_chunk_relations = []
    for chunk_idx, chunk in enumerate(chunks):
        document_chunk_relations.append({
            "document_id": chunk.get('document_id', f'doc{chunk_idx}'),
            "chunk_id": chunk_idx
        })
    
    # Create chunk-pattern relationships
    chunk_pattern_relations = []
    for chunk_idx, chunk in enumerate(chunks):
        # Chunk-topic relations
        for topic_id in chunk.get('topics', []):
            chunk_pattern_relations.append({
                "chunk_id": chunk_idx,
                "pattern_type": "topic",
                "pattern_id": topic_id
            })
        
        # Chunk-cluster relations
        for cluster_id in chunk.get('clusters', []):
            chunk_pattern_relations.append({
                "chunk_id": chunk_idx,
                "pattern_type": "cluster",
                "pattern_id": cluster_id
            })
    
    return {
        "documents": [{"document_id": doc_id, "content_length": len(content)} for doc_id, content in documents.items()],
        "topics": topic_data,
        "clusters": cluster_data,
        "entities": entity_data,
        "chunks": chunk_data,
        "document_chunk_relations": document_chunk_relations,
        "chunk_pattern_relations": chunk_pattern_relations
    }

def get_sample_visualization_data():
    """Generate sample visualization data for demonstration."""
    return {
        "documents": [
            {"document_id": "doc1", "content_length": 5000},
            {"document_id": "doc2", "content_length": 3500},
            {"document_id": "doc3", "content_length": 4200}
        ],
        "topics": [
            {"topic_id": 0, "keywords": ["machine", "learning", "algorithm", "data", "model"], "weight": 0.35},
            {"topic_id": 1, "keywords": ["neural", "network", "deep", "training", "layer"], "weight": 0.25},
            {"topic_id": 2, "keywords": ["retrieval", "augmented", "generation", "rag", "chunks"], "weight": 0.40}
        ],
        "clusters": [
            {"cluster_id": 0, "sentence_count": 12, "sample_sentences": ["This is a sample sentence about machine learning.", "Data processing is an important step.", "Models need to be trained properly."]},
            {"cluster_id": 1, "sentence_count": 8, "sample_sentences": ["Neural networks have multiple layers.", "Deep learning requires significant computation.", "Training can take a long time."]},
            {"cluster_id": 2, "sentence_count": 15, "sample_sentences": ["RAG systems combine retrieval and generation.", "Chunks should be semantically meaningful.", "Document processing is the first step in RAG."]}
        ],
        "entities": [
            {"document_id": "doc1", "entity_type": "PERSON", "entity_text": "John Smith", "frequency": 3},
            {"document_id": "doc1", "entity_type": "ORG", "entity_text": "OpenAI", "frequency": 5},
            {"document_id": "doc2", "entity_type": "ORG", "entity_text": "Google", "frequency": 2},
            {"document_id": "doc3", "entity_type": "PRODUCT", "entity_text": "GPT-4", "frequency": 7}
        ],
        "chunks": [
            {"chunk_id": 0, "document_id": "doc1", "start_position": 0, "end_position": 500, "content_preview": "This is the beginning of document 1...", "topics": [0], "entities": ["OpenAI"], "clusters": [0]},
            {"chunk_id": 1, "document_id": "doc1", "start_position": 450, "end_position": 950, "content_preview": "Continuing with document 1...", "topics": [0, 1], "entities": ["John Smith"], "clusters": [0, 1]},
            {"chunk_id": 2, "document_id": "doc2", "start_position": 0, "end_position": 600, "content_preview": "This is the beginning of document 2...", "topics": [1], "entities": ["Google"], "clusters": [1]},
            {"chunk_id": 3, "document_id": "doc3", "start_position": 0, "end_position": 700, "content_preview": "This is the beginning of document 3...", "topics": [2], "entities": ["GPT-4"], "clusters": [2]}
        ],
        "document_chunk_relations": [
            {"document_id": "doc1", "chunk_id": 0},
            {"document_id": "doc1", "chunk_id": 1},
            {"document_id": "doc2", "chunk_id": 2},
            {"document_id": "doc3", "chunk_id": 3}
        ],
        "chunk_pattern_relations": [
            {"chunk_id": 0, "pattern_type": "topic", "pattern_id": 0},
            {"chunk_id": 0, "pattern_type": "cluster", "pattern_id": 0},
            {"chunk_id": 1, "pattern_type": "topic", "pattern_id": 0},
            {"chunk_id": 1, "pattern_type": "topic", "pattern_id": 1},
            {"chunk_id": 1, "pattern_type": "cluster", "pattern_id": 0},
            {"chunk_id": 1, "pattern_type": "cluster", "pattern_id": 1},
            {"chunk_id": 2, "pattern_type": "topic", "pattern_id": 1},
            {"chunk_id": 2, "pattern_type": "cluster", "pattern_id": 1},
            {"chunk_id": 3, "pattern_type": "topic", "pattern_id": 2},
            {"chunk_id": 3, "pattern_type": "cluster", "pattern_id": 2}
        ]
    }

@app.route('/api/generate-network-graph', methods=['GET'])
def generate_network_graph():
    """Generate a network graph visualization of chunks and patterns."""
    try:
        # Get visualization data
        visualization_data_files = [f for f in os.listdir(VISUALIZATION_DATA_DIR) if f.startswith('visualization_data_')]
        if not visualization_data_files:
            visualization_data = get_sample_visualization_data()
        else:
            latest_file = max(visualization_data_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            visualization_data_path = os.path.join(VISUALIZATION_DATA_DIR, latest_file)
            
            with open(visualization_data_path, 'r') as f:
                visualization_data = json.load(f)
        
        # Create network graph
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
        
        # Add document nodes
        for doc in visualization_data['documents']:
            net.add_node(f"doc_{doc['document_id']}", label=f"Document: {doc['document_id']}", 
                        title=f"Document: {doc['document_id']}\nLength: {doc['content_length']} chars",
                        color="#3498db", shape="dot", size=20)
        
        # Add chunk nodes
        for chunk in visualization_data['chunks']:
            net.add_node(f"chunk_{chunk['chunk_id']}", label=f"Chunk {chunk['chunk_id']}", 
                        title=f"Chunk {chunk['chunk_id']}\nDocument: {chunk['document_id']}\nPreview: {chunk['content_preview']}",
                        color="#2ecc71", shape="dot", size=15)
        
        # Add topic nodes
        for topic in visualization_data['topics']:
            keywords = ", ".join(topic['keywords'][:5])
            net.add_node(f"topic_{topic['topic_id']}", label=f"Topic {topic['topic_id']}", 
                        title=f"Topic {topic['topic_id']}\nKeywords: {keywords}\nWeight: {topic['weight']}",
                        color="#e74c3c", shape="triangle", size=25)
        
        # Add cluster nodes
        for cluster in visualization_data['clusters']:
            sample = cluster['sample_sentences'][0] if cluster['sample_sentences'] else "No sample"
            if len(sample) > 50:
                sample = sample[:50] + "..."
            net.add_node(f"cluster_{cluster['cluster_id']}", label=f"Cluster {cluster['cluster_id']}", 
                        title=f"Cluster {cluster['cluster_id']}\nSentences: {cluster['sentence_count']}\nSample: {sample}",
                        color="#f39c12", shape="diamond", size=20)
        
        # Add document-chunk edges
        for rel in visualization_data['document_chunk_relations']:
            net.add_edge(f"doc_{rel['document_id']}", f"chunk_{rel['chunk_id']}", color="#3498db", width=2)
        
        # Add chunk-pattern edges
        for rel in visualization_data['chunk_pattern_relations']:
            if rel['pattern_type'] == 'topic':
                net.add_edge(f"chunk_{rel['chunk_id']}", f"topic_{rel['pattern_id']}", color="#e74c3c", width=1)
            elif rel['pattern_type'] == 'cluster':
                net.add_edge(f"chunk_{rel['chunk_id']}", f"cluster_{rel['pattern_id']}", color="#f39c12", width=1)
        
        # Save and return the network graph
        graph_path = os.path.join(os.path.dirname(__file__), "static", "network_graph.html")
        net.save_graph(graph_path)
        
        return jsonify({"status": "success", "graph_path": "/static/network_graph.html"})
        
    except Exception as e:
        logger.error(f"Error generating network graph: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import time
    app.run(debug=True, port=5001)
