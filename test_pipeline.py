"""
Test Pipeline Script

This script tests the individual components of the RAG system chunking pipeline
to ensure they are working correctly.
"""

import os
import logging
from dotenv import load_dotenv
import json
from pprint import pprint

# Import modules
from modules.document_extractor import DocumentExtractor
from modules.pattern_identifier import PatternIdentifier
from modules.chunker import Chunker
from modules.lightrag_upserter import LightRAGUpserter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_document_extractor():
    """Test the document extractor component."""
    logger.info("Testing Document Extractor...")
    
    # Get documents directory from environment
    documents_directory = os.getenv('DOCUMENTS_DIRECTORY')
    
    # Initialize document extractor
    document_extractor = DocumentExtractor(documents_directory)
    
    # List text files
    text_files = document_extractor.list_text_files()
    logger.info(f"Found {len(text_files)} text files")
    for file in text_files:
        logger.info(f"  - {os.path.basename(file)}")
    
    # Extract all documents
    documents = document_extractor.extract_all_documents()
    logger.info(f"Extracted {len(documents)} documents")
    for doc_id, content in documents.items():
        logger.info(f"  - {doc_id}: {len(content)} characters")
    
    return documents

def test_pattern_identifier(documents):
    """Test the pattern identifier component."""
    logger.info("Testing Pattern Identifier...")
    
    # Initialize pattern identifier
    pattern_identifier = PatternIdentifier()
    
    # Identify patterns
    patterns = pattern_identifier.identify_patterns(documents)
    
    # Log pattern information
    if 'topic_patterns' in patterns:
        topic_patterns = patterns['topic_patterns']
        if 'topics' in topic_patterns:
            logger.info("Identified topics:")
            for i, topic_keywords in enumerate(topic_patterns['topics']):
                logger.info(f"  - Topic {i}: {', '.join(topic_keywords[:5])}")
    
    if 'cluster_patterns' in patterns:
        cluster_patterns = patterns['cluster_patterns']
        if 'clusters' in cluster_patterns:
            logger.info(f"Identified {len(cluster_patterns['clusters'])} clusters")
    
    if 'entity_patterns' in patterns:
        entity_patterns = patterns['entity_patterns']
        total_entities = sum(len(entities) for entities in entity_patterns.values())
        logger.info(f"Extracted {total_entities} entities across all documents")
    
    return patterns

def test_chunker(documents, patterns):
    """Test the chunker component."""
    logger.info("Testing Chunker...")
    
    # Initialize chunker
    chunker = Chunker(
        max_chunk_size=int(os.getenv('MAX_CHUNK_SIZE', 1000)),
        min_chunk_size=int(os.getenv('MIN_CHUNK_SIZE', 100)),
        overlap_size=int(os.getenv('OVERLAP_SIZE', 50))
    )
    
    # Process documents
    chunks = chunker.process_documents(documents, patterns)
    
    # Log chunk information
    logger.info(f"Created {len(chunks)} chunks")
    
    # Log sample chunks
    if chunks:
        logger.info("Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            logger.info(f"  - Chunk {i} from {chunk['document_id']}: {len(chunk['text'])} characters")
            if 'dominant_topic_idx' in chunk:
                logger.info(f"    Dominant topic: {chunk['dominant_topic_idx']}")
            if 'entities' in chunk:
                entity_count = len(chunk.get('entities', []))
                logger.info(f"    Entities: {entity_count}")
    
    return chunks

def test_lightrag_upserter(chunks):
    """Test the LightRAG upserter component."""
    logger.info("Testing LightRAG Upserter...")
    
    # Get LightRAG URL from environment
    lightrag_url = os.getenv('LIGHTRAG_DEPLOYMENT_URL')
    
    # Initialize LightRAG upserter
    lightrag_upserter = LightRAGUpserter(lightrag_url, batch_size=int(os.getenv('BATCH_SIZE', 10)))
    
    # Check connection
    connection_status = lightrag_upserter.check_connection()
    logger.info(f"Connection status: {'Connected' if connection_status else 'Failed to connect'}")
    
    if connection_status:
        # Format a sample chunk
        if chunks:
            sample_chunk = chunks[0]
            formatted_chunk = lightrag_upserter.format_chunk_for_upsert(sample_chunk)
            logger.info("Sample formatted chunk:")
            pprint(formatted_chunk)
        
        # Upsert chunks
        upsert_results = lightrag_upserter.upsert_chunks(chunks)
        logger.info(f"Upsert results: {upsert_results['successful_upserts']} successful, {upsert_results['failed_upserts']} failed")
        
        return upsert_results
    else:
        logger.error("Skipping upsert test due to connection failure")
        return None

def main():
    """Main function to run the test pipeline."""
    logger.info("Starting RAG System Chunking Pipeline Test")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Test document extractor
        documents = test_document_extractor()
        
        # Test pattern identifier
        patterns = test_pattern_identifier(documents)
        
        # Test chunker
        chunks = test_chunker(documents, patterns)
        
        # Test LightRAG upserter
        upsert_results = test_lightrag_upserter(chunks)
        
        logger.info("Test pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Test pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
