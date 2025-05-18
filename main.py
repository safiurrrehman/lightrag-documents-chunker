"""
RAG System Chunking Pipeline

This is the main entry point for the RAG system chunking pipeline.
It orchestrates the document extraction, pattern identification,
chunking, and upserting processes.
"""

import os
import logging
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import time

# Import modules
from modules.document_extractor import DocumentExtractor
from modules.pattern_identifier import PatternIdentifier
from modules.chunker import Chunker
from modules.lightrag_upserter import LightRAGUpserter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RAG System Chunking Pipeline')
    
    parser.add_argument('--documents_directory', 
                        type=str, 
                        help='Path to the directory containing documents')
    
    parser.add_argument('--lightrag_url', 
                        type=str, 
                        help='URL for the LightRAG deployment')
    
    parser.add_argument('--max_chunk_size', 
                        type=int, 
                        default=1000,
                        help='Maximum size of a chunk in characters')
    
    parser.add_argument('--min_chunk_size', 
                        type=int, 
                        default=100,
                        help='Minimum size of a chunk in characters')
    
    parser.add_argument('--overlap_size', 
                        type=int, 
                        default=50,
                        help='Size of overlap between adjacent chunks in characters')
    
    parser.add_argument('--num_topics', 
                        type=int, 
                        default=5,
                        help='Number of topics to identify in the documents')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=10,
                        help='Number of chunks to upsert in a single batch')
    
    return parser.parse_args()

def main():
    """Main function to run the RAG system chunking pipeline."""
    start_time = time.time()
    
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get configuration from environment variables or command line arguments
    documents_directory = args.documents_directory or os.getenv('DOCUMENTS_DIRECTORY')
    lightrag_url = args.lightrag_url or os.getenv('LIGHTRAG_DEPLOYMENT_URL')
    
    if not documents_directory:
        raise ValueError("Documents directory not specified. Use --documents_directory or set DOCUMENTS_DIRECTORY environment variable.")
    
    if not lightrag_url:
        raise ValueError("LightRAG deployment URL not specified. Use --lightrag_url or set LIGHTRAG_DEPLOYMENT_URL environment variable.")
    
    logger.info("Starting RAG System Chunking Pipeline")
    logger.info(f"Documents Directory: {documents_directory}")
    logger.info(f"LightRAG Deployment URL: {lightrag_url}")
    
    try:
        # Step 1: Document Extraction
        logger.info("Step 1: Document Extraction")
        document_extractor = DocumentExtractor(documents_directory)
        documents = document_extractor.extract_all_documents()
        logger.info(f"Extracted {len(documents)} documents")
        
        # Step 2: Pattern Identification
        logger.info("Step 2: Pattern Identification")
        pattern_identifier = PatternIdentifier()
        patterns = pattern_identifier.identify_patterns(documents)
        logger.info("Identified patterns across documents")
        
        # Step 3: Chunking Process
        logger.info("Step 3: Chunking Process")
        chunker = Chunker(
            max_chunk_size=args.max_chunk_size,
            min_chunk_size=args.min_chunk_size,
            overlap_size=args.overlap_size
        )
        chunks = chunker.process_documents(documents, patterns)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4: Upserting to LightRAG
        logger.info("Step 4: Upserting to LightRAG")
        lightrag_upserter = LightRAGUpserter(lightrag_url, batch_size=args.batch_size)
        upsert_results = lightrag_upserter.upsert_chunks(chunks)
        logger.info(f"Upsert results: {upsert_results['successful_upserts']} successful, {upsert_results['failed_upserts']} failed")
        
        # Calculate and log execution time
        execution_time = time.time() - start_time
        logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        
        return {
            'status': 'success',
            'documents_processed': len(documents),
            'chunks_created': len(chunks),
            'chunks_upserted': upsert_results['successful_upserts'],
            'execution_time': execution_time
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    main()
