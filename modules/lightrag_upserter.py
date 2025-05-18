"""
LightRAG Upserter Module

This module handles the connection to a LightRAG deployment and
upserts processed chunks to the deployment.
"""

import logging
import json
import requests
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightRAGUpserter:
    """
    A class to handle upserting processed chunks to a LightRAG deployment.
    
    This class provides methods to connect to a LightRAG deployment and
    upsert chunks with their pattern representations.
    """
    
    def __init__(self, deployment_url: str, batch_size: int = 10):
        """
        Initialize the LightRAGUpserter with the deployment URL.
        
        Args:
            deployment_url: URL of the LightRAG deployment
            batch_size: Number of chunks to upsert in a single batch
        """
        self.deployment_url = deployment_url
        self.batch_size = batch_size
        logger.info(f"Initialized LightRAGUpserter with deployment URL: {deployment_url}")
    
    def check_connection(self) -> bool:
        """
        Check if the connection to the LightRAG deployment is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Append health check endpoint if not already in URL
            health_url = self.deployment_url
            if not health_url.endswith('/'):
                health_url += '/'
            health_url += 'health'
            
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                logger.info("Successfully connected to LightRAG deployment")
                return True
            else:
                logger.error(f"Failed to connect to LightRAG deployment: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Error connecting to LightRAG deployment: {str(e)}")
            return False
    
    def format_chunk_for_upsert(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a chunk for upserting to LightRAG.
        
        Args:
            chunk: Chunk with pattern information
            
        Returns:
            Formatted chunk ready for upserting
        """
        # Extract basic chunk information
        document_id = chunk.get('document_id', '')
        text = chunk.get('text', '')
        
        # Create a unique ID for the chunk
        chunk_id = f"{document_id}_{chunk.get('start_idx', 0)}_{chunk.get('end_idx', 0)}"
        
        # Extract pattern information
        metadata = {
            'document_id': document_id,
            'source': 'document'
        }
        
        # Add topic information if available
        if 'dominant_topic_idx' in chunk:
            metadata['dominant_topic_idx'] = chunk['dominant_topic_idx']
            
        if 'topic_keywords' in chunk:
            metadata['topic_keywords'] = chunk['topic_keywords']
        
        # Add entity information if available
        if 'entities' in chunk:
            entity_texts = [entity['text'] for entity in chunk.get('entities', [])]
            entity_labels = [entity['label'] for entity in chunk.get('entities', [])]
            
            metadata['entities'] = entity_texts
            metadata['entity_labels'] = entity_labels
        
        # Create the formatted chunk
        formatted_chunk = {
            'id': chunk_id,
            'text': text,
            'metadata': metadata
        }
        
        # Add embedding if available
        if 'embedding' in chunk and chunk['embedding'] is not None:
            formatted_chunk['embedding'] = chunk['embedding'].tolist()
        
        return formatted_chunk
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upsert chunks to the LightRAG deployment.
        
        Args:
            chunks: List of chunks with pattern information
            
        Returns:
            Dictionary with upsert results
        """
        logger.info(f"Upserting {len(chunks)} chunks to LightRAG deployment")
        
        if not self.check_connection():
            raise ConnectionError("Failed to connect to LightRAG deployment")
        
        results = {
            'total_chunks': len(chunks),
            'successful_upserts': 0,
            'failed_upserts': 0,
            'errors': []
        }
        
        try:
            # Prepare upsert endpoint
            upsert_url = self.deployment_url
            if not upsert_url.endswith('/'):
                upsert_url += '/'
            upsert_url += 'upsert'
            
            # Process chunks in batches
            for i in tqdm(range(0, len(chunks), self.batch_size), desc="Upserting chunks"):
                batch = chunks[i:i+self.batch_size]
                
                # Format chunks for upserting
                formatted_chunks = [self.format_chunk_for_upsert(chunk) for chunk in batch]
                
                # Prepare payload with batch number
                payload = {
                    'chunks': formatted_chunks,
                    'batch_number': i//self.batch_size + 1
                }
                
                try:
                    # Send upsert request
                    response = requests.post(
                        upsert_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        results['successful_upserts'] += len(batch)
                        logger.info(f"Successfully upserted batch {i//self.batch_size + 1}")
                    else:
                        results['failed_upserts'] += len(batch)
                        error_msg = f"Failed to upsert batch {i//self.batch_size + 1}: {response.status_code} - {response.text}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                        
                except requests.RequestException as e:
                    results['failed_upserts'] += len(batch)
                    error_msg = f"Error upserting batch {i//self.batch_size + 1}: {str(e)}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Upsert complete: {results['successful_upserts']} successful, {results['failed_upserts']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error in upsert process: {str(e)}")
            raise
