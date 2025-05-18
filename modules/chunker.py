"""
Chunker Module

This module implements algorithms to chunk documents based on identified patterns.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Chunker:
    """
    A class to chunk documents based on identified patterns.
    
    This class provides methods to segment documents into meaningful chunks
    that maintain context and represent coherent units of information.
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000, 
                 min_chunk_size: int = 100,
                 overlap_size: int = 50):
        """
        Initialize the Chunker with configuration parameters.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            min_chunk_size: Minimum size of a chunk in characters
            overlap_size: Size of overlap between adjacent chunks in characters
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        logger.info(f"Initialized Chunker with max_size={max_chunk_size}, min_size={min_chunk_size}, overlap={overlap_size}")
    
    def chunk_by_pattern_boundaries(self, 
                                   documents: Dict[str, str], 
                                   patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk documents based on identified pattern boundaries.
        
        Args:
            documents: Dictionary mapping document IDs to their content
            patterns: Dictionary containing identified patterns
            
        Returns:
            List of chunks with metadata
        """
        logger.info(f"Chunking {len(documents)} documents based on pattern boundaries")
        
        chunks = []
        cluster_patterns = patterns.get('cluster_patterns', {})
        preprocessed_docs = patterns.get('preprocessed_docs', {})
        
        try:
            # Use sentence clusters as chunk boundaries
            if cluster_patterns and 'labels' in cluster_patterns:
                labels = cluster_patterns['labels']
                all_sentences = cluster_patterns['all_sentences']
                sentence_sources = cluster_patterns['sentence_sources']
                embeddings = cluster_patterns['embeddings']
                
                # Process each document
                for doc_id, content in documents.items():
                    logger.info(f"Chunking document: {doc_id}")
                    
                    # Get indices of sentences from this document
                    doc_indices = [i for i, source in enumerate(sentence_sources) if source == doc_id]
                    
                    if not doc_indices:
                        logger.warning(f"No sentences found for document {doc_id}")
                        continue
                    
                    # Group sentences by cluster
                    current_chunk_text = ""
                    current_chunk_sentences = []
                    current_chunk_embeddings = []
                    chunk_start_idx = 0
                    
                    for i in range(len(doc_indices)):
                        idx = doc_indices[i]
                        sentence = all_sentences[idx]
                        label = labels[idx]
                        embedding = embeddings[idx]
                        
                        # Check if we should start a new chunk
                        new_chunk_needed = False
                        
                        # Start new chunk if current one exceeds max size
                        if len(current_chunk_text) + len(sentence) > self.max_chunk_size:
                            new_chunk_needed = True
                        
                        # Start new chunk if cluster changes (and not noise)
                        if i > 0 and label != -1 and label != labels[doc_indices[i-1]]:
                            # Only create new chunk if current one meets minimum size
                            if len(current_chunk_text) >= self.min_chunk_size:
                                new_chunk_needed = True
                        
                        if new_chunk_needed:
                            # Create chunk from accumulated sentences
                            if current_chunk_sentences:
                                chunk_embedding = np.mean(current_chunk_embeddings, axis=0) if current_chunk_embeddings else None
                                
                                chunk = {
                                    'document_id': doc_id,
                                    'text': current_chunk_text,
                                    'sentences': current_chunk_sentences,
                                    'embedding': chunk_embedding,
                                    'start_idx': chunk_start_idx,
                                    'end_idx': i - 1
                                }
                                
                                chunks.append(chunk)
                                
                                # Reset for next chunk, with overlap
                                overlap_start = max(0, len(current_chunk_sentences) - self.overlap_size)
                                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                                current_chunk_embeddings = current_chunk_embeddings[overlap_start:]
                                current_chunk_text = " ".join(current_chunk_sentences)
                                chunk_start_idx = i - len(current_chunk_sentences)
                        
                        # Add current sentence to chunk
                        current_chunk_sentences.append(sentence)
                        current_chunk_embeddings.append(embedding)
                        if current_chunk_text:
                            current_chunk_text += " " + sentence
                        else:
                            current_chunk_text = sentence
                    
                    # Add the last chunk if not empty
                    if current_chunk_sentences and len(current_chunk_text) >= self.min_chunk_size:
                        chunk_embedding = np.mean(current_chunk_embeddings, axis=0) if current_chunk_embeddings else None
                        
                        chunk = {
                            'document_id': doc_id,
                            'text': current_chunk_text,
                            'sentences': current_chunk_sentences,
                            'embedding': chunk_embedding,
                            'start_idx': chunk_start_idx,
                            'end_idx': len(doc_indices) - 1
                        }
                        
                        chunks.append(chunk)
            
            # Fallback to simple chunking if pattern-based chunking failed
            if not chunks:
                logger.warning("Pattern-based chunking produced no chunks, falling back to simple chunking")
                chunks = self.chunk_by_size(documents)
            
            logger.info(f"Created {len(chunks)} chunks based on pattern boundaries")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in pattern-based chunking: {str(e)}")
            logger.warning("Falling back to simple chunking")
            return self.chunk_by_size(documents)
    
    def chunk_by_size(self, documents: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Chunk documents based on size, with overlap between chunks.
        
        Args:
            documents: Dictionary mapping document IDs to their content
            
        Returns:
            List of chunks with metadata
        """
        logger.info(f"Chunking {len(documents)} documents based on size")
        
        chunks = []
        
        try:
            for doc_id, content in documents.items():
                logger.info(f"Chunking document by size: {doc_id}")
                
                # Split content into sentences (simple approach)
                sentences = content.split('. ')
                sentences = [s + '.' if not s.endswith('.') else s for s in sentences]
                
                current_chunk = ""
                current_sentences = []
                
                for sentence in sentences:
                    # Check if adding this sentence would exceed max chunk size
                    if len(current_chunk) + len(sentence) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                        # Create chunk from accumulated text
                        chunk = {
                            'document_id': doc_id,
                            'text': current_chunk,
                            'sentences': current_sentences,
                            'embedding': None  # No embedding for simple chunking
                        }
                        
                        chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_start = max(0, len(current_sentences) - self.overlap_size)
                        current_sentences = current_sentences[overlap_start:]
                        current_chunk = " ".join(current_sentences)
                    
                    # Add current sentence to chunk
                    current_sentences.append(sentence)
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                
                # Add the last chunk if not empty
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunk = {
                        'document_id': doc_id,
                        'text': current_chunk,
                        'sentences': current_sentences,
                        'embedding': None
                    }
                    
                    chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks based on size")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in size-based chunking: {str(e)}")
            raise
    
    def enrich_chunks_with_patterns(self, 
                                   chunks: List[Dict[str, Any]], 
                                   patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enrich chunks with pattern information.
        
        Args:
            chunks: List of document chunks
            patterns: Dictionary containing identified patterns
            
        Returns:
            List of chunks enriched with pattern information
        """
        logger.info(f"Enriching {len(chunks)} chunks with pattern information")
        
        enriched_chunks = []
        
        try:
            # Extract pattern information
            topic_patterns = patterns.get('topic_patterns', {})
            entity_patterns = patterns.get('entity_patterns', {})
            
            # Process each chunk
            for chunk in chunks:
                doc_id = chunk['document_id']
                
                # Add topic information if available
                if topic_patterns and 'doc_topic_dist' in topic_patterns and 'doc_ids' in topic_patterns:
                    doc_idx = topic_patterns['doc_ids'].index(doc_id) if doc_id in topic_patterns['doc_ids'] else -1
                    
                    if doc_idx >= 0:
                        # Get topic distribution for this document
                        topic_dist = topic_patterns['doc_topic_dist'][doc_idx]
                        
                        # Get dominant topic
                        dominant_topic_idx = np.argmax(topic_dist)
                        
                        # Add topic information to chunk
                        chunk['dominant_topic_idx'] = int(dominant_topic_idx)
                        chunk['topic_distribution'] = topic_dist.tolist()
                        
                        if 'topics' in topic_patterns:
                            chunk['topic_keywords'] = topic_patterns['topics'][dominant_topic_idx]
                
                # Add entity information if available
                if entity_patterns and doc_id in entity_patterns:
                    doc_entities = entity_patterns[doc_id]
                    
                    # Filter entities that appear in this chunk
                    chunk_text = chunk['text']
                    chunk_entities = [
                        entity for entity in doc_entities
                        if entity['text'] in chunk_text
                    ]
                    
                    chunk['entities'] = chunk_entities
                
                enriched_chunks.append(chunk)
            
            logger.info(f"Successfully enriched {len(enriched_chunks)} chunks")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Error enriching chunks with patterns: {str(e)}")
            # Return original chunks if enrichment fails
            return chunks
    
    def process_documents(self, 
                         documents: Dict[str, str], 
                         patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process documents to create pattern-aware chunks.
        
        Args:
            documents: Dictionary mapping document IDs to their content
            patterns: Dictionary containing identified patterns
            
        Returns:
            List of chunks with pattern information
        """
        logger.info(f"Processing {len(documents)} documents for chunking")
        
        try:
            # Create chunks based on pattern boundaries
            chunks = self.chunk_by_pattern_boundaries(documents, patterns)
            
            # Enrich chunks with pattern information
            enriched_chunks = self.enrich_chunks_with_patterns(chunks, patterns)
            
            logger.info(f"Successfully processed documents into {len(enriched_chunks)} chunks")
            return enriched_chunks
            
        except Exception as e:
            logger.error(f"Error processing documents for chunking: {str(e)}")
            raise
