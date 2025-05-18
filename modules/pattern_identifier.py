"""
Pattern Identifier Module

This module implements methods to identify common patterns across documents
using various NLP techniques.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import spacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatternIdentifier:
    """
    A class to identify patterns across multiple documents.
    
    This class provides methods to extract key phrases, identify topics,
    and cluster similar content across documents.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PatternIdentifier with necessary models.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        logger.info("Initializing PatternIdentifier")
        
        # Download necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        
        # Initialize sentence transformer model
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {str(e)}")
            raise
        
        # Initialize spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model: en_core_web_sm")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def preprocess_documents(self, documents: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Preprocess documents by splitting them into sentences.
        
        Args:
            documents: Dictionary mapping document IDs to their content
            
        Returns:
            Dictionary mapping document IDs to lists of sentences
        """
        preprocessed_docs = {}
        
        for doc_id, content in documents.items():
            try:
                # Split document into sentences
                sentences = sent_tokenize(content)
                preprocessed_docs[doc_id] = sentences
                logger.info(f"Preprocessed document {doc_id}: {len(sentences)} sentences")
            except Exception as e:
                logger.error(f"Error preprocessing document {doc_id}: {str(e)}")
                preprocessed_docs[doc_id] = [content]  # Fallback to using the whole document
        
        return preprocessed_docs
    
    def identify_topics(self, documents: Dict[str, str], num_topics: int = 5) -> Dict[str, Any]:
        """
        Identify topics across documents using Latent Dirichlet Allocation (LDA).
        
        Args:
            documents: Dictionary mapping document IDs to their content
            num_topics: Number of topics to identify
            
        Returns:
            Dictionary containing topic model and related information
        """
        logger.info(f"Identifying {num_topics} topics across {len(documents)} documents")
        
        try:
            # Vectorize documents using TF-IDF
            vectorizer = TfidfVectorizer(
                max_df=0.95, 
                min_df=2,
                stop_words='english'
            )
            
            doc_contents = list(documents.values())
            doc_ids = list(documents.keys())
            
            tfidf_matrix = vectorizer.fit_transform(doc_contents)
            
            # Apply LDA for topic modeling
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                learning_method='online'
            )
            
            lda.fit(tfidf_matrix)
            
            # Get feature names for interpretation
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top words for each topic
            topic_keywords = []
            for topic_idx, topic in enumerate(lda.components_):
                top_keywords_idx = topic.argsort()[:-10-1:-1]
                top_keywords = [feature_names[i] for i in top_keywords_idx]
                topic_keywords.append(top_keywords)
            
            # Get document-topic distributions
            doc_topic_dist = lda.transform(tfidf_matrix)
            
            # Create topic representation
            topic_representation = {
                'model': lda,
                'vectorizer': vectorizer,
                'topics': topic_keywords,
                'doc_topic_dist': doc_topic_dist,
                'doc_ids': doc_ids
            }
            
            logger.info("Successfully identified topics")
            return topic_representation
        
        except Exception as e:
            logger.error(f"Error in topic identification: {str(e)}")
            raise
    
    def extract_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Extract sentence embeddings using the sentence transformer model.
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            NumPy array of sentence embeddings
        """
        try:
            embeddings = self.sentence_model.encode(sentences, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise
    
    def cluster_sentences(self, 
                          preprocessed_docs: Dict[str, List[str]], 
                          eps: float = 0.3, 
                          min_samples: int = 3) -> Dict[str, Any]:
        """
        Cluster sentences across documents based on semantic similarity.
        
        Args:
            preprocessed_docs: Dictionary mapping document IDs to lists of sentences
            eps: DBSCAN parameter for maximum distance between points
            min_samples: DBSCAN parameter for minimum points in a cluster
            
        Returns:
            Dictionary containing clustering results
        """
        logger.info(f"Clustering sentences from {len(preprocessed_docs)} documents")
        
        try:
            # Flatten sentences and keep track of document sources
            all_sentences = []
            sentence_sources = []
            
            for doc_id, sentences in preprocessed_docs.items():
                all_sentences.extend(sentences)
                sentence_sources.extend([doc_id] * len(sentences))
            
            # Extract embeddings
            logger.info(f"Extracting embeddings for {len(all_sentences)} sentences")
            embeddings = self.extract_embeddings(all_sentences)
            
            # Cluster embeddings using DBSCAN
            logger.info("Clustering embeddings with DBSCAN")
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
            
            # Organize results
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Noise points
                    continue
                    
                if label not in clusters:
                    clusters[label] = []
                    
                clusters[label].append({
                    'sentence': all_sentences[i],
                    'document': sentence_sources[i],
                    'embedding': embeddings[i]
                })
            
            logger.info(f"Identified {n_clusters} clusters")
            
            # Calculate cluster centroids
            centroids = {}
            for label, items in clusters.items():
                cluster_embeddings = np.array([item['embedding'] for item in items])
                centroids[label] = np.mean(cluster_embeddings, axis=0)
            
            clustering_result = {
                'clusters': clusters,
                'centroids': centroids,
                'n_clusters': n_clusters,
                'all_sentences': all_sentences,
                'sentence_sources': sentence_sources,
                'embeddings': embeddings,
                'labels': labels
            }
            
            return clustering_result
        
        except Exception as e:
            logger.error(f"Error in sentence clustering: {str(e)}")
            raise
    
    def extract_key_entities(self, documents: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract key entities from documents using spaCy.
        
        Args:
            documents: Dictionary mapping document IDs to their content
            
        Returns:
            Dictionary mapping document IDs to lists of extracted entities
        """
        logger.info(f"Extracting key entities from {len(documents)} documents")
        
        entities_by_doc = {}
        
        for doc_id, content in documents.items():
            try:
                doc = self.nlp(content)
                
                # Extract named entities
                entities = []
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                
                entities_by_doc[doc_id] = entities
                logger.info(f"Extracted {len(entities)} entities from document {doc_id}")
                
            except Exception as e:
                logger.error(f"Error extracting entities from document {doc_id}: {str(e)}")
                entities_by_doc[doc_id] = []
        
        return entities_by_doc
    
    def identify_patterns(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """
        Identify patterns across documents using multiple techniques.
        
        Args:
            documents: Dictionary mapping document IDs to their content
            
        Returns:
            Dictionary containing all identified patterns and representations
        """
        logger.info(f"Identifying patterns across {len(documents)} documents")
        
        try:
            # Preprocess documents
            preprocessed_docs = self.preprocess_documents(documents)
            
            # Apply multiple pattern identification techniques
            topic_patterns = self.identify_topics(documents)
            cluster_patterns = self.cluster_sentences(preprocessed_docs)
            entity_patterns = self.extract_key_entities(documents)
            
            ## topic information for debugging
            #self.print_topic_information(topic_patterns)
            
            # Combine all patterns into a comprehensive representation
            patterns = {
                'topic_patterns': topic_patterns,
                'cluster_patterns': cluster_patterns,
                'entity_patterns': entity_patterns,
                'preprocessed_docs': preprocessed_docs
            }
            
            logger.info("Successfully identified patterns across documents")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern identification: {str(e)}")
            raise
            
    def print_topic_information(self, topic_patterns: Dict[str, Any]) -> None:
        """
        Print topic information for debugging purposes.
        
        Args:
            topic_patterns: Dictionary containing topic model and related information
        """
        if 'topics' in topic_patterns:
            logger.info("\n===== TOPIC INFORMATION =====\n")
            for topic_idx, keywords in enumerate(topic_patterns['topics']):
                logger.info(f"Topic #{topic_idx}: {', '.join(keywords[:10])}")
            logger.info("\n============================\n")
