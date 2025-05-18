"""
Document Extractor Module

This module handles the extraction and basic preprocessing of text documents
from a specified directory.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """
    A class to handle document extraction operations.
    
    This class provides functionality to list and read text files from a
    specified directory, with basic preprocessing capabilities.
    """
    
    def __init__(self, documents_directory: str):
        """
        Initialize the DocumentExtractor with the path to the documents directory.
        
        Args:
            documents_directory: Path to the directory containing text documents
        """
        self.documents_directory = documents_directory
        logger.info(f"Initialized DocumentExtractor with directory: {documents_directory}")
        
    def list_text_files(self) -> List[str]:
        """
        List all .txt files in the specified directory.
        
        Returns:
            List of file paths for all .txt files in the directory
        
        Raises:
            FileNotFoundError: If the documents directory does not exist
        """
        try:
            if not os.path.exists(self.documents_directory):
                raise FileNotFoundError(f"Directory not found: {self.documents_directory}")
                
            text_files = []
            for file in os.listdir(self.documents_directory):
                if file.endswith('.txt'):
                    text_files.append(os.path.join(self.documents_directory, file))
            
            logger.info(f"Found {len(text_files)} text files in {self.documents_directory}")
            return text_files
        except Exception as e:
            logger.error(f"Error listing text files: {str(e)}")
            raise
    
    def read_document(self, file_path: str) -> Tuple[str, str]:
        """
        Read the content of a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple containing (document_id, document_content)
            
        Raises:
            FileNotFoundError: If the file does not exist
            IOError: If there's an error reading the file
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            document_id = os.path.basename(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            logger.info(f"Successfully read document: {document_id}")
            return document_id, content
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {str(e)}")
            raise
    
    def extract_all_documents(self) -> Dict[str, str]:
        """
        Extract all text documents from the directory.
        
        Returns:
            Dictionary mapping document IDs to their content
        """
        documents = {}
        try:
            text_files = self.list_text_files()
            
            for file_path in text_files:
                try:
                    document_id, content = self.read_document(file_path)
                    documents[document_id] = content
                except Exception as e:
                    logger.warning(f"Skipping document {file_path} due to error: {str(e)}")
                    continue
            
            logger.info(f"Successfully extracted {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error extracting documents: {str(e)}")
            raise
