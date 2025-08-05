import os
import json
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from bs4 import BeautifulSoup
import docx
import re
from pathlib import Path
import glob

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = "all-mpnet-base-v2", chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor
        
        Args:
            embedding_model_name: Name of the SentenceTransformer model
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.embeddings = []
        self.index = None
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_html(self, file_path: str) -> str:
        """Extract text from HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                return text
        except Exception as e:
            print(f"Error reading HTML {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension in ['.html', '.htm']:
            return self.extract_text_from_html(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file format: {file_extension}")
            return ""
    
    def get_supported_files(self, folder_path: str) -> List[str]:
        """
        Get all supported files from a folder and its subdirectories
        
        Args:
            folder_path: Path to the folder containing documents
            
        Returns:
            List of file paths with supported extensions
        """
        supported_extensions = ['.pdf', '.docx', '.txt', '.html', '.htm']
        file_paths = []
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist!")
            return []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension in supported_extensions:
                    file_paths.append(file_path)
                    
        return sorted(file_paths)  # Sort for consistent processing order
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            source: Source file path
            
        Returns:
            List of chunk dictionaries with text, source, and metadata
        """
        chunks = []
        text = re.sub(r'\s+', ' ', text.strip())  # Clean up whitespace
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('.', start + self.chunk_size - 100, end)
                if sentence_end != -1 and sentence_end > start + 100:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': chunk_id,
                    'start_idx': start,
                    'end_idx': end
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
                
        return chunks
    
    def process_documents_from_folder(self, folder_path: str):
        """
        Process all supported documents in a folder and create embeddings
        
        Args:
            folder_path: Path to folder containing documents
        """
        # Get all supported files
        document_paths = self.get_supported_files(folder_path)
        
        if not document_paths:
            print(f"No supported documents found in {folder_path}")
            print("Supported formats: PDF, DOCX, TXT, HTML")
            return
        
        print(f"Found {len(document_paths)} supported documents in {folder_path}")
        print("Files to be processed:")
        for i, path in enumerate(document_paths, 1):
            relative_path = os.path.relpath(path, folder_path)
            print(f"  {i:2d}. {relative_path}")
        print()
        
        all_chunks = []
        
        for doc_path in document_paths:
            relative_path = os.path.relpath(doc_path, folder_path)
            print(f"Processing: {relative_path}")
            text = self.extract_text(doc_path)
            
            if text:
                chunks = self.chunk_text(text, doc_path)
                all_chunks.extend(chunks)
                print(f"  ✓ Created {len(chunks)} chunks")
            else:
                print(f"  ✗ No text extracted from {relative_path}")
        
        self.documents = all_chunks
        print(f"\nTotal chunks created: {len(self.documents)}")
        
        if self.documents:
            # Create embeddings
            print("Creating embeddings...")
            texts = [doc['text'] for doc in self.documents]
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            print("Building FAISS index...")
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype(np.float32))
            
            print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def process_documents(self, document_paths: List[str]):
        """
        Process multiple documents and create embeddings
        
        Args:
            document_paths: List of file paths to process
        """
        print(f"Processing {len(document_paths)} documents...")
        
        all_chunks = []
        
        for doc_path in document_paths:
            print(f"Processing: {doc_path}")
            text = self.extract_text(doc_path)
            
            if text:
                chunks = self.chunk_text(text, doc_path)
                all_chunks.extend(chunks)
                print(f"  Created {len(chunks)} chunks")
            else:
                print(f"  No text extracted from {doc_path}")
        
        self.documents = all_chunks
        print(f"Total chunks created: {len(self.documents)}")
        
        if self.documents:
            # Create embeddings
            print("Creating embeddings...")
            texts = [doc['text'] for doc in self.documents]
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            print("Building FAISS index...")
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype(np.float32))
            
            print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self, index_dir: str):
        """
        Save the FAISS index and documents to disk
        
        Args:
            index_dir: Directory to save the index files
        """
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, "faiss_index.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents metadata
        docs_path = os.path.join(index_dir, "documents.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save configuration
        config = {
            'embedding_model_name': self.embedding_model.get_sentence_embedding_dimension(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'num_documents': len(self.documents)
        }
        config_path = os.path.join(index_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nIndex saved to {index_dir}")
        print(f"  - FAISS index: {index_path}")
        print(f"  - Documents: {docs_path}")
        print(f"  - Config: {config_path}")

def main():
    """Process all documents in the documents folder"""
    
    # Initialize processor
    processor = DocumentProcessor(
        embedding_model_name="all-mpnet-base-v2",
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Define the documents folder path
    documents_folder = "documents"
    
    # Create documents folder if it doesn't exist
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)
        print(f"Created {documents_folder} folder.")
        print("Please add your college documents to this folder and run the script again.")
        print("Supported formats: PDF, DOCX, TXT, HTML")
        return
    
    # Process all documents in the folder
    processor.process_documents_from_folder(documents_folder)
    
    # Save the index if documents were processed
    if processor.documents:
        processor.save_index("college_rag_index")
        print(f"\n✅ Successfully processed {len(processor.documents)} chunks from the documents folder!")
    else:
        print("❌ No documents were processed. Please check the documents folder.")

if __name__ == "__main__":
    main()
