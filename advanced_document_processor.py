import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from bs4 import BeautifulSoup
import docx
import re
from pathlib import Path
import nltk
from collections import defaultdict

# Download and handle NLTK resources properly
def setup_nltk():
    """Setup NLTK resources with proper error handling"""
    resources_to_download = [
        'punkt',
        'punkt_tab', 
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources_to_download:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                print(f"📦 Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except:
                print(f"⚠️  Could not download {resource}, will use fallback method")

# Setup NLTK
setup_nltk()

# Fallback sentence tokenizer if NLTK fails
def fallback_sent_tokenize(text):
    """Fallback sentence tokenizer using regex"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def safe_sent_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except:
        return fallback_sent_tokenize(text)

class AdvancedDocumentProcessor:
    def __init__(self, 
                 embedding_model_name: str = "all-mpnet-base-v2", 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """
        Advanced document processor with robust error handling
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.documents = []
        self.embeddings = []
        self.index = None
        
        # Try to load spaCy, but don't fail if not available
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, ImportError):
            print("⚠️  SpaCy not available, using basic text processing")
            self.nlp = None
    
    def extract_text_from_pdf_robust(self, file_path: str) -> Dict[str, Any]:
        """Robust PDF text extraction with multiple fallback methods"""
        pages_data = []
        
        try:
            # Method 1: Try with PyPDF2 (newer version)
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            cleaned_text = self.clean_text(page_text)
                            
                            if cleaned_text.strip():
                                pages_data.append({
                                    'page_number': page_num + 1,
                                    'text': cleaned_text,
                                    'word_count': len(cleaned_text.split()),
                                    'char_count': len(cleaned_text)
                                })
                        except Exception as page_error:
                            print(f"  ⚠️  Error extracting page {page_num + 1}: {str(page_error)[:100]}")
                            continue
                            
                except Exception as pdf_error:
                    print(f"  ❌ PyPDF2 extraction failed: {str(pdf_error)[:100]}")
                    
                    # Method 2: Try alternative PDF libraries
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            for page_num, page in enumerate(pdf.pages):
                                try:
                                    page_text = page.extract_text()
                                    if page_text:
                                        cleaned_text = self.clean_text(page_text)
                                        if cleaned_text.strip():
                                            pages_data.append({
                                                'page_number': page_num + 1,
                                                'text': cleaned_text,
                                                'word_count': len(cleaned_text.split()),
                                                'char_count': len(cleaned_text)
                                            })
                                except:
                                    continue
                    except ImportError:
                        print("  💡 Consider installing pdfplumber for better PDF support: pip install pdfplumber")
                
        except Exception as e:
            print(f"  ❌ Could not process PDF: {str(e)[:100]}")
            return {'pages': [], 'total_pages': 0, 'full_text': ''}
        
        # If we got some pages, process them
        if pages_data:
            full_text = ' '.join([page['text'] for page in pages_data])
            return {
                'pages': pages_data,
                'total_pages': len(pages_data),
                'full_text': full_text
            }
        else:
            print(f"  ❌ No text could be extracted from PDF")
            return {'pages': [], 'total_pages': 0, 'full_text': ''}
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better error handling"""
        if not text:
            return ""
        
        try:
            # Handle encoding issues
            if isinstance(text, bytes):
                try:
                    text = text.decode('utf-8', errors='ignore')
                except:
                    text = str(text, errors='ignore')
            
            # Remove control characters and handle surrogates
            text = ''.join(char for char in text if ord(char) < 65536 and char.isprintable() or char.isspace())
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Fix common PDF extraction issues
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
            text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
            text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
            
            # Remove excessive punctuation but keep meaningful ones
            text = re.sub(r'[^\w\s.!?,:;()\-\'\"$%]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            
            return text
            
        except Exception as e:
            print(f"  ⚠️  Text cleaning error: {e}")
            # Return basic cleaned version
            return ' '.join(str(text).split()) if text else ""
    
    def extract_text_from_docx_robust(self, file_path: str) -> Dict[str, Any]:
        """Robust DOCX extraction"""
        try:
            doc = docx.Document(file_path)
            paragraphs = []
            full_text = ""
            
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    cleaned_text = self.clean_text(paragraph.text)
                    if cleaned_text:
                        paragraphs.append({
                            'paragraph_number': i + 1,
                            'text': cleaned_text,
                            'style': paragraph.style.name if paragraph.style else 'Normal'
                        })
                        full_text += cleaned_text + " "
            
            return {
                'paragraphs': paragraphs,
                'full_text': full_text.strip()
            }
        except Exception as e:
            print(f"  ❌ Error reading DOCX: {e}")
            return {'paragraphs': [], 'full_text': ''}
    
    def extract_text_from_html_robust(self, file_path: str) -> Dict[str, Any]:
        """Robust HTML extraction"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove unwanted elements
                for script in soup(["script", "style", "meta", "link"]):
                    script.decompose()
                
                sections = []
                for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li']):
                    text = tag.get_text().strip()
                    if text:
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            sections.append({
                                'tag': tag.name,
                                'text': cleaned_text,
                                'class': tag.get('class', [])
                            })
                
                full_text = ' '.join([section['text'] for section in sections])
                return {
                    'sections': sections,
                    'full_text': full_text
                }
        except Exception as e:
            print(f"  ❌ Error reading HTML: {e}")
            return {'sections': [], 'full_text': ''}
    
    def extract_text_from_txt_robust(self, file_path: str) -> Dict[str, Any]:
        """Robust TXT extraction with encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'ascii']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                        content = file.read()
                        break
                except:
                    continue
            
            if content is None:
                print(f"  ❌ Could not read file with any encoding")
                return {'paragraphs': [], 'full_text': ''}
            
            # Clean and split into paragraphs
            content = self.clean_text(content)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            return {
                'paragraphs': paragraphs,
                'full_text': ' '.join(paragraphs)
            }
        except Exception as e:
            print(f"  ❌ Error reading TXT: {e}")
            return {'paragraphs': [], 'full_text': ''}
    
    def semantic_chunking_robust(self, text: str, source: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Robust semantic chunking with fallback methods"""
        if not text.strip():
            return []
        
        chunks = []
        
        # Use safe sentence tokenization
        sentences = safe_sent_tokenize(text)
        
        if not sentences:  # Fallback to simple splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        current_chunk = ""
        current_sentences = []
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it's substantial
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_data = {
                        'text': current_chunk.strip(),
                        'source': source,
                        'chunk_id': chunk_id,
                        'sentence_count': len(current_sentences),
                        'char_count': len(current_chunk),
                        'word_count': len(current_chunk.split())
                    }
                    
                    if metadata:
                        chunk_data.update(metadata)
                    
                    chunks.append(chunk_data)
                    chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 1 else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
        
        # Add the last chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunk_data = {
                'text': current_chunk.strip(),
                'source': source,
                'chunk_id': chunk_id,
                'sentence_count': len(current_sentences),
                'char_count': len(current_chunk),
                'word_count': len(current_chunk.split())
            }
            
            if metadata:
                chunk_data.update(metadata)
            
            chunks.append(chunk_data)
        
        return chunks
    
    def extract_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Robust document extraction and chunking"""
        file_extension = Path(file_path).suffix.lower()
        all_chunks = []
        
        try:
            if file_extension == '.pdf':
                pdf_data = self.extract_text_from_pdf_robust(file_path)
                
                if pdf_data['pages']:
                    # Process each page
                    for page_data in pdf_data['pages']:
                        page_chunks = self.semantic_chunking_robust(
                            page_data['text'], 
                            file_path,
                            {
                                'page_number': page_data['page_number'],
                                'document_type': 'pdf',
                                'total_pages': pdf_data['total_pages']
                            }
                        )
                        all_chunks.extend(page_chunks)
                else:
                    # If page-wise failed, try full document
                    if pdf_data['full_text']:
                        doc_chunks = self.semantic_chunking_robust(
                            pdf_data['full_text'],
                            file_path,
                            {'document_type': 'pdf', 'extraction_method': 'full_document'}
                        )
                        all_chunks.extend(doc_chunks)
            
            elif file_extension == '.docx':
                docx_data = self.extract_text_from_docx_robust(file_path)
                if docx_data['full_text']:
                    doc_chunks = self.semantic_chunking_robust(
                        docx_data['full_text'],
                        file_path,
                        {
                            'document_type': 'docx',
                            'paragraph_count': len(docx_data['paragraphs'])
                        }
                    )
                    all_chunks.extend(doc_chunks)
            
            elif file_extension in ['.html', '.htm']:
                html_data = self.extract_text_from_html_robust(file_path)
                if html_data['full_text']:
                    doc_chunks = self.semantic_chunking_robust(
                        html_data['full_text'],
                        file_path,
                        {
                            'document_type': 'html',
                            'section_count': len(html_data['sections'])
                        }
                    )
                    all_chunks.extend(doc_chunks)
            
            elif file_extension == '.txt':
                txt_data = self.extract_text_from_txt_robust(file_path)
                if txt_data['full_text']:
                    doc_chunks = self.semantic_chunking_robust(
                        txt_data['full_text'],
                        file_path,
                        {
                            'document_type': 'txt',
                            'paragraph_count': len(txt_data['paragraphs'])
                        }
                    )
                    all_chunks.extend(doc_chunks)
        
        except Exception as e:
            print(f"  ❌ Unexpected error processing {file_path}: {e}")
        
        return all_chunks
    
    def get_supported_files(self, folder_path: str) -> List[str]:
        """Get supported files"""
        supported_extensions = ['.pdf', '.docx', '.txt', '.html', '.htm']
        file_paths = []
        
        if not os.path.exists(folder_path):
            return []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = Path(file_path).suffix.lower()
                
                if file_extension in supported_extensions:
                    file_paths.append(file_path)
                    
        return sorted(file_paths)
    
    def process_documents_from_folder(self, folder_path: str):
        """Process documents with comprehensive error handling"""
        document_paths = self.get_supported_files(folder_path)
        
        if not document_paths:
            print(f"No supported documents found in {folder_path}")
            return
        
        print(f"🔍 Found {len(document_paths)} documents to process")
        print("📄 Files to be processed:")
        for i, path in enumerate(document_paths, 1):
            relative_path = os.path.relpath(path, folder_path)
            try:
                file_size = os.path.getsize(path) / 1024
                print(f"  {i:2d}. {relative_path} ({file_size:.1f} KB)")
            except:
                print(f"  {i:2d}. {relative_path} (size unknown)")
        print()
        
        all_chunks = []
        successful_files = 0
        
        for doc_path in document_paths:
            relative_path = os.path.relpath(doc_path, folder_path)
            print(f"📖 Processing: {relative_path}")
            
            try:
                chunks = self.extract_and_chunk_document(doc_path)
                
                if chunks:
                    all_chunks.extend(chunks)
                    avg_chunk_size = np.mean([chunk['char_count'] for chunk in chunks])
                    print(f"  ✅ Created {len(chunks)} chunks (avg: {avg_chunk_size:.0f} chars)")
                    successful_files += 1
                else:
                    print(f"  ⚠️  No readable content found in {relative_path}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {relative_path}: {str(e)[:100]}")
        
        self.documents = all_chunks
        print(f"\n📊 Processing Summary:")
        print(f"  Successfully processed: {successful_files}/{len(document_paths)} files")
        print(f"  Total chunks created: {len(self.documents)}")
        
        if self.documents:
            chunk_sizes = [doc['char_count'] for doc in self.documents]
            print(f"  Average chunk size: {np.mean(chunk_sizes):.0f} chars")
            print(f"  Size range: {np.min(chunk_sizes)} - {np.max(chunk_sizes)} chars")
            
            # Create embeddings
            print(f"\n🧠 Creating embeddings...")
            texts = [doc['text'] for doc in self.documents]
            
            try:
                # Process in smaller batches for stability
                batch_size = 16
                embeddings_list = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    print(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                    embeddings_list.append(batch_embeddings)
                
                self.embeddings = np.vstack(embeddings_list)
                
                # Create FAISS index
                print("🔗 Building FAISS index...")
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(self.embeddings)
                self.index.add(self.embeddings.astype(np.float32))
                
                print(f"✅ FAISS index built with {self.index.ntotal} vectors")
                
            except Exception as e:
                print(f"❌ Error creating embeddings: {e}")
                self.embeddings = None
                self.index = None
        else:
            print("❌ No content could be extracted from any documents")
    
    def save_index(self, index_dir: str):
        """Save index with error handling"""
        if not self.documents:
            print("❌ No documents to save!")
            return
        
        if self.index is None:
            print("❌ No FAISS index to save!")
            return
        
        try:
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
                'embedding_model_dimension': self.embedding_model.get_sentence_embedding_dimension(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'min_chunk_size': self.min_chunk_size,
                'num_documents': len(self.documents),
                'processing_method': 'robust_semantic_chunking',
                'document_stats': {
                    'total_chunks': len(self.documents),
                    'avg_chunk_size': float(np.mean([doc['char_count'] for doc in self.documents])),
                    'chunk_size_stats': {
                        'min': int(np.min([doc['char_count'] for doc in self.documents])),
                        'max': int(np.max([doc['char_count'] for doc in self.documents])),
                        'std': float(np.std([doc['char_count'] for doc in self.documents]))
                    }
                }
            }
            
            config_path = os.path.join(index_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n💾 Successfully saved index to: {index_dir}")
            print(f"  📁 FAISS index: {index_path}")
            print(f"  📁 Documents: {docs_path}")
            print(f"  📁 Config: {config_path}")
            
        except Exception as e:
            print(f"❌ Error saving index: {e}")

def main():
    """Main processing function with comprehensive error handling"""
    
    print("🚀 Starting Robust Document Processing...")
    print("=" * 60)
    
    # Initialize processor
    try:
        processor = AdvancedDocumentProcessor(
            embedding_model_name="all-mpnet-base-v2",
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100
        )
        print("✅ Document processor initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return
    
    documents_folder = "documents"
    
    if not os.path.exists(documents_folder):
        os.makedirs(documents_folder)
        print(f"📁 Created {documents_folder} folder.")
        print("📝 Please add your documents and run again.")
        return
    
    # Process documents
    processor.process_documents_from_folder(documents_folder)
    
    # Save if successful
    if processor.documents and processor.index is not None:
        processor.save_index("college_rag_index")
        print(f"\n🎉 SUCCESS! Processed {len(processor.documents)} chunks from your documents!")
        print("\n📝 Next steps:")
        print("   1. Set your Groq API key in enhanced_rag_query.py")
        print("   2. Run: python enhanced_rag_query.py")
        print("   3. Start asking questions about your college!")
    else:
        print("\n❌ Processing failed. Please check your documents and try again.")
        print("\n💡 Tips:")
        print("   • Make sure PDFs are not password protected")
        print("   • Check if files are corrupted")
        print("   • Try with a smaller set of documents first")

if __name__ == "__main__":
    main()
