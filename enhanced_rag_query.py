import os
import json
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
from collections import defaultdict
import re

# ==========================================
# CONFIGURATION - SET YOUR API KEY HERE
# ==========================================
GROQ_API_KEY = "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"  # Replace with your actual API key
INDEX_DIRECTORY = "college_rag_index"
EMBEDDING_MODEL = "all-mpnet-base-v2"

class EnhancedCollegeRAGSystem:
    def __init__(self, index_dir: str, groq_api_key: str, embedding_model_name: str = "all-mpnet-base-v2"):
        """Enhanced RAG system with better retrieval and context construction"""
        print("🔧 Initializing Enhanced RAG System...")
        print("📦 Loading advanced embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.groq_client = Groq(api_key=groq_api_key)
        
        print("🔍 Loading enhanced FAISS index...")
        self.load_index(index_dir)
        
    def load_index(self, index_dir: str):
        """Load enhanced index with metadata"""
        # Load FAISS index
        index_path = os.path.join(index_dir, "faiss_index.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}\nPlease run the document processor first!")
        
        self.index = faiss.read_index(index_path)
        
        # Load documents with metadata
        docs_path = os.path.join(index_dir, "documents.pkl")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents file not found at {docs_path}")
        
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load enhanced config
        config_path = os.path.join(index_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Create document statistics
        self.document_stats = self._create_document_stats()
        print(f"✅ Loaded enhanced index with {len(self.documents)} chunks")
        print(f"📊 Average chunk size: {self.document_stats['avg_chunk_size']:.0f} characters")
    
    def _create_document_stats(self):
        """Create statistics about loaded documents"""
        if not self.documents:
            return {}
        
        sources = set(doc['source'] for doc in self.documents)
        doc_types = defaultdict(int)
        page_counts = defaultdict(list)
        
        for doc in self.documents:
            doc_type = doc.get('document_type', 'unknown')
            doc_types[doc_type] += 1
            
            if 'page_number' in doc:
                source_name = os.path.basename(doc['source'])
                page_counts[source_name].append(doc['page_number'])
        
        return {
            'total_documents': len(sources),
            'total_chunks': len(self.documents),
            'document_types': dict(doc_types),
            'avg_chunk_size': np.mean([doc['char_count'] for doc in self.documents]),
            'sources': list(sources),
            'page_coverage': {k: f"{min(v)}-{max(v)}" for k, v in page_counts.items() if v}
        }
    
    def enhanced_retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced retrieval with query expansion and re-ranking"""
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Initial retrieval with more candidates for better re-ranking
        search_k = min(top_k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        # Re-rank results based on multiple factors
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['semantic_score'] = float(score)
                
                # Calculate additional relevance factors
                doc['query_term_overlap'] = self._calculate_term_overlap(query, doc['text'])
                doc['chunk_quality_score'] = self._calculate_chunk_quality(doc)
                doc['document_relevance'] = self._calculate_document_relevance(query, doc)
                
                # Combined relevance score with weighted factors
                doc['final_score'] = (
                    doc['semantic_score'] * 0.5 +
                    doc['query_term_overlap'] * 0.25 +
                    doc['chunk_quality_score'] * 0.15 +
                    doc['document_relevance'] * 0.1
                )
                
                candidates.append(doc)
        
        # Sort by final score and return top_k
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add rank information and filter for diversity
        final_results = []
        seen_sources = set()
        
        for i, doc in enumerate(candidates):
            doc['rank'] = i + 1
            source_name = os.path.basename(doc['source'])
            
            # Add document if we haven't seen too many from same source
            source_count = sum(1 for r in final_results if os.path.basename(r['source']) == source_name)
            
            if source_count < 3 or len(final_results) < top_k // 2:  # Allow diversity
                final_results.append(doc)
                seen_sources.add(source_name)
                
                if len(final_results) >= top_k:
                    break
        
        return final_results
    
    def _calculate_term_overlap(self, query: str, text: str) -> float:
        """Calculate term overlap between query and text"""
        query_terms = set(word.lower() for word in query.split() if len(word) > 2)
        text_terms = set(word.lower() for word in text.split() if len(word) > 2)
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(text_terms))
        return overlap / len(query_terms)
    
    def _calculate_chunk_quality(self, chunk: Dict[str, Any]) -> float:
        """Calculate quality score for a chunk"""
        # Prefer longer, more complete chunks
        size_score = min(chunk['char_count'] / 1000, 1.0)
        
        # Prefer chunks with complete sentences
        sentence_score = min(chunk.get('sentence_count', 1) / 5, 1.0)
        
        # Prefer chunks with good word density
        word_density = chunk['word_count'] / max(chunk['char_count'], 1) * 100
        density_score = min(word_density / 15, 1.0)  # Optimal around 15% word density
        
        return (size_score + sentence_score + density_score) / 3
    
    def _calculate_document_relevance(self, query: str, chunk: Dict[str, Any]) -> float:
        """Calculate document-level relevance"""
        doc_type = chunk.get('document_type', 'unknown')
        
        # Weight different document types based on query
        query_lower = query.lower()
        type_weights = {
            'pdf': 0.8,  # Generally reliable
            'docx': 0.7,
            'txt': 0.6,
            'html': 0.5
        }
        
        base_weight = type_weights.get(doc_type, 0.5)
        
        # Boost relevance for documents that seem more official
        if any(term in query_lower for term in ['fee', 'cost', 'admission', 'requirement']):
            if 'fee' in chunk['source'].lower() or 'admission' in chunk['source'].lower():
                base_weight += 0.2
        
        return min(base_weight, 1.0)
    
    def create_enhanced_context(self, retrieved_docs: List[Dict[str, Any]], query: str) -> str:
        """Create enhanced context with better organization and structure"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        context_parts.append(f"USER QUERY: {query}")
        context_parts.append("=" * 60)
        context_parts.append("")
        
        # Group chunks by source document for better organization
        docs_by_source = defaultdict(list)
        for doc in retrieved_docs:
            source_name = os.path.basename(doc['source'])
            docs_by_source[source_name].append(doc)
        
        # Sort sources by relevance (average score of chunks)
        sorted_sources = sorted(docs_by_source.items(), 
                              key=lambda x: np.mean([doc['final_score'] for doc in x[1]]), 
                              reverse=True)
        
        for doc_num, (source_name, chunks) in enumerate(sorted_sources, 1):
            context_parts.append(f"DOCUMENT {doc_num}: {source_name}")
            context_parts.append("-" * 50)
            
            # Sort chunks within document by page number if available, then by score
            chunks.sort(key=lambda x: (x.get('page_number', 999), -x['final_score']))
            
            for chunk_num, chunk in enumerate(chunks, 1):
                # Create detailed chunk header
                metadata_parts = []
                if 'page_number' in chunk:
                    metadata_parts.append(f"Page {chunk['page_number']}")
                if 'chunk_id' in chunk:
                    metadata_parts.append(f"Section {chunk['chunk_id'] + 1}")
                
                metadata_parts.append(f"Relevance: {chunk['final_score']:.3f}")
                metadata_str = " | ".join(metadata_parts)
                
                context_parts.append(f"\nCHUNK {chunk_num} ({metadata_str}):")
                context_parts.append(chunk['text'])
                context_parts.append("")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_enhanced_answer(self, query: str, context: str) -> str:
        """Generate answer with enhanced prompting and better instructions"""
        
        system_prompt = """You are an expert college information assistant with access to comprehensive college documentation. Your role is to provide accurate, detailed, and helpful information about all aspects of college life including admissions, academics, fees, facilities, and student services.

CORE INSTRUCTIONS:
1. Use ONLY the information provided in the context to answer questions
2. If information is incomplete or missing, clearly state what additional details might be needed
3. Provide specific details when available (exact dates, amounts, requirements, procedures, etc.)
4. Structure responses with clear formatting using bullet points, numbered lists, or sections
5. When multiple documents contain relevant information, synthesize them coherently
6. Always cite page numbers or document sections when mentioned in the context
7. Be comprehensive but well-organized - include all relevant details without unnecessary repetition

RESPONSE FORMATTING:
- Start with a direct, clear answer to the main question
- Use bullet points (•) for lists and sub-points
- Use numbered lists (1, 2, 3) for sequential processes or procedures
- Use clear section headers when covering multiple topics
- End with any relevant additional context, related information, or limitations
- If the context lacks sufficient information, clearly state what's missing

CITATION REQUIREMENTS:
- When referencing specific information, mention the source document and page number when available
- Example: "According to the Fee Structure document (Page 3)..."
- For processes or procedures, cite the relevant document sections

ANSWER QUALITY:
- Prioritize accuracy over completeness
- Be specific rather than general
- Provide actionable information when possible
- Maintain a helpful, professional tone"""

        user_prompt = f"""Based on the provided context from college documents, please provide a comprehensive and well-structured answer to the student's question.

CONTEXT INFORMATION:
{context}

STUDENT QUESTION: {query}

Please analyze the context thoroughly and provide a detailed, well-organized answer that directly addresses the question using only the information available in the context above."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,  # Increased for more comprehensive responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}\nPlease check your Groq API key and internet connection."
    
    def query(self, question: str, top_k: int = 10, show_debug: bool = False) -> Dict[str, Any]:
        """Enhanced query processing with comprehensive results"""
        
        # Enhanced retrieval
        retrieved_docs = self.enhanced_retrieve(question, top_k)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information in the college documents to answer your question. Please try:\n• Rephrasing your question with different keywords\n• Being more specific about what you're looking for\n• Asking about a different topic covered in the documents",
                'sources': [],
                'debug_info': {'retrieved_chunks': 0, 'search_terms': question.split()}
            }
        
        # Create enhanced context
        context = self.create_enhanced_context(retrieved_docs, question)
        
        # Generate enhanced answer
        answer = self.generate_enhanced_answer(question, context)
        
        # Prepare detailed source information
        sources = []
        for doc in retrieved_docs:
            source_info = {
                'source': os.path.basename(doc['source']),
                'relevance_score': doc['final_score'],
                'semantic_score': doc['semantic_score'],
                'page_number': doc.get('page_number', 'N/A'),
                'chunk_size': doc['char_count'],
                'word_count': doc['word_count'],
                'document_type': doc.get('document_type', 'unknown'),
                'preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            }
            sources.append(source_info)
        
        result = {
            'answer': answer,
            'sources': sources,
            'query_stats': {
                'retrieved_chunks': len(retrieved_docs),
                'unique_sources': len(set(s['source'] for s in sources)),
                'avg_relevance': np.mean([s['relevance_score'] for s in sources]),
                'coverage_span': f"{min(s['page_number'] for s in sources if s['page_number'] != 'N/A')}-{max(s['page_number'] for s in sources if s['page_number'] != 'N/A')}" if any(s['page_number'] != 'N/A' for s in sources) else 'N/A'
            }
        }
        
        if show_debug:
            result['debug_info'] = {
                'context_length': len(context),
                'context': context,
                'retrieved_docs': retrieved_docs
            }
        
        return result

def display_enhanced_welcome():
    """Display enhanced welcome message with system capabilities"""
    print("=" * 85)
    print("🎓 ENHANCED COLLEGE RAG SYSTEM - Advanced AI Assistant")
    print("=" * 85)
    print("\n🚀 ADVANCED FEATURES:")
    print("   • Multi-factor relevance scoring with semantic similarity")
    print("   • Page-aware document processing and citation")
    print("   • Enhanced context construction with source organization")
    print("   • Intelligent query expansion and re-ranking")
    print("   • Comprehensive error handling and fallback methods")
    print("\n📚 WHAT I CAN HELP YOU WITH:")
    print("   • 🎯 Admission Requirements & Processes")
    print("     - Application procedures, deadlines, eligibility criteria")
    print("     - Required documents, entrance exams, selection process")
    print("   • 💰 Fee Structure & Financial Information")
    print("     - Tuition fees, additional charges, payment schedules")
    print("     - Scholarships, financial aid, refund policies")
    print("   • 📖 Academic Programs & Curriculum")
    print("     - Course details, program structure, specializations")
    print("     - Credit requirements, grading system, academic policies")
    print("   • 🏫 Campus Life & Facilities")
    print("     - Hostel accommodation, dining, transportation")
    print("     - Libraries, labs, sports facilities, medical services")
    print("   • 👨‍🏫 Faculty & Staff Information")
    print("     - Department details, faculty profiles, contact information")
    print("   • 🎭 Student Activities & Services")
    print("     - Clubs, societies, events, student support services")
    print("   • 💼 Placement & Career Services")
    print("     - Placement statistics, recruiting companies, career support")
    print("\n⚡ SPECIAL COMMANDS:")
    print("   • 'help' - Show detailed usage tips and examples")
    print("   • 'stats' - Display system statistics and document coverage")
    print("   • 'debug [your question]' - Show detailed retrieval information")
    print("   • 'quit', 'exit', or 'q' - End the session")
    print("\n💡 PRO TIPS:")
    print("   • Be specific in your questions for more accurate results")
    print("   • Ask follow-up questions to dive deeper into topics")
    print("   • Use keywords related to your area of interest")
    print("\n" + "=" * 85 + "\n")

def display_help():
    """Display comprehensive help information"""
    print("\n" + "=" * 70)
    print("📖 COMPREHENSIVE HELP GUIDE")
    print("=" * 70)
    print("\n🎯 EFFECTIVE QUESTIONING TECHNIQUES:")
    print("\n   SPECIFIC vs GENERAL:")
    print("   ✅ Good: 'What documents are required for MBA admission?'")
    print("   ❌ Avoid: 'What do I need?'")
    print("\n   ✅ Good: 'What is the fee structure for Computer Science program?'")
    print("   ❌ Avoid: 'How much does it cost?'")
    print("\n📋 EXAMPLE QUESTIONS BY CATEGORY:")
    print("\n   🎓 ADMISSIONS:")
    print("   • What are the eligibility criteria for the Engineering program?")
    print("   • When is the application deadline for MBA admissions?")
    print("   • What is the selection process for undergraduate programs?")
    print("   • Which entrance exams are accepted for admission?")
    print("\n   💰 FEES & FINANCES:")
    print("   • What is the complete fee structure for the first year?")
    print("   • Are there any scholarships available for meritorious students?")
    print("   • What is the refund policy for course fees?")
    print("   • How can I pay the fees in installments?")
    print("\n   🏫 FACILITIES & CAMPUS:")
    print("   • What hostel facilities are available for students?")
    print("   • Describe the library resources and timings")
    print("   • What sports and recreational facilities are provided?")
    print("   • Is transportation provided by the college?")
    print("\n   📚 ACADEMICS:")
    print("   • What is the curriculum structure for the CS program?")
    print("   • How many credits are required for graduation?")
    print("   • What are the assessment methods used?")
    print("   • Are there any industry internship programs?")
    print("\n🔍 ADVANCED QUERY TECHNIQUES:")
    print("   • Multi-part questions: 'What are the admission requirements and fee structure for MBA?'")
    print("   • Comparative queries: 'Compare the Computer Science and Information Technology programs'")
    print("   • Process-oriented: 'What are the step-by-step procedures for online admission?'")
    print("   • Specific details: 'What is the hostel fee per semester including mess charges?'")
    print("\n⚡ SYSTEM COMMANDS:")
    print("   • 'stats' - View document coverage and system statistics")
    print("   • 'debug [question]' - See detailed retrieval and ranking information")
    print("   • 'help' - Display this help guide")
    print("   • 'quit' or 'exit' - End the session")
    print("\n" + "=" * 70 + "\n")

def main():
    """Enhanced interactive interface with comprehensive features"""
    
    # Validate API key (check for placeholder)
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("❌ Please set your Groq API key in the GROQ_API_KEY variable at the top of this file.")
        print("\n📝 Steps to get your API key:")
        print("   1. Visit: https://console.groq.com/")
        print("   2. Sign up for an account")
        print("   3. Generate an API key")
        print("   4. Replace the placeholder with your actual key")
        return
    
    try:
        rag_system = EnhancedCollegeRAGSystem(
            index_dir=INDEX_DIRECTORY,
            groq_api_key=GROQ_API_KEY,
            embedding_model_name=EMBEDDING_MODEL
        )
        
        display_enhanced_welcome()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n🔧 SOLUTION:")
        print("   1. Make sure you have documents in the 'documents' folder")
        print("   2. Run the document processor first:")
        print("      python advanced_document_processor.py")
        print("   3. Then run this query system again")
        return
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Check your internet connection")
        print("   • Verify your Groq API key is correct")
        print("   • Make sure all required packages are installed")
        return
    
    query_count = 0
    
    while True:
        try:
            question = input("🤔 Your question: ").strip()
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Enhanced College RAG System!")
                print("📚 Hope you found the information helpful!")
                break
            
            elif question.lower() == 'stats':
                print(f"\n📊 SYSTEM STATISTICS:")
                print("=" * 50)
                print(f"📄 Total Documents: {rag_system.document_stats['total_documents']}")
                print(f"🔢 Total Chunks: {rag_system.document_stats['total_chunks']}")
                print(f"📏 Average Chunk Size: {rag_system.document_stats['avg_chunk_size']:.0f} characters")
                print(f"📋 Document Types: {rag_system.document_stats['document_types']}")
                
                print(f"\n📚 SOURCE DOCUMENTS:")
                for i, source in enumerate(rag_system.document_stats['sources'], 1):
                    source_name = os.path.basename(source)
                    page_info = rag_system.document_stats['page_coverage'].get(source_name, "N/A")
                    print(f"   {i:2d}. {source_name} (Pages: {page_info})")
                
                print("=" * 50 + "\n")
                continue
            
            elif question.lower() == 'help':
                display_help()
                continue
            
            elif question.lower().startswith('debug '):
                debug_question = question[6:].strip()
                if debug_question:
                    query_count += 1
                    print(f"\n🔍 DEBUG MODE - Processing query #{query_count}...")
                    print("🔬 Showing detailed retrieval information...")
                    
                    result = rag_system.query(debug_question, top_k=8, show_debug=True)
                    
                    print(f"\n🤖 **ANSWER:**")
                    print("=" * 70)
                    print(result['answer'])
                    print("=" * 70)
                    
                    print(f"\n📊 **QUERY STATISTICS:**")
                    stats = result['query_stats']
                    print(f"   Retrieved chunks: {stats['retrieved_chunks']}")
                    print(f"   Unique sources: {stats['unique_sources']}")
                    print(f"   Average relevance: {stats['avg_relevance']:.3f}")
                    print(f"   Page coverage: {stats['coverage_span']}")
                    
                    if result['sources']:
                        print(f"\n🔍 **DETAILED SOURCES:**")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"\n   {i}. 📄 {source['source']}")
                            if source['page_number'] != 'N/A':
                                print(f"      📍 Page: {source['page_number']}")
                            print(f"      🎯 Relevance Score: {source['relevance_score']:.3f}")
                            print(f"      🧠 Semantic Score: {source['semantic_score']:.3f}")
                            print(f"      📏 Chunk Size: {source['chunk_size']} chars ({source['word_count']} words)")
                            print(f"      📄 Type: {source['document_type'].upper()}")
                            print(f"      👁️  Preview: {source['preview']}")
                    
                    print("\n" + "=" * 85 + "\n")
                else:
                    print("❓ Please provide a question after 'debug'. Example: debug What are the admission requirements?")
                continue
            
            elif not question:
                print("❓ Please enter a question or use 'help' for guidance.")
                continue
            
            # Process regular query
            query_count += 1
            print(f"\n🔍 Processing query #{query_count}...")
            print("⚡ Using advanced semantic matching and retrieval...")
            
            result = rag_system.query(question, top_k=10)
            
            # Display enhanced answer
            print(f"\n🤖 **ANSWER:**")
            print("=" * 75)
            print(result['answer'])
            print("=" * 75)
            
            # Display query statistics
            stats = result['query_stats']
            print(f"\n📊 **QUERY SUMMARY:**")
            print(f"   📄 Sources consulted: {stats['unique_sources']} documents")
            print(f"   🔢 Chunks analyzed: {stats['retrieved_chunks']}")
            print(f"   🎯 Average relevance: {stats['avg_relevance']:.3f}")
            if stats['coverage_span'] != 'N/A':
                print(f"   📖 Page range: {stats['coverage_span']}")
            
            # Display source information
            if result['sources']:
                print(f"\n📚 **SOURCES** (Top {len(result['sources'])} matches):")
                for i, source in enumerate(result['sources'], 1):
                    page_info = f", Page {source['page_number']}" if source['page_number'] != 'N/A' else ""
                    print(f"   {i:2d}. 📄 {source['source']}{page_info}")
                    print(f"       🎯 Relevance: {source['relevance_score']:.3f} | 📏 {source['chunk_size']} chars")
            
            print("\n" + "=" * 85 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing your question: {e}")
            print("💡 Please try rephrasing your question or use 'help' for guidance.\n")

if __name__ == "__main__":
    main()
