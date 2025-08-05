import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ==========================================
# CONFIGURATION - UPDATE YOUR API KEY HERE
# ==========================================


INDEX_DIRECTORY = "college_rag_index"     # Directory containing the FAISS index
EMBEDDING_MODEL = "all-mpnet-base-v2"     # Embedding model name

class CollegeRAGSystem:
    def __init__(self, index_dir: str, groq_api_key: str, embedding_model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the RAG system
        
        Args:
            index_dir: Directory containing the FAISS index and documents
            groq_api_key: Groq API key for LLM inference
            embedding_model_name: Name of the SentenceTransformer model
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Load index and documents
        self.load_index(index_dir)
        
    def load_index(self, index_dir: str):
        """Load FAISS index and documents from disk"""
        
        # Load FAISS index
        index_path = os.path.join(index_dir, "faiss_index.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load documents
        docs_path = os.path.join(index_dir, "documents.pkl")
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents file not found at {docs_path}")
        
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load config
        config_path = os.path.join(index_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        print(f"✅ Loaded index with {len(self.documents)} document chunks")
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks with scores
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Retrieve documents
        retrieved_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # Ensure valid index
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(score)
                doc['rank'] = i + 1
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def create_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create context string from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            source_name = os.path.basename(doc['source'])
            context_parts.append(
                f"Document {i+1} (Source: {source_name}):\n{doc['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using Groq LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        
        system_prompt = """You are a helpful assistant for a college information system. You provide accurate, detailed, and helpful information about college admission processes, fee structures, programs, curriculum, student activities, placements, and facilities.

Instructions:
- Use only the information provided in the context to answer questions
- If the context doesn't contain enough information to answer the question, say so clearly
- Provide specific details when available (dates, amounts, requirements, etc.)
- Structure your response clearly with bullet points or sections when appropriate
- Be concise but comprehensive
- If asked about multiple topics, organize your response accordingly"""

        user_prompt = f"""Context Information:
{context}

Question: {query}

Please provide a detailed and helpful answer based on the context information above."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",  # You can change this to other Groq models
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, top_k: int = 5, show_sources: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query pipeline
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            show_sources: Whether to include source information in response
            
        Returns:
            Dictionary containing answer and metadata
        """
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, top_k)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'retrieved_docs': []
            }
        
        # Create context
        context = self.create_context(retrieved_docs)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Prepare sources
        sources = []
        if show_sources:
            for doc in retrieved_docs:
                source_info = {
                    'source': os.path.basename(doc['source']),
                    'relevance_score': doc['relevance_score'],
                    'text_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                }
                sources.append(source_info)
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_docs': retrieved_docs,
            'context': context
        }

def display_welcome_message():
    """Display welcome message and instructions"""
    print("=" * 70)
    print("🎓 COLLEGE RAG SYSTEM - AI-Powered College Information Assistant")
    print("=" * 70)
    print("\n📚 Ask me anything about the college including:")
    print("   • Admission process and requirements")
    print("   • Fee structure and payment details")
    print("   • Academic programs and curriculum")
    print("   • Student clubs and activities")
    print("   • Placement statistics and opportunities")
    print("   • Hostel and accommodation facilities")
    print("   • Any other college-related information")
    print("\n💡 Tips:")
    print("   • Be specific with your questions for better answers")
    print("   • Type 'quit', 'exit', or 'q' to stop")
    print("   • Type 'help' for more information")
    print("\n" + "=" * 70 + "\n")

def display_help():
    """Display help information"""
    print("\n" + "=" * 50)
    print("📖 HELP - How to use this system effectively:")
    print("=" * 50)
    print("\n🔍 Example Questions:")
    print("   • What is the admission process for computer science?")
    print("   • How much is the tuition fee for MBA program?")
    print("   • What are the hostel facilities available?")
    print("   • Tell me about placement statistics")
    print("   • What clubs and activities are available?")
    print("\n⚡ Commands:")
    print("   • 'help' - Show this help message")
    print("   • 'quit', 'exit', 'q' - Exit the program")
    print("   • 'clear' - Clear the screen")
    print("\n" + "=" * 50 + "\n")

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Interactive query interface"""
    
    # Check if API key is set
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        print("❌ Error: Please set your Groq API key in the GROQ_API_KEY variable at the top of this file.")
        print("\n📝 Steps to get your API key:")
        print("   1. Visit: https://console.groq.com/")
        print("   2. Sign up for an account")
        print("   3. Generate an API key")
        print("   4. Replace 'YOUR_GROQ_API_KEY_HERE' with your actual key")
        return
    
    # Initialize RAG system
    try:
        print("🚀 Initializing College RAG System...")
        print("📦 Loading embedding model...")
        print("🔍 Loading FAISS index...")
        
        rag_system = CollegeRAGSystem(
            index_dir=INDEX_DIRECTORY,
            groq_api_key=GROQ_API_KEY,
            embedding_model_name=EMBEDDING_MODEL
        )
        
        display_welcome_message()
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("\n🔧 Possible solutions:")
        print("   • Make sure you have run document_processor.py first")
        print("   • Check that the 'college_rag_index' folder exists")
        print("   • Verify your Groq API key is correct")
        return
    
    # Interactive query loop
    query_count = 0
    
    while True:
        try:
            question = input("🤔 Your question: ").strip()
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using College RAG System. Goodbye!")
                break
            
            elif question.lower() == 'help':
                display_help()
                continue
            
            elif question.lower() == 'clear':
                clear_screen()
                display_welcome_message()
                continue
            
            elif not question:
                print("❓ Please enter a question or type 'help' for assistance.")
                continue
            
            # Process the query
            query_count += 1
            print(f"\n🔍 Searching for relevant information... (Query #{query_count})")
            
            result = rag_system.query(question, top_k=5)
            
            # Display the answer
            print(f"\n🤖 **Answer:**")
            print("-" * 50)
            print(result['answer'])
            print("-" * 50)
            
            # Display sources if available
            if result['sources']:
                print(f"\n📋 **Sources** (Top {len(result['sources'])} relevant documents):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. 📄 {source['source']} (Relevance: {source['relevance_score']:.3f})")
            
            print("\n" + "=" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.\n")

if __name__ == "__main__":
    main()
