import os
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import your existing RAG system
from enhanced_rag_query import EnhancedCollegeRAGSystem, GROQ_API_KEY, INDEX_DIRECTORY, EMBEDDING_MODEL

# Global variable to hold RAG system instance
rag_system: Optional[EnhancedCollegeRAGSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG system on startup"""
    global rag_system
    try:
        print("🚀 Initializing College RAG System...")
        rag_system = EnhancedCollegeRAGSystem(
            index_dir=INDEX_DIRECTORY,
            groq_api_key=GROQ_API_KEY,
            embedding_model_name=EMBEDDING_MODEL
        )
        print("✅ RAG system initialized successfully!")
        yield
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        rag_system = None
        yield
    finally:
        print("🔄 Shutting down RAG system...")

# Create FastAPI app with lifespan events
app = FastAPI(
    title="Enhanced College RAG API",
    description="Advanced AI-powered college information system with semantic search",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080", "*"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the college", min_length=1, max_length=500)
    top_k: Optional[int] = Field(10, description="Number of top results to retrieve", ge=1, le=20)
    debug: Optional[bool] = Field(False, description="Enable debug mode for detailed information")

class SourceInfo(BaseModel):
    source: str
    relevance_score: float
    semantic_score: float
    page_number: Any  # Can be int or 'N/A'
    chunk_size: int
    word_count: int
    document_type: str
    preview: str

class QueryStats(BaseModel):
    retrieved_chunks: int
    unique_sources: int
    avg_relevance: float
    coverage_span: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    query_stats: QueryStats
    debug_info: Optional[Dict[str, Any]] = None

class SystemStats(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    avg_chunk_size: float
    document_types: Dict[str, int]
    sources: List[str]
    page_coverage: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    message: str
    suggestions: List[str]

# Dependency to ensure RAG system is initialized
def get_rag_system():
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Service Unavailable",
                "message": "RAG system is not initialized",
                "suggestions": [
                    "Make sure the document processor has been run",
                    "Check that the index files exist in the college_rag_index directory",
                    "Verify the Groq API key is valid",
                    "Check system logs for initialization errors"
                ]
            }
        )
    return rag_system

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced College RAG API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    global rag_system
    if rag_system is None:
        return {
            "status": "unhealthy",
            "rag_system": "not_initialized",
            "message": "RAG system failed to initialize"
        }
    
    return {
        "status": "healthy",
        "rag_system": "initialized",
        "documents_loaded": len(rag_system.documents) if hasattr(rag_system, 'documents') else 0,
        "message": "All systems operational"
    }

@app.get("/stats", response_model=SystemStats)
async def get_system_stats(rag: EnhancedCollegeRAGSystem = Depends(get_rag_system)):
    """Get system statistics and document information"""
    try:
        stats = rag.document_stats
        return SystemStats(
            status="active",
            total_documents=stats.get('total_documents', 0),
            total_chunks=stats.get('total_chunks', 0),
            avg_chunk_size=stats.get('avg_chunk_size', 0),
            document_types=stats.get('document_types', {}),
            sources=[os.path.basename(source) for source in stats.get('sources', [])],
            page_coverage=stats.get('page_coverage', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_college_info(
    request: QueryRequest, 
    rag: EnhancedCollegeRAGSystem = Depends(get_rag_system)
):
    """
    Query the college RAG system with advanced semantic search
    
    This endpoint processes natural language questions about college information
    including admissions, fees, programs, facilities, and more.
    """
    try:
        # Validate question
        if not request.question or request.question.strip() == "":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Question",
                    "message": "Question cannot be empty",
                    "suggestions": ["Please provide a valid question about the college"]
                }
            )
        
        # Process query
        result = rag.query(
            question=request.question.strip(), 
            top_k=request.top_k, 
            show_debug=request.debug
        )
        
        # Convert to response model
        sources = [
            SourceInfo(
                source=source['source'],
                relevance_score=source['relevance_score'],
                semantic_score=source['semantic_score'],
                page_number=source['page_number'],
                chunk_size=source['chunk_size'],
                word_count=source['word_count'],
                document_type=source['document_type'],
                preview=source['preview']
            )
            for source in result['sources']
        ]
        
        query_stats = QueryStats(
            retrieved_chunks=result['query_stats']['retrieved_chunks'],
            unique_sources=result['query_stats']['unique_sources'],
            avg_relevance=result['query_stats']['avg_relevance'],
            coverage_span=result['query_stats']['coverage_span']
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            query_stats=query_stats,
            debug_info=result.get('debug_info') if request.debug else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query Processing Error",
                "message": f"An error occurred while processing your question: {str(e)}",
                "suggestions": [
                    "Try rephrasing your question",
                    "Check if your question is related to college information",
                    "Contact support if the problem persists"
                ]
            }
        )

@app.post("/batch-query", response_model=List[QueryResponse])
async def batch_query_college_info(
    requests: List[QueryRequest], 
    rag: EnhancedCollegeRAGSystem = Depends(get_rag_system)
):
    """Process multiple queries in a single request"""
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 10 queries"
        )
    
    results = []
    for req in requests:
        try:
            result = rag.query(
                question=req.question.strip(), 
                top_k=req.top_k, 
                show_debug=req.debug
            )
            
            sources = [
                SourceInfo(**source) for source in result['sources']
            ]
            
            query_stats = QueryStats(**result['query_stats'])
            
            results.append(QueryResponse(
                answer=result['answer'],
                sources=sources,
                query_stats=query_stats,
                debug_info=result.get('debug_info') if req.debug else None
            ))
        except Exception as e:
            results.append(QueryResponse(
                answer=f"Error processing question: {str(e)}",
                sources=[],
                query_stats=QueryStats(
                    retrieved_chunks=0,
                    unique_sources=0,
                    avg_relevance=0.0,
                    coverage_span="N/A"
                )
            ))
    
    return results

@app.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents(rag: EnhancedCollegeRAGSystem = Depends(get_rag_system)):
    """List all available source documents"""
    try:
        stats = rag.document_stats
        documents = []
        
        for source in stats.get('sources', []):
            source_name = os.path.basename(source)
            doc_info = {
                "name": source_name,
                "path": source,
                "page_coverage": stats.get('page_coverage', {}).get(source_name, "N/A"),
                "chunks": len([d for d in rag.documents if d['source'] == source])
            }
            documents.append(doc_info)
        
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/search-suggestions", response_model=List[str])
async def get_search_suggestions():
    """Get suggested search queries for better user experience"""
    suggestions = [
        "What is the admission process for undergraduate programs?",
        "What are the fee structure and payment options?",
        "What facilities are available in the college?",
        "Tell me about the placement statistics and career services",
        "What are the hostel and accommodation facilities?",
        "What academic programs and courses are offered?",
        "What are the eligibility criteria for different programs?",
        "What documents are required for admission?",
        "Are there any scholarships or financial aid available?",
        "What are the college timings and academic calendar?"
    ]
    return suggestions

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/docs - API documentation",
            "/query - Query college information", 
            "/stats - System statistics",
            "/health - Health check"
        ]
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "suggestions": [
            "Try again in a few moments",
            "Check the server logs for more details",
            "Contact support if the problem persists"
        ]
    }

def start_api_server(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    reload: bool = False,
    log_level: str = "info"
):
    """Start the FastAPI server"""
    print("🚀 Starting Enhanced College RAG API Server...")
    print(f"📡 Server will be available at: http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Interactive API Explorer: http://{host}:{port}/redoc")
    
    uvicorn.run(
        "college_rag_api:app",  # module:app
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

if __name__ == "__main__":
    start_api_server(
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
