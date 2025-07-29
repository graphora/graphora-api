from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.services.chunking.config import (
    ChunkingConfig, 
    ChunkingStrategy,
    DEFAULT_CONFIGS
)
from app.services.chunking.chunker import DocumentChunker
from app.services.chunking.hybrid_chunker import DocumentType
from app.utils.logger import logger

from app.config import settings

router = APIRouter(prefix=f"{settings.API_V1_STR}/chunking", tags=["chunking"])

class ChunkingTestRequest(BaseModel):
    """Request to test chunking with different configurations"""
    text: str
    config: Optional[ChunkingConfig] = None
    strategy_override: Optional[ChunkingStrategy] = None

class ChunkingTestResponse(BaseModel):
    """Response from chunking test"""
    chunks: List[str]
    num_chunks: int
    total_tokens: int
    document_type: str
    strategy_used: str
    chunk_sizes: List[int]
    avg_chunk_size: float
    quality_scores: List[float]

class ChunkingAnalysisResponse(BaseModel):
    """Response from document analysis with chunking recommendations"""
    document_type: str
    recommended_strategy: str
    recommended_config: ChunkingConfig
    confidence_score: float
    document_stats: Dict[str, Any]
    chunking_estimate: Dict[str, Any]
    alternative_configs: Dict[str, ChunkingConfig]

@router.get("/strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """Get available chunking strategies and their descriptions"""
    return {
        "strategies": {
            "semantic": {
                "name": "Semantic Chunking",
                "description": "Splits text based on semantic similarity using embeddings",
                "best_for": ["Long narrative documents", "Articles", "Books"],
                "pros": ["Maintains semantic coherence", "Good for continuous text"],
                "cons": ["Ignores document structure", "Computationally expensive"]
            },
            "structural": {
                "name": "Structural Chunking", 
                "description": "Preserves document structure like headings, lists, and quotes",
                "best_for": ["Structured documents", "Documentation", "Lists and outlines"],
                "pros": ["Preserves formatting", "Fast processing", "Good for extraction"],
                "cons": ["May create uneven chunk sizes", "Less semantic awareness"]
            },
            "recursive": {
                "name": "Recursive Character Splitting",
                "description": "Splits text using hierarchical separators with overlap",
                "best_for": ["Technical documents", "Code documentation", "Mixed content"],
                "pros": ["Consistent chunk sizes", "Good overlap handling", "Versatile"],
                "cons": ["May break semantic units", "Less structure awareness"]
            },
            "hybrid": {
                "name": "Hybrid Chunking (Recommended)",
                "description": "Automatically selects best strategy based on document type",
                "best_for": ["All document types", "Unknown content", "Mixed collections"],
                "pros": ["Adaptive", "Best of all strategies", "Intelligent"],
                "cons": ["Slightly more complex", "May need fine-tuning"]
            }
        },
        "document_types": {
            "structured": "Documents with clear structure (headings, lists, sections)",
            "narrative": "Long-form text like articles, essays, stories",
            "technical": "Documentation, manuals, code-related content",
            "mixed": "Documents with combination of different content types"
        }
    }

@router.get("/defaults")
async def get_default_configs() -> Dict[str, ChunkingConfig]:
    """Get default configurations for different document types"""
    return DEFAULT_CONFIGS

@router.post("/test", response_model=ChunkingTestResponse)
async def test_chunking(request: ChunkingTestRequest) -> ChunkingTestResponse:
    """Test chunking configuration with sample text"""
    try:
        # Initialize chunker with provided config
        config = request.config or DEFAULT_CONFIGS["hybrid"]
        chunker = DocumentChunker(config=config)
        
        # Process document
        result, metadata = await chunker.process_document(
            text=request.text,
            transform_id="test_chunk",
            strategy_override=request.strategy_override
        )
        
        # Extract quality scores from metadata
        quality_scores = [
            meta.quality_metrics.semantic_score if meta.quality_metrics else 0.0
            for meta in metadata
        ]
        
        # Classify document type for reference
        from app.services.chunking.hybrid_chunker import DocumentClassifier
        classifier = DocumentClassifier()
        doc_type = classifier.classify_document(request.text)
        
        chunk_sizes = [len(chunk) for chunk in result.chunks]
        
        return ChunkingTestResponse(
            chunks=result.chunks,
            num_chunks=result.num_chunks,
            total_tokens=result.total_tokens,
            document_type=doc_type.value,
            strategy_used=request.strategy_override.value if request.strategy_override else config.strategy.value,
            chunk_sizes=chunk_sizes,
            avg_chunk_size=sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            quality_scores=quality_scores
        )
        
    except Exception as e:
        logger.error(f"Chunking test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chunking test failed: {str(e)}")

@router.post("/validate")
async def validate_chunking_config(config: ChunkingConfig) -> Dict[str, Any]:
    """Validate a chunking configuration"""
    try:
        # Basic validation
        if config.min_chunk_size >= config.max_chunk_size:
            raise HTTPException(
                status_code=400,
                detail="Minimum chunk size must be less than maximum chunk size"
            )
        
        if config.min_chunk_size < 100 or config.max_chunk_size > 50000:
            raise HTTPException(
                status_code=400,
                detail="Chunk sizes must be between 100 and 50,000 characters"
            )
        
        return {
            "valid": True,
            "message": "Configuration is valid",
            "estimated_performance": {
                "processing_speed": "fast" if config.strategy == ChunkingStrategy.STRUCTURAL else "medium",
                "memory_usage": "low" if config.strategy != ChunkingStrategy.SEMANTIC else "medium",
                "quality": "high" if config.strategy == ChunkingStrategy.HYBRID else "medium"
            }
        }
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

class DocumentAnalysisRequest(BaseModel):
    """Request for document analysis"""
    text: str

@router.post("/analyze", response_model=ChunkingAnalysisResponse)
async def analyze_document_for_chunking(request: DocumentAnalysisRequest) -> ChunkingAnalysisResponse:
    """Analyze document and provide chunking recommendations"""
    try:
        from app.services.chunking.hybrid_chunker import DocumentClassifier
        
        classifier = DocumentClassifier()
        doc_type = classifier.classify_document(request.text)
        
        # Get recommended configuration
        recommended_config = DEFAULT_CONFIGS.get(doc_type.value, DEFAULT_CONFIGS["hybrid"])
        
        # Analyze document characteristics
        lines = request.text.split('\n')
        words = request.text.split()
        
        # Calculate confidence score based on document characteristics
        confidence_indicators = {
            "structured": sum([
                any(line.strip().startswith('#') for line in lines) * 0.3,
                any(line.strip().startswith(('1.', '-', '*', '+')) for line in lines) * 0.4,
                (len([l for l in lines if l.strip()]) / len(lines) if lines else 0) * 0.3
            ]),
            "narrative": min(1.0, len(words) / 1000) * 0.8,
            "technical": sum([
                ('function' in request.text.lower()) * 0.2,
                ('class' in request.text.lower()) * 0.2, 
                ('api' in request.text.lower()) * 0.2,
                ('```' in request.text) * 0.4
            ])
        }
        
        confidence_score = confidence_indicators.get(doc_type.value, 0.5)
        
        # Prepare alternative configurations
        alternative_configs = {
            strategy: config for strategy, config in DEFAULT_CONFIGS.items() 
            if strategy != doc_type.value
        }
        
        document_stats = {
            "total_characters": len(request.text),
            "total_words": len(words),
            "total_lines": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "has_headings": any(line.strip().startswith('#') for line in lines),
            "has_lists": any(line.strip().startswith(('1.', '-', '*', '+')) for line in lines),
            "has_quotes": '"' in request.text or "'" in request.text or '>' in request.text,
            "complexity_score": min(1.0, len(set(words)) / len(words) if words else 0)
        }
        
        chunking_estimate = {
            "estimated_chunks": max(1, len(request.text) // recommended_config.max_chunk_size),
            "avg_chunk_size": min(recommended_config.max_chunk_size, len(request.text)),
            "processing_time_estimate": f"{len(request.text) // 1000}s",
            "memory_estimate": f"{len(request.text) // 10000}MB"
        }
        
        return ChunkingAnalysisResponse(
            document_type=doc_type.value,
            recommended_strategy=recommended_config.strategy.value,
            recommended_config=recommended_config,
            confidence_score=confidence_score,
            document_stats=document_stats,
            chunking_estimate=chunking_estimate,
            alternative_configs=alternative_configs
        )
        
    except Exception as e:
        logger.error(f"Document analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")