import hashlib
from datetime import datetime, timezone
from typing import List, Tuple
import traceback

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.services.chunking.models import (
    ChunkingResult,
    ChunkMetadata,
    ChunkQualityMetrics
)
from app.utils.logger import logger
from app.config import settings

class DocumentChunker:
    """Chunk documents using semantic chunking"""
    
    def __init__(self):
        """Initialize chunker with semantic text splitter"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="gradient",
                min_chunk_size=settings.MIN_SEMANTIC_CHUNK_SIZE
            )
            logger.info("Initialized semantic chunker")
        except Exception as e:
            logger.error(f"Failed to initialize chunker: {str(e)}")
            traceback.print_exc()
            raise
    
    def _compute_chunk_quality(self, chunk_text: str) -> ChunkQualityMetrics:
        """Compute quality metrics for a chunk"""
        words = chunk_text.split()
        sentences = [s for s in chunk_text.split('.') if s.strip()]
        
        return ChunkQualityMetrics(
            token_count=len(words),
            sentence_count=len(sentences),
            avg_sentence_length=len(words) / max(len(sentences), 1),
            semantic_score=1.0  # Set by SemanticChunker's internal scoring
        )
    
    async def process_document(
        self,
        text: str,
        transform_id: str
    ) -> Tuple[ChunkingResult, List[ChunkMetadata]]:
        """Process document and return chunks with metadata"""
        try:
            # Semantic chunking
            semantic_start = datetime.now(timezone.utc)
            if(len(text) <= settings.MIN_CHUNK_SIZE):
                documents = [Document(page_content=text)]
            else:
                documents = self.text_splitter.create_documents([text])
            semantic_time = datetime.now(timezone.utc) - semantic_start
            # Process chunks
            chunk_start = datetime.now(timezone.utc)
            chunk_texts = []
            chunk_metadata = []
            
            for i, doc in enumerate(documents):
                chunk_text = doc.page_content
                
                # Find chunk position in original text
                start_idx = text.find(chunk_text)
                end_idx = start_idx + len(chunk_text)
                
                # Compute quality metrics
                quality = self._compute_chunk_quality(chunk_text)
                
                # Create hash
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                
                # Create metadata
                metadata = ChunkMetadata(
                    transform_id=transform_id,
                    chunk_id=f"{transform_id}_chunk_{i}",
                    chunk_index=i,
                    chunk_hash=chunk_hash,
                    start_position=start_idx,
                    end_position=end_idx,
                    chunk_size=len(chunk_text),
                    quality_metrics=quality,
                    processing_timestamp=datetime.now(timezone.utc)
                )
                
                chunk_texts.append(chunk_text)
                chunk_metadata.append(metadata)
            
            chunk_time = datetime.now(timezone.utc) - chunk_start

            print('*'*30)
            print([len(c) for c in chunk_texts])
            print('*'*30)
            
            # Create result
            result = ChunkingResult(
                transform_id=transform_id,
                chunks=chunk_texts,
                num_chunks=len(documents),
                total_tokens=sum(len(c.split()) for c in chunk_texts),
                semantic_processing_time=semantic_time.total_seconds(),
                chunk_processing_time=chunk_time.total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            return result, chunk_metadata
            
        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            traceback.print_exc()
            raise
        
class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass