import hashlib
import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import traceback

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.services.chunking.models import (
    ChunkingResult,
    ChunkMetadata,
    ChunkQualityMetrics
)
from app.utils.logger import logger
from app.config import settings

class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"
    RECURSIVE = "recursive"

class DocumentType(str, Enum):
    """Document type classification"""
    STRUCTURED = "structured"  # Lists, headings, clear structure
    NARRATIVE = "narrative"    # Long form text, essays, articles
    TECHNICAL = "technical"    # Documentation, manuals
    MIXED = "mixed"           # Combination of types

class DocumentClassifier:
    """Classify document type to choose optimal chunking strategy"""
    
    @staticmethod
    def classify_document(text: str) -> DocumentType:
        """Classify document based on structural patterns"""
        lines = text.split('\n')
        text_length = len(text)
        
        # Count structural elements
        heading_count = sum(1 for line in lines if re.match(r'^#+\s+|^[#*-]\s+', line.strip()))
        list_count = sum(1 for line in lines if re.match(r'^\d+\.\s+|^[-*+]\s+', line.strip()))
        paragraph_count = len([line for line in lines if len(line.strip()) > 50])
        
        logger.info(f"Document analysis: {text_length} chars, {len(lines)} lines, "
                   f"{heading_count} headings, {list_count} lists, {paragraph_count} paragraphs")
        
        # Classification logic
        if heading_count > 3 or list_count > 3:
            doc_type = DocumentType.STRUCTURED
        elif len(text) > 5000 and paragraph_count > 10:
            doc_type = DocumentType.NARRATIVE
        elif 'function' in text.lower() or 'class' in text.lower() or 'api' in text.lower():
            doc_type = DocumentType.TECHNICAL
        else:
            doc_type = DocumentType.MIXED
            
        logger.info(f"Document classified as: {doc_type}")
        return doc_type

class StructuralChunker:
    """Structure-aware chunking that preserves document elements"""
    
    def __init__(self, min_chunk_size: int = 500, max_chunk_size: int = 3000, 
                 preserve_lists: bool = True, preserve_headings: bool = True, 
                 preserve_quotes: bool = True):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.preserve_lists = preserve_lists
        self.preserve_headings = preserve_headings
        self.preserve_quotes = preserve_quotes
        logger.info(f"StructuralChunker initialized with min_size={min_chunk_size}, max_size={max_chunk_size}, "
                   f"preserve_lists={preserve_lists}, preserve_headings={preserve_headings}, preserve_quotes={preserve_quotes}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text while preserving structural elements"""
        logger.info(f"Starting structural chunking for text of {len(text)} characters")
        chunks = []
        current_chunk = ""
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Handle headings (keep with following content) - only if preserve_headings is True
            if self.preserve_headings and self._is_heading(line):
                if current_chunk and len(current_chunk) > self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                current_chunk += line + '\n'
                
                # Add following paragraphs until next structural element
                i += 1
                while i < len(lines) and not self._is_structural_element(lines[i]):
                    current_chunk += lines[i] + '\n'
                    i += 1
                continue
            
            # Handle lists (keep complete lists together) - only if preserve_lists is True
            elif self.preserve_lists and self._is_list_item(line):
                list_content = ""
                list_start = i
                
                # Collect complete list
                while i < len(lines) and (self._is_list_item(lines[i]) or lines[i].strip() == ""):
                    list_content += lines[i] + '\n'
                    i += 1
                
                # If current chunk + list exceeds max size, finalize current chunk
                if current_chunk and len(current_chunk + list_content) > self.max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = list_content
                else:
                    current_chunk += list_content
                continue
            
            # Handle quotes and special formatting - only if preserve_quotes is True
            elif self.preserve_quotes and self._is_quote(line):
                quote_content = ""
                while i < len(lines) and (self._is_quote(lines[i]) or lines[i].strip() == ""):
                    quote_content += lines[i] + '\n'
                    i += 1
                current_chunk += quote_content
                continue
            
            # Regular content
            else:
                current_chunk += line + '\n'
                
                # Check if chunk is getting too large
                if len(current_chunk) > self.max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            i += 1
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Structural chunking completed: {len(chunks)} chunks created, "
                   f"sizes: {[len(c) for c in chunks]}")
        return chunks
    
    def _is_heading(self, line: str) -> bool:
        """Check if line is a heading"""
        return bool(re.match(r'^#+\s+|^[A-Z][^.!?]*$', line)) and len(line) < 100
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item"""
        return bool(re.match(r'^\d+\.\s+|^[-*+]\s+', line.strip()))
    
    def _is_quote(self, line: str) -> bool:
        """Check if line is a quote"""
        return line.strip().startswith('>') or '***' in line or line.strip().startswith('"')
    
    def _is_structural_element(self, line: str) -> bool:
        """Check if line is any structural element"""
        return (self._is_heading(line) or 
                self._is_list_item(line) or 
                self._is_quote(line))

class HybridDocumentChunker:
    """Advanced chunker that adapts strategy based on document type"""
    
    def __init__(self, 
                 semantic_chunker: Optional[SemanticChunker] = None,
                 strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
                 config: Optional['ChunkingConfig'] = None):
        self.strategy = strategy
        self.config = config
        self.classifier = DocumentClassifier()
        
        # Initialize structural chunker with config parameters
        if config:
            self.structural_chunker = StructuralChunker(
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size,
                preserve_lists=config.preserve_lists,
                preserve_headings=config.preserve_headings,
                preserve_quotes=config.preserve_quotes
            )
        else:
            self.structural_chunker = StructuralChunker()
        
        self.semantic_chunker = semantic_chunker
        
        # Initialize recursive chunker with config parameters
        if config:
            self.recursive_chunker = RecursiveCharacterTextSplitter(
                chunk_size=config.max_chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            # Fallback to default settings
            recursive_chunk_size = settings.MAX_CHUNK_SIZE // 2
            self.recursive_chunker = RecursiveCharacterTextSplitter(
                chunk_size=recursive_chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
        # Log configuration
        logger.info(f"HybridDocumentChunker initialized with:")
        logger.info(f"  - Default strategy: {strategy}")
        logger.info(f"  - Config provided: {config is not None}")
        logger.info(f"  - Semantic chunker available: {semantic_chunker is not None}")
        if config:
            logger.info(f"  - Min chunk size: {config.min_chunk_size}")
            logger.info(f"  - Max chunk size: {config.max_chunk_size}")
            logger.info(f"  - Chunk overlap: {config.chunk_overlap}")
            logger.info(f"  - Preserve lists: {config.preserve_lists}")
            logger.info(f"  - Preserve headings: {config.preserve_headings}")
            logger.info(f"  - Preserve quotes: {config.preserve_quotes}")
        else:
            recursive_chunk_size = settings.MAX_CHUNK_SIZE // 2
            logger.info(f"  - Using default settings - recursive chunk size: {recursive_chunk_size}")
            logger.info(f"  - Max chunk size setting: {settings.MAX_CHUNK_SIZE}")
            if hasattr(settings, 'MIN_CHUNK_SIZE'):
                logger.info(f"  - Min chunk size setting: {settings.MIN_CHUNK_SIZE}")
    
    def _compute_chunk_quality(self, chunk_text: str, doc_type: DocumentType) -> ChunkQualityMetrics:
        """Compute quality metrics for a chunk based on document type"""
        words = chunk_text.split()
        sentences = [s for s in chunk_text.split('.') if s.strip()]
        
        # Adjust quality scoring based on document type
        if doc_type == DocumentType.STRUCTURED:
            # For structured docs, check if structure is preserved
            has_structure = bool(re.search(r'^\d+\.\s+|^#+\s+|^[-*+]\s+', chunk_text, re.MULTILINE))
            structure_bonus = 0.2 if has_structure else 0.0
        else:
            structure_bonus = 0.0
        
        base_score = min(1.0, len(words) / 100)  # Basic length-based score
        final_score = base_score + structure_bonus
        
        return ChunkQualityMetrics(
            token_count=len(words),
            sentence_count=len(sentences),
            avg_sentence_length=len(words) / max(len(sentences), 1),
            semantic_score=final_score
        )
    
    async def process_document(
        self,
        text: str,
        transform_id: str,
        strategy_override: Optional[ChunkingStrategy] = None
    ) -> Tuple[ChunkingResult, List[ChunkMetadata]]:
        """Process document with adaptive chunking strategy"""
        try:
            logger.info(f"Starting document processing for transform_id: {transform_id}")
            logger.info(f"Document length: {len(text)} characters, {len(text.split())} words")
            
            # Classify document type
            doc_type = self.classifier.classify_document(text)
            
            # Choose strategy
            strategy = strategy_override or self.strategy
            logger.info(f"Chunking strategy: {strategy} " +
                       (f"(overridden from default: {self.strategy})" if strategy_override else "(default)"))
            print(f"Chunking strategy: {strategy} " +
                       (f"(overridden from default: {self.strategy})" if strategy_override else "(default)"))
            
            # Apply chunking based on strategy and document type
            if strategy == ChunkingStrategy.HYBRID:
                logger.info(f"Applying hybrid chunking for document type: {doc_type}")
                print(f"Applying hybrid chunking for document type: {doc_type}")
                chunks = await self._hybrid_chunk(text, doc_type)
            elif strategy == ChunkingStrategy.STRUCTURAL:
                logger.info("Applying structural chunking strategy")
                print("Applying structural chunking strategy")
                chunks = self.structural_chunker.chunk_text(text)
            elif strategy == ChunkingStrategy.SEMANTIC and self.semantic_chunker:
                logger.info("Applying semantic chunking strategy")
                print("Applying semantic chunking strategy")
                documents = self.semantic_chunker.create_documents([text])
                chunks = [doc.page_content for doc in documents]
            else:
                # Fallback to recursive
                fallback_reason = "semantic chunker not available" if strategy == ChunkingStrategy.SEMANTIC else "fallback"
                logger.info(f"Applying recursive chunking strategy ({fallback_reason})")
                print(f"Applying recursive chunking strategy ({fallback_reason})")
                documents = self.recursive_chunker.create_documents([text])
                chunks = [doc.page_content for doc in documents]
            
            logger.info(f"Generated {len(chunks)} chunks using {strategy} strategy")
            print(f"Generated {len(chunks)} chunks using {strategy} strategy")

            # Log chunk statistics
            if chunks:
                chunk_sizes = [len(chunk) for chunk in chunks]
                logger.info(f"Chunk size statistics:")
                logger.info(f"  - Min size: {min(chunk_sizes)} chars")
                logger.info(f"  - Max size: {max(chunk_sizes)} chars")
                logger.info(f"  - Average size: {sum(chunk_sizes) / len(chunk_sizes):.1f} chars")
                logger.info(f"  - Total content: {sum(chunk_sizes)} chars")
                logger.info(f"  - Coverage: {(sum(chunk_sizes) / len(text) * 100):.1f}%")
                print(f"Chunk size statistics:")
                print(f"  - Min size: {min(chunk_sizes)} chars")
                print(f"  - Max size: {max(chunk_sizes)} chars")
                print(f"  - Average size: {sum(chunk_sizes) / len(chunk_sizes):.1f} chars")
                print(f"  - Total content: {sum(chunk_sizes)} chars")
                print(f"  - Coverage: {(sum(chunk_sizes) / len(text) * 100):.1f}%")

            # Process chunks into metadata
            chunk_texts = []
            chunk_metadata = []
            
            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                # Find chunk position in original text
                start_idx = text.find(chunk_text[:50])  # Use first 50 chars for finding
                end_idx = start_idx + len(chunk_text) if start_idx != -1 else len(chunk_text)
                
                # Compute quality metrics
                quality = self._compute_chunk_quality(chunk_text, doc_type)
                
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
            
            # Create result
            total_tokens = sum(len(c.split()) for c in chunk_texts)
            result = ChunkingResult(
                transform_id=transform_id,
                chunks=chunk_texts,
                num_chunks=len(chunk_texts),
                total_tokens=total_tokens,
                semantic_processing_time=0.0,  # Will be set by specific chunker
                chunk_processing_time=0.0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            logger.info(f"Document processing completed successfully:")
            logger.info(f"  - Final chunks: {len(chunk_texts)}")
            logger.info(f"  - Total tokens: {total_tokens}")
            logger.info(f"  - Transform ID: {transform_id}")
            
            return result, chunk_metadata
            
        except Exception as e:
            logger.error(f"Failed to process document with transform_id: {transform_id}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Strategy used: {strategy_override or self.strategy}")
            logger.error(f"Document length: {len(text)} characters")
            traceback.print_exc()
            raise
    
    async def _hybrid_chunk(self, text: str, doc_type: DocumentType) -> List[str]:
        """Apply hybrid chunking strategy based on document type"""
        
        logger.info(f"Hybrid chunking decision for document type: {doc_type}")
        
        if doc_type == DocumentType.STRUCTURED:
            # Use structural chunking for structured documents
            logger.info("→ Selected: Structural chunking (preserves document structure)")
            return self.structural_chunker.chunk_text(text)
        
        elif doc_type == DocumentType.NARRATIVE and self.semantic_chunker:
            # Use semantic chunking for narrative documents
            logger.info("→ Selected: Semantic chunking (context-aware for narrative)")
            if len(text) <= getattr(settings, 'MIN_CHUNK_SIZE', 500):
                logger.info("→ Document too small for chunking, returning as single chunk")
                return [text]
            documents = self.semantic_chunker.create_documents([text])
            chunks = [doc.page_content for doc in documents]
            logger.info(f"→ Semantic chunking produced {len(chunks)} chunks")
            return chunks
        
        elif doc_type == DocumentType.NARRATIVE and not self.semantic_chunker:
            # Fallback for narrative when semantic chunker not available
            logger.info("→ Selected: Recursive chunking (semantic chunker not available for narrative)")
            documents = self.recursive_chunker.create_documents([text])
            return [doc.page_content for doc in documents]
        
        else:
            # Use recursive chunking for mixed/technical documents
            logger.info(f"→ Selected: Recursive chunking (optimal for {doc_type} documents)")
            documents = self.recursive_chunker.create_documents([text])
            chunks = [doc.page_content for doc in documents]
            logger.info(f"→ Recursive chunking produced {len(chunks)} chunks")
            return chunks

class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass