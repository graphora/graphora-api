import hashlib
from datetime import datetime, timezone
from typing import List, Tuple
import psutil
import numpy as np
from sentence_transformers import SentenceTransformer
import multiprocessing as mp
from queue import Queue
import threading
import torch
from app.utils.logger import logger
from app.services.chunking.models import (
    ChunkMetadata,
    ChunkingResult,
    ChunkProcessingMetrics,
    ChunkQualityMetrics
)

class SemanticChunker:
    """Semantic chunker that uses embeddings to find natural breakpoints"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        overlap_size: int = 50,
        similarity_threshold: float = 0.7,
        batch_size: int = 32,
        device: str = "cpu"  # Force CPU for now
    ):
        """Initialize semantic chunker"""
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Initialize model
        self.model = SentenceTransformer(model_name, device=device)
        
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        # Move tensors to CPU if needed
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.cpu().numpy()
        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.cpu().numpy()
            
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_breakpoints(self, text: str) -> List[int]:
        """Find natural breakpoints in text using semantic similarity"""
        # Split into sentences (simple for now)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return []
            
        # Get embeddings
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True  # Force numpy conversion
        )
        
        # Find breakpoints based on similarity
        breakpoints = []
        current_length = 0
        last_breakpoint = 0
        
        for i in range(len(sentences) - 1):
            current_length += len(sentences[i])
            
            # Check if we should consider a breakpoint
            if current_length >= self.min_chunk_size:
                similarity = self._compute_similarity(embeddings[i], embeddings[i + 1])
                
                if similarity < self.similarity_threshold or current_length >= self.max_chunk_size:
                    breakpoints.append(i + 1)
                    current_length = 0
                    last_breakpoint = i + 1
        
        return breakpoints
        
    def _chunk_text(self, text: str, breakpoints: List[int]) -> List[Tuple[str, int, int]]:
        """
        Chunk text at breakpoints
        
        Returns list of (chunk_text, start_pos, end_pos)
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return []
            
        # Convert breakpoints to character positions
        char_breakpoints = [0]  # Start with 0
        pos = 0
        for i, sent in enumerate(sentences):
            pos += len(sent) + 1  # +1 for the period
            if i + 1 in breakpoints:
                char_breakpoints.append(pos)
        char_breakpoints.append(len(text))  # End with text length
        
        # Create chunks with overlap
        chunks = []
        for i in range(len(char_breakpoints) - 1):
            start = max(0, char_breakpoints[i] - self.overlap_size if i > 0 else char_breakpoints[i])
            end = min(len(text), char_breakpoints[i + 1] + self.overlap_size if i < len(char_breakpoints) - 2 else char_breakpoints[i + 1])
            chunk_text = text[start:end].strip()
            chunks.append((chunk_text, start, end))
            
        return chunks
        
    def _compute_chunk_quality(self, chunk: str) -> ChunkQualityMetrics:
        """Compute quality metrics for a chunk"""
        # Simple quality metrics for now
        return ChunkQualityMetrics(
            coherence_score=0.8,  # Placeholder
            relevance_score=0.8,  # Placeholder
            size_score=min(1.0, max(0.0, len(chunk) / self.max_chunk_size))
        )
        
    async def process_document(
        self,
        text: str,
        transform_id: str
    ) -> Tuple[ChunkingResult, List[ChunkMetadata]]:
        """
        Process a document and return chunks with metadata
        
        Args:
            text: Document text to chunk
            transform_id: ID of the transformation
            
        Returns:
            Tuple of (ChunkingResult, List[ChunkMetadata])
        """
        try:
            # Find breakpoints
            semantic_start = datetime.now(timezone.utc)
            breakpoints = self.find_breakpoints(text)
            semantic_time = datetime.now(timezone.utc) - semantic_start
            
            # Create chunks
            chunk_start = datetime.now(timezone.utc)
            chunks = self._chunk_text(text, breakpoints)
            chunk_time = datetime.now(timezone.utc) - chunk_start
            
            # Create metadata for each chunk
            chunk_metadata = []
            chunk_texts = []
            for i, (chunk_text, start, end) in enumerate(chunks):
                # Compute quality metrics
                quality = self._compute_chunk_quality(chunk_text)
                
                # Create hash
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                
                metadata = ChunkMetadata(
                    transform_id=transform_id,
                    chunk_id=f"{transform_id}_chunk_{i}",
                    chunk_index=i,
                    chunk_hash=chunk_hash,
                    start_position=start,
                    end_position=end,
                    chunk_size=len(chunk_text),
                    quality_metrics=quality,
                    processing_timestamp=datetime.now(timezone.utc)
                )
                chunk_texts.append(chunk_text)
                chunk_metadata.append(metadata)
            
            # Create result
            result = ChunkingResult(
                transform_id=transform_id,
                chunks=chunk_texts,
                num_chunks=len(chunks),
                total_tokens=sum(len(c[0].split()) for c in chunks),
                semantic_processing_time=semantic_time.total_seconds(),
                chunk_processing_time=chunk_time.total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            return result, chunk_metadata
            
        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            raise
            
    def __del__(self):
        """Cleanup when chunker is destroyed"""
        # Nothing to clean up for now
        pass

class ChunkWorker:
    """Worker for parallel chunk processing"""
    
    def __init__(
        self,
        worker_id: str,
        model_name: str,
        semantic_threshold: float,
        device: str = "cpu"  # Force CPU for now
    ):
        self.worker_id = worker_id
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.device = device
        self.process = psutil.Process()
        
        # Initialize model
        self.model = SentenceTransformer(model_name, device=device)
    
    def process_chunk(
        self,
        text: str,
        start_pos: int
    ) -> Tuple[str, ChunkMetadata]:
        """Process a single chunk of text"""
        start_time = datetime.now(timezone.utc)
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Compute embeddings for sentences in chunk
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Compute quality metrics
        quality_metrics = ChunkQuality(
            coherence_score=0.8,  # Placeholder
            relevance_score=0.8,  # Placeholder
            size_score=min(1.0, max(0.0, len(text) / 1000))
        )
        
        # Create chunk metadata
        end_time = datetime.now(timezone.utc)
        processing_time = (end_time - start_time).total_seconds() * 1000
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        metadata = ChunkMetadata(
            chunk_id=f"chunk_{start_pos}_{len(text)}",
            start_pos=start_pos,
            end_pos=start_pos + len(text),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            quality_metrics=quality_metrics,
            processing_metrics=ChunkProcessingMetrics(
                worker_id=self.worker_id,
                start_time=start_time,
                end_time=end_time,
                processing_time_ms=processing_time,
                memory_used_mb=current_memory - initial_memory,
                cpu_usage_percent=self.process.cpu_percent(),
                queue_wait_time_ms=0.0  # Set by worker pool
            ),
            tokens=len(text.split()),
            sentences=len(sentences),
            language_detected="en"  # TODO: Add language detection
        )
        
        return text, metadata
    
    def __del__(self):
        """Cleanup when worker is destroyed"""
        # Nothing to clean up for now
        pass

class ChunkWorkerPool:
    """Pool of workers for parallel chunk processing"""
    
    def __init__(
        self,
        num_workers: int,
        model_name: str,
        semantic_threshold: float,
        device: str = "cpu"  # Force CPU for now
    ):
        self.num_workers = num_workers
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.device = device
        self.queue = Queue()
        self.results = {}
        self.workers = []
        self.worker_threads = []
        
        # Initialize workers
        for i in range(num_workers):
            worker = ChunkWorker(
                f"worker_{i}",
                model_name,
                semantic_threshold,
                device=device
            )
            self.workers.append(worker)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker,),
                daemon=True
            )
            self.worker_threads.append(thread)
            thread.start()
    
    def _worker_loop(self, worker: ChunkWorker):
        """Main worker loop"""
        while True:
            try:
                task_id, text, start_pos, queue_start = self.queue.get()
                if task_id is None:  # Shutdown signal
                    break
                
                # Process chunk
                chunk, metadata = worker.process_chunk(text, start_pos)
                
                # Update queue wait time
                metadata.processing_metrics.queue_wait_time_ms = (
                    datetime.now(timezone.utc) - queue_start
                ).total_seconds() * 1000
                
                # Store result
                self.results[task_id] = (chunk, metadata)
                
            except Exception as e:
                self.results[task_id] = e
            finally:
                self.queue.task_done()
    
    def process_chunks(
        self,
        chunks: List[Tuple[str, int]]
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Process multiple chunks in parallel"""
        # Submit all chunks to queue
        for i, (text, start_pos) in enumerate(chunks):
            self.queue.put((
                i,
                text,
                start_pos,
                datetime.now(timezone.utc)
            ))
        
        # Wait for all chunks to be processed
        self.queue.join()
        
        # Check for errors and collect results
        results = []
        for i in range(len(chunks)):
            result = self.results.pop(i)
            if isinstance(result, Exception):
                raise ChunkingError(f"Worker failed: {str(result)}")
            results.append(result)
        
        return results
    
    def shutdown(self):
        """Shutdown worker pool"""
        for _ in self.workers:
            self.queue.put((None, None, None, None))
        for thread in self.worker_threads:
            thread.join()

class HybridChunker:
    """Hybrid document chunker with parallel processing"""
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        semantic_threshold: float = 0.7,
        model_name: str = "all-MiniLM-L6-v2",
        num_workers: int = max(1, mp.cpu_count() - 1),
        device: str = "cpu"  # Force CPU for now
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.semantic_threshold = semantic_threshold
        self.model_name = model_name
        self.num_workers = num_workers
        self.device = device
        
        # Initialize components
        self.semantic_chunker = SemanticChunker(
            model_name=model_name,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            similarity_threshold=semantic_threshold,
            device=device
        )
        self.worker_pool = ChunkWorkerPool(
            num_workers,
            model_name,
            semantic_threshold,
            device=device
        )
    
    async def process_document(
        self,
        text: str,
        transform_id: str
    ) -> ChunkingResult:
        """Process document using parallel chunking"""
        if not text:
            raise ChunkingError("Empty document")
        
        start_time = datetime.now(timezone.utc)
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Find semantic breakpoints
            semantic_start = datetime.now(timezone.utc)
            breakpoints = self.semantic_chunker.find_breakpoints(text)
            embedding_time = (
                datetime.now(timezone.utc) - semantic_start
            ).total_seconds() * 1000
            
            # Create initial chunks
            chunks_to_process = []
            current_pos = 0
            
            for breakpoint in breakpoints + [len(text)]:
                chunk_text = text[current_pos:breakpoint].strip()
                if not chunk_text:
                    continue
                
                # Split if needed
                for sub_chunk, _ in self._split_large_chunk(chunk_text):
                    chunks_to_process.append((
                        sub_chunk,
                        current_pos
                    ))
                
                current_pos = breakpoint
            
            # Process chunks in parallel
            parallel_start = datetime.now(timezone.utc)
            processed_chunks = self.worker_pool.process_chunks(chunks_to_process)
            parallel_time = (
                datetime.now(timezone.utc) - parallel_start
            ).total_seconds() * 1000
            
            # Collect results
            chunks = []
            metadata = []
            worker_metrics = []
            
            for chunk, chunk_metadata in processed_chunks:
                chunks.append(chunk)
                metadata.append(chunk_metadata)
                if chunk_metadata.processing_metrics:
                    worker_metrics.append(chunk_metadata.processing_metrics)
            
            # Calculate overall metrics
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            metrics = ChunkingMetrics(
                total_processing_time_ms=(
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000,
                embedding_time_ms=embedding_time,
                chunk_refinement_time_ms=0.0,  # Not used with parallel processing
                embedding_model_latency_ms=sum(
                    w.processing_time_ms for w in worker_metrics
                ) / len(worker_metrics),
                parallel_processing_time_ms=parallel_time,
                worker_count=self.num_workers,
                queue_depth=len(chunks_to_process),
                
                original_doc_size=len(text),
                total_chunks=len(chunks),
                avg_chunk_size=sum(len(c) for c in chunks) / len(chunks),
                max_chunk_size=max(len(c) for c in chunks),
                min_chunk_size=min(len(c) for c in chunks),
                total_tokens=sum(m.tokens for m in metadata if m.tokens),
                
                semantic_splits=len(breakpoints),
                forced_splits=sum(1 for m in metadata if m.is_forced_split),
                merged_chunks=sum(1 for m in metadata if m.is_merged),
                avg_semantic_coherence=sum(
                    m.quality.coherence_score for m in metadata
                ) / len(metadata),
                avg_boundary_smoothness=sum(
                    m.quality.relevance_score for m in metadata
                ) / len(metadata),
                avg_content_density=sum(
                    m.quality.size_score for m in metadata
                ) / len(metadata),
                avg_readability_score=0.0,  # Not used
                topic_consistency_score=0.0,  # Not used
                
                peak_memory_mb=current_memory - initial_memory,
                avg_cpu_usage_percent=sum(
                    w.cpu_usage_percent for w in worker_metrics
                ) / len(worker_metrics),
                total_embedding_calls=len(metadata) * 2,  # Initial + refinement
                cache_hit_rate=0.0  # TODO: Implement embedding cache
            )
            
            return ChunkingResult(
                transform_id=transform_id,
                chunks=chunks,
                chunk_metadata=metadata,
                metrics=metrics.__dict__,
                timestamp=datetime.now(timezone.utc).isoformat(),
                worker_metrics=worker_metrics
            )
            
        except Exception as e:
            raise ChunkingError(f"Failed to process document: {str(e)}")
        
    def __del__(self):
        """Cleanup worker pool"""
        if hasattr(self, 'worker_pool'):
            self.worker_pool.shutdown()

    def _split_large_chunk(self, chunk: str) -> List[Tuple[str, bool]]:
        """
        Split oversized chunk into smaller ones
        Returns: List of (chunk, is_forced_split) tuples
        """
        if len(chunk) <= self.max_chunk_size:
            return [(chunk, False)]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(chunk):
            # Try to find a sentence boundary
            end_pos = current_pos + self.max_chunk_size
            if end_pos > len(chunk):
                end_pos = len(chunk)
            
            # Look for sentence boundary
            boundary = chunk.rfind('.', current_pos, end_pos)
            if boundary == -1 or boundary <= current_pos + self.min_chunk_size:
                # No good boundary, force split
                boundary = end_pos
                is_forced = True
            else:
                boundary += 1  # Include the period
                is_forced = False
            
            chunks.append((chunk[current_pos:boundary].strip(), is_forced))
            current_pos = boundary
        
        return chunks

class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass
