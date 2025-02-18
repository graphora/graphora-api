import hashlib
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict
import psutil
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading
import textstat

from app.services.chunking.models import (
    ChunkingMetrics,
    ChunkMetadata,
    ChunkingResult,
    ChunkQualityMetrics,
    ChunkProcessingMetrics
)
from app.config import settings

class SemanticChunker:
    """Handles semantic-based text chunking using embeddings"""
    
    def __init__(self, model: SentenceTransformer, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold
    
    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for a list of sentences"""
        return self.model.encode(sentences, convert_to_tensor=True)
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def find_breakpoints(self, text: str) -> List[int]:
        """Find semantic breakpoints in text"""
        # Split into sentences (simple for now, can be improved)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return []
        
        # Compute embeddings
        embeddings = self._compute_embeddings(sentences)
        breakpoints = []
        
        # Find points where semantic similarity drops
        for i in range(len(embeddings) - 1):
            similarity = self._compute_similarity(
                embeddings[i],
                embeddings[i + 1]
            )
            if similarity < self.threshold:
                # Find the actual character position
                pos = len('.'.join(sentences[:i+1])) + 1
                breakpoints.append(pos)
        
        return breakpoints

class ChunkWorker:
    """Worker for parallel chunk processing"""
    
    def __init__(
        self,
        worker_id: str,
        model: SentenceTransformer,
        semantic_threshold: float
    ):
        self.worker_id = worker_id
        self.model = model
        self.semantic_threshold = semantic_threshold
        self.process = psutil.Process()
    
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
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Compute quality metrics
        quality_metrics = ChunkQualityMetrics(
            semantic_coherence=self._compute_semantic_coherence(embeddings),
            boundary_smoothness=self._compute_boundary_smoothness(text),
            content_density=self._compute_content_density(text),
            readability_score=textstat.flesch_reading_ease(text),
            topic_consistency=self._compute_topic_consistency(embeddings),
            formatting_quality=self._compute_formatting_quality(text)
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
    
    def _compute_semantic_coherence(self, embeddings: np.ndarray) -> float:
        """Compute average semantic similarity between sentences"""
        if len(embeddings) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(float(sim))
        
        return sum(similarities) / len(similarities)
    
    def _compute_boundary_smoothness(self, text: str) -> float:
        """Compute how well chunk boundaries align with natural breaks"""
        if not text:
            return 0.0
        
        # Check if chunk starts/ends with complete sentences
        starts_with_capital = text[0].isupper() if text else False
        ends_with_period = text.strip().endswith('.') if text else False
        
        return (starts_with_capital + ends_with_period) / 2
    
    def _compute_content_density(self, text: str) -> float:
        """Compute ratio of meaningful content to total size"""
        if not text:
            return 0.0
        
        # Remove extra whitespace and compute ratio
        cleaned_text = ' '.join(text.split())
        return len(cleaned_text) / len(text)
    
    def _compute_topic_consistency(self, embeddings: np.ndarray) -> float:
        """Compute how well the chunk maintains a single topic"""
        if len(embeddings) <= 1:
            return 1.0
        
        # Compare all sentences to the mean embedding
        mean_embedding = np.mean(embeddings, axis=0)
        similarities = []
        
        for emb in embeddings:
            sim = np.dot(emb, mean_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(mean_embedding)
            )
            similarities.append(float(sim))
        
        return sum(similarities) / len(similarities)
    
    def _compute_formatting_quality(self, text: str) -> float:
        """Compute quality of text formatting"""
        if not text:
            return 0.0
        
        scores = []
        
        # Check for consistent line endings
        lines = text.split('\n')
        if lines:
            consistent_endings = sum(l.strip().endswith('.') for l in lines) / len(lines)
            scores.append(consistent_endings)
        
        # Check for consistent capitalization
        words = text.split()
        if words:
            consistent_caps = sum(w[0].isupper() for w in words if w) / len(words)
            scores.append(consistent_caps)
        
        # Check for balanced parentheses/brackets
        balanced = all(
            text.count(open_char) == text.count(close_char)
            for open_char, close_char in [('(', ')'), ('[', ']'), ('{', '}')]
        )
        scores.append(1.0 if balanced else 0.0)
        
        return sum(scores) / len(scores) if scores else 0.0

class ChunkWorkerPool:
    """Pool of workers for parallel chunk processing"""
    
    def __init__(
        self,
        num_workers: int,
        model_name: str,
        semantic_threshold: float
    ):
        self.num_workers = num_workers
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.queue = Queue()
        self.results = {}
        self.workers = []
        self.worker_threads = []
        
        # Initialize workers
        for i in range(num_workers):
            worker = ChunkWorker(
                f"worker_{i}",
                SentenceTransformer(model_name),
                semantic_threshold
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
        max_chunk_size: int = settings.MAX_CHUNK_SIZE,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
        semantic_threshold: float = settings.SEMANTIC_THRESHOLD,
        model_name: str = settings.EMBEDDING_MODEL,
        num_workers: int = max(1, mp.cpu_count() - 1)
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.semantic_threshold = semantic_threshold
        self.model_name = model_name
        self.num_workers = num_workers
        
        # Initialize components
        self.semantic_chunker = SemanticChunker(
            SentenceTransformer(model_name),
            threshold=semantic_threshold
        )
        self.worker_pool = ChunkWorkerPool(
            num_workers,
            model_name,
            semantic_threshold
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
                    m.quality_metrics.semantic_coherence for m in metadata
                ) / len(metadata),
                avg_boundary_smoothness=sum(
                    m.quality_metrics.boundary_smoothness for m in metadata
                ) / len(metadata),
                avg_content_density=sum(
                    m.quality_metrics.content_density for m in metadata
                ) / len(metadata),
                avg_readability_score=sum(
                    m.quality_metrics.readability_score for m in metadata
                ) / len(metadata),
                topic_consistency_score=sum(
                    m.quality_metrics.topic_consistency for m in metadata
                ) / len(metadata),
                
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
