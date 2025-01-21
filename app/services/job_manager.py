from typing import Any, Optional, Callable, Awaitable
from datetime import datetime
from app.utils.logger import logger
from app.schemas.job import JobStatus
from fastapi import FastAPI
from asyncio import Lock

class JobManager:
    """Manages async job status and execution"""
    
    def __init__(self, app: FastAPI):
        if not hasattr(app.state, "jobs"):
            app.state.jobs = {}
        if not hasattr(app.state, "jobs_lock"):
            app.state.jobs_lock = Lock()
        self.app = app
    
    async def create_job(self, job_id: str) -> None:
        """Create a new job with initial status"""
        async with self.app.state.jobs_lock:
            self.app.state.jobs[job_id] = JobStatus(
                status='processing',
                progress=0.0,
                started_at=datetime.now()
            )
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current status of a job"""
        return self.app.state.jobs.get(job_id)
        
    async def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress"""
        async with self.app.state.jobs_lock:
            if job_id in self.app.state.jobs:
                job = self.app.state.jobs[job_id]
                # Only update progress if job hasn't failed
                if job.status != 'failed':
                    job.progress = min(max(progress, 0.0), 100.0)
            
    async def complete_job(self, job_id: str) -> None:
        """Mark job as completed"""
        async with self.app.state.jobs_lock:
            if job_id in self.app.state.jobs:
                job = self.app.state.jobs[job_id]
                job.status = 'completed'
                job.progress = 100.0
                job.completed_at = datetime.now()
            
    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with error message"""
        async with self.app.state.jobs_lock:
            if job_id in self.app.state.jobs:
                job = self.app.state.jobs[job_id]
                job.status = 'failed'
                job.error = error
                job.completed_at = datetime.now()
            
    async def run_async_job(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Run an async job and manage its status
        
        Args:
            func: Async function to run
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        The first argument to func should be the job_id
        """
        job_id = args[0] if args else kwargs.get('transform_id')
        if not job_id:
            raise ValueError("Job ID not provided")
            
        try:
            await func(*args, **kwargs)
            await self.complete_job(job_id)
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            await self.fail_job(job_id, str(e))
            raise

# Global job manager instance
_job_manager: Optional[JobManager] = None

def get_job_manager(app: FastAPI) -> JobManager:
    """Get or create global job manager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager(app)
    return _job_manager
