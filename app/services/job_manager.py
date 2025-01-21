from typing import Dict, Optional, Callable, Any
from datetime import datetime
import asyncio
from app.schemas.job import JobStatus
from app.utils.logger import logger
from fastapi import FastAPI
from asyncio import Lock

class JobManager:
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
        """Get status of a job"""
        return self.app.state.jobs.get(job_id)
    
    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed with error message"""
        async with self.app.state.jobs_lock:
            if job_id not in self.app.state.jobs:
                # Create a failed job if it doesn't exist
                self.app.state.jobs[job_id] = JobStatus(
                    status='failed',
                    progress=0.0,
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error=error
                )
            else:
                # Update existing job to failed state
                job = self.app.state.jobs[job_id]
                job.status = 'failed'
                job.error = error
                job.completed_at = datetime.now()
    
    async def run_async_job(self, job_id: str, 
                           func: Callable, *args, **kwargs) -> None:
        """Run a job asynchronously"""
        try:
            # Run the actual job
            result = await func(*args, **kwargs)
            
            # Update status to completed
            async with self.app.state.jobs_lock:
                job = self.app.state.jobs[job_id]
                job.status = 'completed'
                job.progress = 100.0
                job.completed_at = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            await self.fail_job(job_id, str(e))
            raise
    
    async def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress"""
        async with self.app.state.jobs_lock:
            if job_id in self.app.state.jobs:
                job = self.app.state.jobs[job_id]
                # Only update progress if job hasn't failed
                if job.status != 'failed':
                    job.progress = min(max(progress, 0.0), 100.0)

# Global instance
_job_manager: Optional[JobManager] = None

def get_job_manager(app: FastAPI) -> JobManager:
    """Get or create the JobManager instance"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager(app)
    return _job_manager
