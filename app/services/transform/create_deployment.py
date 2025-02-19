from prefect.deployments import Deployment
from prefect.filesystems import LocalFileSystem
from flows import document_transformation_flow
from app.config import settings
import asyncio

async def create_deployment():
    """Create a deployment for the document transformation flow"""
    # Create local storage
    storage = LocalFileSystem(
        basepath=str(settings.UPLOAD_DIR),
        persist_local=True
    )
    
    # Create deployment
    deployment = await Deployment.build_from_flow(
        flow=document_transformation_flow,
        name="document-transformation",
        work_pool_name=settings.PREFECT_WORKPOOL_TRANSFORM,
        storage=storage
    )
    
    # Apply deployment
    deployment_id = await deployment.apply()
    print(f"Created deployment with ID: {deployment_id}")

if __name__ == "__main__":
    asyncio.run(create_deployment())
