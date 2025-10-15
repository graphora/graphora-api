"""
Healthcare domain-specific API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from app.domain.healthcare.schemas import (
    PatientListResponse,
    PatientJourney,
    LaboratoryResultsResponse,
)
from app.domain.healthcare.service import HealthcareService
from app.services.storage.neo4j import Neo4jStorage
from app.services.user_db_service import UserDatabaseService
from app.utils.logger import logger
from app.auth import get_current_user_id
import traceback

router = APIRouter(prefix="/api/v1/domain/healthcare", tags=["Healthcare"])


async def get_healthcare_service(user_id: str = Depends(get_current_user_id)):
    """
    Dependency to get a healthcare service instance with user-specific database
    """
    try:
        # Get user's production database configuration (domain apps use production data)
        user_config = await UserDatabaseService.get_user_config(user_id)

        neo4j_storage = Neo4jStorage(
            uri=user_config.prodDb.uri,
            username=user_config.prodDb.username,
            password=user_config.prodDb.password,
            database="neo4j",  # Default database name
        )
        service = HealthcareService(neo4j_storage=neo4j_storage)
        # Note: We're not closing the connection here to avoid async issues
        # The connection will be closed when the application shuts down
        return service
    except ValueError as e:
        # User configuration not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Other errors (database connection, etc.)
        logger.error(f"Error creating healthcare service for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to initialize healthcare service"
        )


@router.get(
    "/patients",
    response_model=PatientListResponse,
    description="Get a list of patients",
)
async def get_patients(
    limit: Optional[int] = 100,
    healthcare_service: HealthcareService = Depends(get_healthcare_service),
) -> PatientListResponse:
    """
    Get a list of patients with basic information

    Parameters:
    - limit: Maximum number of patients to return (default: 100)

    Returns:
    - PatientListResponse containing list of patients with their basic information
    """
    try:
        # Validate inputs
        if limit < 0:
            raise HTTPException(status_code=400, detail="Limit must be non-negative")

        if limit > 1000:
            raise HTTPException(
                status_code=400, detail="Maximum limit is 1000 patients"
            )

        # Get patients data
        patients = await healthcare_service.get_patients(limit=limit)

        return PatientListResponse(patients=patients)

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error retrieving patients data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving patients data: {str(e)}"
        )


@router.get("/patients/{patient_id}/journey", response_model=PatientJourney)
async def get_patient_journey(
    patient_id: str,
    healthcare_service: HealthcareService = Depends(get_healthcare_service),
):
    """
    Get a patient's journey including events, reports, and treatment outcomes

    Args:
        patient_id: ID of the patient

    Returns:
        PatientJourney object with all journey information
    """
    try:
        patient_journey = await healthcare_service.get_patient_journey(
            patient_id=patient_id
        )
        if patient_journey is None or patient_journey.patientInfo is None:
            raise HTTPException(
                status_code=404, detail=f"Patient with ID {patient_id} not found"
            )
        return patient_journey
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving patient journey: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving the patient journey",
        )


@router.get(
    "/patients/{patient_id}/laboratory-results",
    response_model=LaboratoryResultsResponse,
)
async def get_patient_laboratory_results(
    patient_id: str,
    healthcare_service: HealthcareService = Depends(get_healthcare_service),
):
    """
    Get laboratory results for a specific patient

    Args:
        patient_id: ID of the patient

    Returns:
        LaboratoryResultsResponse containing a list of laboratory results with components
    """
    try:
        lab_results = await healthcare_service.get_patient_laboratory_results(
            patient_id=patient_id
        )
        return LaboratoryResultsResponse(laboratoryResults=lab_results)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving laboratory results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving laboratory results",
        )
