"""
Schemas for healthcare domain APIs
"""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, computed_field
from datetime import date


class Patient(BaseModel):
    """Patient schema"""
    id: str
    firstName: str
    lastName: str
    dateOfBirth: Optional[str] = None
    diagnosis: Optional[str] = None
    
    @computed_field
    def name(self) -> str:
        return f"{self.firstName} {self.lastName}"


class PatientInfo(BaseModel):
    """Extended patient information schema"""
    id: str
    firstName: str
    lastName: str
    age: int
    diagnosis: Optional[str] = None
    
    @computed_field
    def name(self) -> str:
        return f"{self.firstName} {self.lastName}"


class JourneyEvent(BaseModel):
    """Patient journey event schema"""
    id: str
    date: date
    type: str
    label: str
    details: Optional[str] = None
    nodeType: Optional[str] = None


class MedicalReport(BaseModel):
    """Medical report schema"""
    date: date
    type: str
    findings: str
    abnormalities: Optional[str] = None
    status: Optional[str] = None


class TreatmentOutcome(BaseModel):
    """Treatment outcome schema"""
    treatment: str
    outcome: str
    complication: Optional[str] = None


class PatientJourney(BaseModel):
    """Patient journey schema"""
    patientInfo: PatientInfo
    journeyEvents: List[JourneyEvent]
    medicalReports: List[MedicalReport]
    treatmentOutcomes: List[TreatmentOutcome]


class PatientListResponse(BaseModel):
    """Response schema for patient list"""
    patients: List[Patient]


class LaboratoryComponent(BaseModel):
    """Laboratory test component schema"""
    id: str
    name: str
    value: str
    units: str
    referenceRange: str


class LaboratoryResult(BaseModel):
    """Laboratory result schema"""
    id: str
    date: str
    time: str
    reportDate: str
    facility: str
    components: List[LaboratoryComponent]


class LaboratoryResultsResponse(BaseModel):
    """Response schema for laboratory results endpoint"""
    laboratoryResults: List[LaboratoryResult]
