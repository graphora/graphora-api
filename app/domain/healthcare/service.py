"""
Healthcare domain-specific services for querying patient data from Neo4j
"""
import traceback
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, date
from app.services.storage.neo4j import Neo4jStorage
from app.domain.healthcare.schemas import Patient, PatientInfo, JourneyEvent, MedicalReport, TreatmentOutcome, PatientJourney, LaboratoryResult, LaboratoryComponent

# Configure logger
logger = logging.getLogger(__name__)

class HealthcareService:
    """Service for healthcare domain-specific operations"""
    
    def __init__(self, neo4j_storage: Neo4jStorage):
        """Initialize with Neo4j storage instance"""
        self.storage = neo4j_storage
        
    async def get_patients(self, limit: int = 100) -> List[Patient]:
        """
        Get list of patients from the database
        
        Args:
            limit: Maximum number of patients to return
            
        Returns:
            List of Patient objects
        """
        try:
            async with self.storage._get_session() as session:
                # Query to get patients with their basic information
                query = """
                MATCH (p:Patient)
                WHERE p.__valid_to IS NULL
                RETURN DISTINCT
                    p.id as id,
                    p.firstName as firstName,
                    p.lastName as lastName,
                    p.dateOfBirth as dateOfBirth
                LIMIT $limit
                """
                
                result = await session.run(query, limit=limit)
                records = await result.fetch(limit)  # Fetch up to 100 records
                
                patients = []
                for record in records:
                    # Convert Neo4j date to Python date if needed
                    dob = record.get("dateOfBirth")
                    
                    patient = Patient(
                        id=record.get("id", f"PT-{len(patients) + 10001}"),
                        firstName=record.get("firstName", "Unknown"),
                        lastName=record.get("lastName", "Unknown"),
                        dateOfBirth=dob,
                    )
                    patients.append(patient)
                
                return patients
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error retrieving patients: {str(e)}")
            # Return empty list as fallback
            return []
    
    async def get_patient_journey(self, patient_id: str) -> PatientJourney:
        """
        Get a patient's journey including events, reports, and treatment outcomes
        
        Args:
            patient_id: ID of the patient
            
        Returns:
            PatientJourney object with all journey information
        """
        try:
            async with self.storage._get_session() as session:
                # Query to get patient info
                patient_query = """
                MATCH (p:Patient {id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
                RETURN 
                    p.id as id,
                    p.firstName as firstName,
                    p.lastName as lastName,
                    p.dateOfBirth as dateOfBirth,
                    d.description as diagnosis
                """
                
                patient_result = await session.run(patient_query, patient_id=patient_id)
                patient_record = await patient_result.single(None)
                
                if not patient_record:
                    # If patient not found, return empty journey with error message
                    logger.error(f"Patient with ID {patient_id} not found")
                    raise ValueError(f"Patient with ID {patient_id} not found")
                
                # Calculate age from date of birth
                dob = patient_record.get("dateOfBirth")
                age = 0
                if isinstance(dob, str):
                    try:
                        birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
                        today = date.today()
                        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    except ValueError:
                        age = 0
                
                patient_info = PatientInfo(
                    id=patient_record.get("id", patient_id),
                    firstName=patient_record.get("firstName", "Unknown"),
                    lastName=patient_record.get("lastName", "Unknown"),
                    age=age,
                    diagnosis=patient_record.get("diagnosis")
                )
                
                # Query to get journey events
                events_query = """
                MATCH (p:Patient {id: $patient_id})-[:HAS_EVENT]->(e:MedicalEvent)
                OPTIONAL MATCH (e)-[:HAS_DIAGNOSIS]->(d:Diagnosis)
                RETURN 
                    e.id as id,
                    e.datetime as date,
                    e.name as label,
                    e.status as type,
                    d.description as details,
                    labels(e)[0] as nodeType
                ORDER BY e.datetime
                """
                
                events_result = await session.run(events_query, patient_id=patient_id)
                events_records = await events_result.fetch(100)  # Fetch up to 100 records
                
                journey_events = []
                for record in events_records:
                    event_date = record.get("date")
                    if isinstance(event_date, str):
                        try:
                            event_date = datetime.strptime(event_date, "%Y-%m-%d").date()
                        except ValueError:
                            event_date = date.today()
                    
                    event = JourneyEvent(
                        id=record.get("id", f"EV-{len(journey_events) + 1}"),
                        date=event_date or date.today(),
                        type=record.get("type", "Unknown"),
                        label=record.get("label") or "Medical Event",
                        details=record.get("details"),
                        nodeType=record.get("nodeType", "MedicalEvent")
                    )
                    journey_events.append(event)
                
                # Query to get medical reports
                reports_query = """
                MATCH (p:Patient {id: $patient_id})
                MATCH (p)-[:HAS_MEDICAL_EXAM]->(e:MedicalExamination)
                RETURN 
                    e.date as date,
                    e.type as type,
                    e.findings as findings,
                    e.recommendations as abnormalities,
                    'Completed' as status
                UNION
                MATCH (p:Patient {id: $patient_id})
                MATCH (p)-[:HAS_LAB_RESULT]->(l:LaboratoryResult)
                RETURN 
                    l.date as date,
                    l.type as type,
                    l.resultStatus as findings,
                    null as abnormalities,
                    l.resultStatus as status
                ORDER BY date
                """
                
                reports_result = await session.run(reports_query, patient_id=patient_id)
                reports_records = await reports_result.fetch(100)  # Fetch up to 100 records
                
                medical_reports = []
                for record in reports_records:
                    report_date = record.get("date")
                    if isinstance(report_date, str):
                        try:
                            report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
                        except ValueError:
                            report_date = date.today()
                    
                    report = MedicalReport(
                        date=report_date or date.today(),
                        type=record.get("type") or "Examination",
                        findings=record.get("findings") or "No findings recorded",
                        abnormalities=record.get("abnormalities"),
                        status=record.get("status") or "Completed"
                    )
                    medical_reports.append(report)
                
                # Query to get treatments and outcomes
                treatments_query = """
                MATCH (p:Patient {id: $patient_id})-[:UNDERGOES]->(proc:Procedure)
                OPTIONAL MATCH (proc)-[:DOCUMENTED_IN]->(report:OperativeReport)
                RETURN 
                    proc.name as treatment,
                    COALESCE(report.findings, 'Completed') as outcome,
                    proc.complications as complication
                """
                
                treatments_result = await session.run(treatments_query, patient_id=patient_id)
                treatments_records = await treatments_result.fetch(100)  # Fetch up to 100 records
                
                treatment_outcomes = []
                for record in treatments_records:
                    outcome = TreatmentOutcome(
                        treatment=record.get("treatment") or "Unknown Treatment",
                        outcome=record.get("outcome") or "Completed",
                        complication=record.get("complication")
                    )
                    treatment_outcomes.append(outcome)
                
                # Return journey with actual data from database
                return PatientJourney(
                    patientInfo=patient_info,
                    journeyEvents=journey_events or [],
                    medicalReports=medical_reports or [],
                    treatmentOutcomes=treatment_outcomes or []
                )
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error retrieving patient journey: {str(e)}")
            # Return empty journey as fallback
            return PatientJourney(
                patientInfo=None,
                journeyEvents=[],
                medicalReports=[],
                treatmentOutcomes=[]
            )

    async def get_patient_laboratory_results(self, patient_id: str) -> List[LaboratoryResult]:
        """
        Get laboratory results for a specific patient
        
        Args:
            patient_id: ID of the patient
            
        Returns:
            List of laboratory results with components
        """
        try:
            async with self.storage._get_session() as session:
                # First check if patient exists
                patient_query = """
                MATCH (p:Patient {id: $patient_id})
                RETURN p.id as id
                """
                
                patient_result = await session.run(patient_query, patient_id=patient_id)
                patient_record = await patient_result.single(None)
                
                if not patient_record:
                    logger.error(f"Patient with ID {patient_id} not found")
                    raise ValueError(f"Patient with ID {patient_id} not found")
                
                # Query to get laboratory results
                lab_results_query = """
                MATCH (p:Patient {id: $patient_id})-[:HAS_LAB_RESULT]->(lr:LaboratoryResult)
                RETURN 
                    lr.id as id,
                    lr.date as date,
                    lr.time as time,
                    lr.reportDate as reportDate,
                    lr.facility as facility
                ORDER BY lr.date, lr.time
                """
                
                lab_results = await session.run(lab_results_query, patient_id=patient_id)
                lab_records = await lab_results.fetch(100)  # Fetch up to 100 records
                
                results = []
                for record in lab_records:
                    # Query to get components for each laboratory result
                    components_query = """
                    MATCH (lr:LaboratoryResult {id: $lab_id})<-[:PART_OF]-(c:TestComponent)
                    RETURN 
                        c.id as id,
                        c.name as name,
                        c.value as value,
                        c.units as units,
                        c.referenceRange as referenceRange
                    """
                    
                    lab_id = record.get("id")
                    components_result = await session.run(components_query, lab_id=lab_id)
                    component_records = await components_result.fetch(100)
                    
                    components = []
                    for comp_record in component_records:
                        component = LaboratoryComponent(
                            id=comp_record.get("id") or f"comp-{len(components)+1:03d}",
                            name=comp_record.get("name") or "Unknown Test",
                            value=comp_record.get("value") or "",
                            units=comp_record.get("units") or "",
                            referenceRange=comp_record.get("referenceRange") or ""
                        )
                        components.append(component)
                    
                    lab_result = LaboratoryResult(
                        id=record.get("id") or f"lab-{len(results)+1:03d}",
                        date=record.get("date") or datetime.now().strftime("%Y-%m-%d"),
                        time=record.get("time") or datetime.now().strftime("%H:%M:%S"),
                        reportDate=record.get("reportDate") or record.get("date") or datetime.now().strftime("%Y-%m-%d"),
                        facility=record.get("facility") or "Unknown Facility",
                        components=components
                    )
                    results.append(lab_result)
                
                return results
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error retrieving laboratory results: {str(e)}")
            # Return empty list as fallback
            return []
