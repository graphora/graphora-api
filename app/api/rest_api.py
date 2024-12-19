from fastapi import APIRouter, UploadFile, File, HTTPException, status, WebSocket, WebSocketDisconnect
from app.services.document_processor import DocumentProcessor
from app.services.extraction_service import ExtractionService
from app.schemas.document import DocumentResponse
from app.schemas.feedback import PydanticFeedbackInput
from app.schemas.schema_generator import SchemaGeneratorInput, SchemaGeneratorResponse
from app.services.ontology_generator_service import OntologyGeneratorService
from app.utils.ontology_cache import OntologyCache, generate_session_id
from datetime import datetime
from app.utils.logger import logger
import traceback

router = APIRouter()
ontology_cache = OntologyCache()

ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/json": ".json",
    "application/csv": ".json",
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# WebSocket endpoint for real-time updates
@router.websocket("/ws/agent-updates/{session_id}")
async def agent_updates(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            # Handle real-time agent updates
            data = await websocket.receive_text()
            # Process and respond to websocket messages
            await websocket.send_json({"status": "received", "data": data})
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@router.post("/generate-schema", response_model=SchemaGeneratorResponse)
def generate_schema(input_data: SchemaGeneratorInput) -> SchemaGeneratorResponse:
    """
    Generate a Pydantic schema from input text.
    The endpoint uses AI to analyze the text and create appropriate schema definitions.
    """
    if not input_data.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty"
        )

    logger.info(f"Received schema generation request for schema: {input_data.base_schema_name}")
    try:
        generator = OntologyGeneratorService()
        
        schema_definition = generator.generate_ontology(
            text=input_data.text
        )
        
        if not schema_definition:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate schema"
            )
        
        logger.info(f"Successfully generated {len(schema_definition.nodes)} node definitions")
        logger.info(f"Successfully generated {len(schema_definition.relationships)} edge definitions")

        # Generate session ID and store ontology
        session_id = generate_session_id()
        ontology_cache.store(session_id, schema_definition)

        logger.info('Session ID: ' + session_id)
        
        return SchemaGeneratorResponse(
            ontology=schema_definition,
            session_id=session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error generating schema: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    """
    Upload and process a document (PDF or text file).
    The document will be processed through the extraction pipeline.
    
    Accepts only POST requests with multipart/form-data containing a file.
    Supported file types: PDF (.pdf) and text (.txt) files.
    """
    logger.info(f"Received upload request for file: {file.filename}")
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_MIME_TYPES.values())}"
        )
    
    try:
        doc_processor = DocumentProcessor()
        doc_output = await doc_processor.process_uploaded_file(file)
        logger.info(f"Successfully processed document with ID: {doc_output.id}")
        
        # Convert DocumentOutput to DocumentResponse
        return DocumentResponse(
            id=doc_output.id,
            content=doc_output.content,
            entities=[{
                'id': entity.id,
                'type': entity.type,
                'value': entity.value,
                'confidence': entity.confidence
            } for entity in doc_output.entities],
            relationships=[{
                'source_id': rel.source_id,
                'target_id': rel.target_id,
                'type': rel.type,
                'confidence': rel.confidence
            } for rel in doc_output.relationships]
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing document"
        )
    
# Implement notification function
async def notify_frontend(message_type: str, data: dict):
    await manager.broadcast({
        "type": message_type,
        "data": data
    })


@router.post("/feedback/{document_id}", status_code=status.HTTP_200_OK)
async def submit_feedback(document_id: str, feedback: PydanticFeedbackInput):
    """
    Submit feedback for a processed document.
    This allows for human-in-the-loop correction of extracted information.
    """
    logger.info(f"Received feedback for document: {document_id}")
    try:
        extraction_service = ExtractionService()
        success = await extraction_service.process_feedback(document_id, feedback)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process feedback"
            )
        logger.info(f"Successfully processed feedback for document: {document_id}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing feedback"
        )

# Health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
