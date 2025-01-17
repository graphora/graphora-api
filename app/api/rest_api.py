from fastapi import APIRouter, File, HTTPException, status, Query
from app.schemas.schema_generator import SchemaGeneratorInput, SchemaGeneratorResponse
from app.services.ontology_parser import OntologyParser
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

@router.post("/ontology", response_model=SchemaGeneratorResponse)
async def generate_schema(input_data: SchemaGeneratorInput) -> SchemaGeneratorResponse:
    if not input_data.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty"
        )
    
    try:
        parser = OntologyParser()
        
        schema_definition = await parser.parse(
            ontology_text=input_data.text
        )
        
        if not schema_definition:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate schema"
            )
        
        logger.info(f"Successfully generated {str(schema_definition)} node definitions")

        # Generate session ID and store ontology
        session_id = generate_session_id()
        ontology_cache.store(session_id, schema_definition)

        logger.info('Session ID: ' + session_id)
        
        return SchemaGeneratorResponse(
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

# @router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
# async def upload_document(
#     file: UploadFile = File(...),
#     session_id: str = Query(..., description="Session ID for accessing stored ontology")
# ) -> DocumentResponse:
#     """
#     Upload and process a document with intelligent chunking and entity extraction.
#     Requires a session ID to access the stored ontology.
#     """
#     logger.info(f"Received upload request for file: {file.filename} with session: {session_id}")
    
#     if file.content_type not in ALLOWED_MIME_TYPES:
#         logger.warning(f"Invalid file type: {file.content_type}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_MIME_TYPES.values())}"
#         )
    
#     try:
#         doc_processor = DocumentProcessor()
#         ontology = ontology_cache.get(session_id)
#         if ontology == None:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail='Ontology Not set for extraction. Please set via /generate-schema first'
#             )
#         doc_output = await doc_processor.process_uploaded_file(file, ontology)
#         logger.info(f"Successfully processed document with ID: {doc_output.id}")
        
#         return DocumentResponse(
#             id=doc_output.id,
#             content=doc_output.content,
#             entities=doc_output.entities,
#             relationships=doc_output.relationships
#         )
#     except ValueError as ve:
#         logger.error(f"Validation error: {str(ve)}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=str(ve)
#         )
#     except Exception as e:
#         logger.error(f"Error processing document: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error processing document"
#         )
    

# Health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
