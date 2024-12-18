from typing import Optional, List, Tuple, Dict
from fastapi import UploadFile
from pypdf import PdfReader
import uuid
from app.schemas.document import (
    DocumentInput, DocumentOutput, Entity, 
    Relationship, MetadataInput
)
from app.services.extraction_service import ExtractionService
from app.services.preprocessing_service import PreprocessingService
from app.utils.logger import logger

ALLOWED_FILE_TYPES = {".pdf", ".txt", ".csv", ".json", ".doc", ".docx"}

class DocumentProcessor:
    def __init__(self):
        self.extraction_service = ExtractionService()
    
    async def process_uploaded_file(self, file: UploadFile) -> DocumentOutput:
        """Process an uploaded document file and extract entities and relationships."""
        try:
            logger.info(f"Starting to process uploaded file: {file.filename}")
            content = await self._extract_text(file)
            doc_id = str(uuid.uuid4())
            logger.info(f"Generated document ID: {doc_id}")
            
            # Process content through extraction pipeline
            logger.info("Starting entity extraction")
            
            # # Initialize preprocessing service
            # preprocessing_service = PreprocessingService()
            
            # # Run preprocessing workflow
            # success, message = await preprocessing_service.preprocess_document(content, doc_id) # changed content here
            # if not success:
            #     logger.error(f"Preprocessing failed: {message}")
            #     raise ValueError(f"Document preprocessing failed: {message}")
            
            # # Get processed data from temporary subgraph
            # temp_graph = await self.extraction_service.get_temp_subgraph(doc_id)
            # if not temp_graph:
            #     raise ValueError("Could not retrieve processed document data")
            
            # raw_entities = temp_graph.get("entities", [])
            # if not raw_entities:
            #     logger.warning(f"No entities extracted after preprocessing")
            #     raise ValueError(f"Entity extraction failed after preprocessing")

            # logger.info(f"Successfully extracted {len(raw_entities)} entities")
            
            # # Process relationships only if we have entities
            # logger.info("Starting relationship extraction")
            # raw_relationships, rel_status = await self.extraction_service.extract_relationships(raw_entities)
            # logger.info(f"Relationship extraction status: {rel_status}")
            
            # # Convert to proper types
            # entities = []
            # for i, e in enumerate(raw_entities):
            #     try:
            #         entities.append(Entity(
            #             id=str(i),
            #             type=e["type"],
            #             value=e["value"],
            #             confidence=e.get("confidence", 0.0)
            #         ))
            #     except Exception as entity_error:
            #         logger.error(f"Error converting entity {e}: {str(entity_error)}")
            #         continue
            
            # relationships = []
            # for r in raw_relationships:
            #     try:
            #         relationships.append(Relationship(
            #             source_id=r["source_id"],
            #             target_id=r["target_id"],
            #             type=r["type"],
            #             confidence=r.get("confidence", 0.0)
            #         ))
            #     except Exception as rel_error:
            #         logger.error(f"Error converting relationship {r}: {str(rel_error)}")
            #         continue
            entities=[]
            relationships=[]
            return DocumentOutput(
                id=doc_id,
                content=content,
                entities=entities,
                relationships=relationships
            )
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise ValueError(f"Failed to process document: {str(e)}")
    
    async def _extract_text(self, file: UploadFile) -> str:
        """
        Extract text content from uploaded file.
        Currently supports PDF and text files.
        """
        if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_FILE_TYPES):
            raise ValueError(f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}")

        try:
            content = ""
            file_ext = file.filename.lower()[-4:]
            
            if file_ext == '.pdf':
                logger.info(f"Processing PDF file: {file.filename}")
                try:
                    pdf = PdfReader(file.file)
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            content += f"\n--- Page {page_num} ---\n{page_text}"
                        logger.debug(f"Extracted text from page {page_num}")
                except Exception as pdf_error:
                    logger.error(f"Error processing PDF: {str(pdf_error)}")
                    raise ValueError(f"Failed to process PDF file: {str(pdf_error)}")
            
            else:
                logger.info(f"Processing text file: {file.filename}")
                try:
                    content = (await file.read()).decode('utf-8')
                except UnicodeDecodeError:
                    logger.error("Failed to decode text file with UTF-8")
                    raise ValueError("Text file must be UTF-8 encoded")
                except Exception as txt_error:
                    logger.error(f"Error reading text file: {str(txt_error)}")
                    raise ValueError(f"Failed to read text file: {str(txt_error)}")
            
            if not content.strip():
                logger.warning(f"No text content extracted from file: {file.filename}")
                raise ValueError("No text content could be extracted from the file")
            
            logger.info(f"Successfully extracted {len(content)} characters from {file.filename}")
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from file {file.filename}: {str(e)}")
            raise ValueError(f"Failed to extract text from file: {str(e)}")
        finally:
            try:
                await file.seek(0)
            except Exception as seek_error:
                logger.warning(f"Failed to reset file pointer: {str(seek_error)}")
    
    async def get_document(self, doc_id: str) -> Optional[DocumentOutput]:
        """Retrieve a processed document by its ID."""
        try:
            # Get document from extraction service's temporary storage
            doc_data = await self.extraction_service.get_temp_subgraph(doc_id)
            if not doc_data:
                logger.warning(f"Document {doc_id} not found")
                return None
            
            # Convert stored data to DocumentOutput format
            entities = [
                Entity(
                    id=entity["id"],
                    type=entity["type"],
                    value=entity["value"],
                    confidence=entity.get("confidence", 0.0)
                )
                for entity in doc_data.get("entities", [])
            ]
            
            relationships = [
                Relationship(
                    source_id=rel["source_id"],
                    target_id=rel["target_id"],
                    type=rel["type"],
                    confidence=rel.get("confidence", 0.0)
                )
                for rel in doc_data.get("relationships", [])
            ]
            
            return DocumentOutput(
                id=doc_id,
                content=doc_data.get("content", ""),
                entities=entities,
                relationships=relationships
            )
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    async def list_documents(self) -> List[DocumentOutput]:
        """List all processed documents."""
        try:
            # Get all documents from extraction service's temporary storage
            temp_graphs = self.extraction_service.temp_graphs
            
            documents = []
            for doc_id, doc_data in temp_graphs.items():
                try:
                    doc = await self.get_document(doc_id)
                    if doc:
                        documents.append(doc)
                except Exception as doc_error:
                    logger.error(f"Error processing document {doc_id}: {str(doc_error)}")
                    continue
            
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
            
    async def process_document(self, input: DocumentInput) -> DocumentOutput:
        """Process a document input and extract entities and relationships."""
        try:
            logger.info("Starting document processing")
            doc_id = str(uuid.uuid4())
            
            # Initialize preprocessing service
            preprocessing_service = PreprocessingService()
            
            # Run preprocessing workflow
            success, message = await preprocessing_service.preprocess_document(input.content, doc_id)
            if not success:
                logger.error(f"Preprocessing failed: {message}")
                raise ValueError(f"Document preprocessing failed: {message}")
            
            # Get processed data from temporary subgraph
            temp_graph = await self.extraction_service.get_temp_subgraph(doc_id)
            if not temp_graph:
                raise ValueError("Could not retrieve processed document data")
            
            # Convert to proper types
            entities = []
            raw_entities = temp_graph.get("entities", [])
            for i, e in enumerate(raw_entities):
                try:
                    entities.append(Entity(
                        id=str(i),
                        type=e["type"],
                        value=e["value"],
                        confidence=e.get("confidence", 0.0)
                    ))
                except Exception as entity_error:
                    logger.error(f"Error converting entity {e}: {str(entity_error)}")
                    continue
            
            raw_relationships = temp_graph.get("relationships", [])
            relationships = [
                Relationship(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    type=r["type"],
                    confidence=r.get("confidence", 0.0)
                )
                for r in raw_relationships
            ]
            
            # Create temporary subgraph for review  - This line was already present in original code and needs to remain
            await self.extraction_service.create_temp_subgraph(
                doc_id,
                input.content,
                raw_entities,
                raw_relationships
            )
            
            return DocumentOutput(
                id=doc_id,
                content=input.content,
                entities=entities,
                relationships=relationships
            )
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise ValueError(f"Failed to process document: {str(e)}")