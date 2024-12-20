from typing import Optional, List, Tuple, Dict
from fastapi import UploadFile
from pypdf import PdfReader
import uuid
from app.schemas.document import (
    DocumentOutput
)
from app.services.extraction_service import ExtractionService
from app.services.ontology_generator_service import Neo4jOntology
from app.utils.logger import logger
import traceback

ALLOWED_FILE_TYPES = {".pdf", ".txt", ".csv", ".json", ".doc", ".docx"}

class DocumentProcessor:
    def __init__(self):
        self.extraction_service = ExtractionService()
    
    async def process_uploaded_file(self, 
                                    file: UploadFile, 
                                    ontology: Neo4jOntology) -> DocumentOutput:
        """Process an uploaded document file and extract entities and relationships."""
        try:
            logger.info(f"Starting to process uploaded file: {file.filename}")
            content = await self._extract_text(file)
            doc_id = str(uuid.uuid4())
            logger.info(f"Generated document ID: {doc_id}")
            
            # Process content through extraction pipeline
            logger.info("Starting entity extraction")
            
            extraction = await self.extraction_service.extract(content=content, ontology=ontology)

            # TODO: Store the data as a temporary subgraph so that its easier to discard the created nodes. Its important not to mess with pre-existing nodes in the DB
            
            return DocumentOutput(
                id=doc_id,
                content=content,
                entities=extraction.entities,
                relationships=extraction.relationships
            )
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            traceback.print_exc()
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
    