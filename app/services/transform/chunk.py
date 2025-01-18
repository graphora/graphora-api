from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

def do_semantic_chunking(text: str) -> List[Document]:
  text_splitter = SemanticChunker(
      HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
      breakpoint_threshold_type="gradient"
  )
  return text_splitter.create_documents([text])