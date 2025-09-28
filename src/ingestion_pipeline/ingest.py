# src/ingestion_pipeline/ingest.py
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import uuid
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils import get_logger
from dotenv import load_dotenv

from src.chunker import Chunker
from src.file_processor import FileProcessor

# Load environment variables
load_dotenv()
FAISS_PATH = os.getenv("FAISS_DB_PATH", "./faiss_db")

# Initialize embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Module-level logger
logger = get_logger(__name__)


class VectorStoreFAISS:
    """
    Handles creation and saving of a FAISS vector store from document chunks.
    """

    def __init__(self):
        self.FAISS_PATH = FAISS_PATH  # Path to save/load FAISS index
        self.embedding_function = embedding_function
        self.file_processor = FileProcessor()
        self.chunker = Chunker(chunk_size=600, chunk_overlap=90)

        logger.info(f"Initialized VectorStoreFAISS with FAISS_PATH={self.FAISS_PATH}")

    def create_chunks_from_file(self, file_path: str) -> List[Document]:
        """
        Process a file and create token-aware chunks.
        """
        logger.info(f"Processing file(s) from {file_path}")
        documents, num_files = self.file_processor.process(file_path)
        logger.info(f"Processed {num_files} file(s) from {file_path}")

        chunks = self.chunker.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from documents")

        return chunks

    def create_new_faiss_index(self, chunks: List["Document"]) -> Tuple[FAISS, Dict[str, int]]:
        """
        Create a FAISS index from document chunks and save it locally.
        Returns the FAISS object and basic stats.
        """
        try:
            logger.info("Starting FAISS index creation...")

            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.metadata.get("id", str(uuid.uuid4())) for chunk in chunks]

            logger.info(f"Preparing to embed {len(chunks)} document chunks...")
            db = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_function,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("FAISS index created successfully.")

            # Try saving the index to the defined path
            try:
                db.save_local(self.FAISS_PATH)
                logger.info(f"FAISS index saved at {self.FAISS_PATH}")
            except Exception as save_err:
                temp_path = f"faiss_temp_{uuid.uuid4().hex[:8]}"
                db.save_local(temp_path)
                logger.warning(
                    f"Failed to save FAISS index at {self.FAISS_PATH}, "
                    f"saved temporarily at {temp_path} | Error: {save_err}"
                )

            stats = {
                "InitialDocChunk_count": 0,
                "AddedDocChunk_count": len(chunks),
                "TotalDocChunk_count": len(db.docstore._dict)
            }
            logger.info(f"FAISS index stats: {stats}")

            return db, stats

        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}", exc_info=True)
            raise

    def ingest(self, file_path: str) -> Tuple[FAISS, Dict[str, int]]:
        """
        Full ingestion pipeline: process file, chunk, create FAISS index.
        """
        logger.info(f"Starting ingestion pipeline for {file_path}")
        chunks = self.create_chunks_from_file(file_path)
        db, stats = self.create_new_faiss_index(chunks)
        logger.info(f"Ingestion completed for {file_path}")
        return db, stats


# ===== Usage =====
if __name__ == "__main__":
    vector_store = VectorStoreFAISS()
    input_path = "./notes.pdf"  # Your file path
    db, stats = vector_store.ingest(input_path)
    logger.info(f"Pipeline finished. Final stats: {stats}")
