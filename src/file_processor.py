# src/file_processor.py
import os
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
import multiprocessing

from src.utils import get_logger    

logger = get_logger(__name__)

class FileProcessor:
    """
    FileProcessor handles loading documents from different file types (.pdf, .docx, .txt),
    supports directory traversal, and can process files in parallel using multiprocessing.
    """

    def check_type(self, file_path: str) -> List[Document]:
        """Detects file type and loads documents using the appropriate loader."""
        logger.debug("Loading file: %s", file_path)
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".pdf":
                loader = PyMuPDFLoader(file_path)
            elif file_extension in {".docx", ".doc"}:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = file_path

            logger.info("Loaded %d documents from %s", len(documents), file_path)
            return documents

        except Exception as e:
            logger.error("Failed to load file %s: %s", file_path, e, exc_info=True)
            return []

    def _process_file(self, file_path: str) -> List[Document]:
        """Helper function to process a single file (used for multiprocessing)."""
        return self.check_type(file_path)

    def process(self, input_path: str) -> List[Document]:
        """
        Processes a single file or all supported files in a directory.
        Uses multiprocessing for faster processing if multiple files are present.
        Returns a list of documents and the number of files processed.
        """
        logger.info("Starting processing for path: %s", input_path)
        if not os.path.exists(input_path):
            logger.error("Path not found: %s", input_path)
            raise FileNotFoundError(f"Path not found: {input_path}")

        # Gather all file paths
        if os.path.isdir(input_path):
            file_paths = [os.path.join(root, f) 
                          for root, _, files in os.walk(input_path) 
                          for f in files]
        else:
            file_paths = [input_path]

        logger.info("Found %d file(s) to process.", len(file_paths))
        num_processes = max(1, multiprocessing.cpu_count() - 2)
        logger.debug("Using %d processes for multiprocessing.", num_processes)

        all_documents = []

        if num_processes <= 1 or len(file_paths) <= 1:
            logger.info("Processing files sequentially...")
            for file_path in file_paths:
                try:
                    all_documents.extend(self.check_type(file_path))
                except Exception:
                    logger.warning("Skipping file due to error: %s", file_path)
        else:
            logger.info("Processing files in parallel using multiprocessing...")
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(self._process_file, file_paths)
            all_documents = [doc for result in results for doc in result]

        logger.info("Processed %d files and extracted %d documents.", len(file_paths), len(all_documents))
        return all_documents, len(file_paths)


if __name__ == "__main__":
    input_path = "./notes.pdf"  # Your file's path
    file_processor = FileProcessor()
    documents, num_files = file_processor.process(input_path)
    logger.info("Processing complete. %d files processed, %d documents extracted.", num_files, len(documents))
