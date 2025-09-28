# src/chunker.py
# This file implements a token-aware document chunker using LangChain's RecursiveCharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from src.token_counter import TokenCounter
from src.utils import get_logger

logger = get_logger(__name__)


class Chunker:
    """A token-aware document chunker using RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Token-aware document chunker.
        Splits documents into chunks based on token count rather than characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = TokenCounter()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.token_counter.count_tokens,
            is_separator_regex=False,
        )

        logger.info(
            "Initialized Chunker with chunk_size=%d and chunk_overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving token-based boundaries.

        Returns:
            List[Document]: Token-aware document chunks
        """
        try:
            logger.info("Starting document splitting for %d documents.", len(documents))
            # Merge all documents into a single document for splitting
            merged_text = "\n\n".join(doc.page_content for doc in documents)
            merged_doc = Document(
                page_content=merged_text,
                metadata={"source": "merged"},
            )

            logger.debug(
                "Merged document length (tokens): %d",
                self.token_counter.count_tokens(merged_text),
            )

            doc_chunks = self.splitter.split_documents([merged_doc])

            logger.info(
                "Split into %d chunks (chunk_size=%d, overlap=%d)",
                len(doc_chunks),
                self.chunk_size,
                self.chunk_overlap,
            )
            return doc_chunks

        except Exception as e:
            logger.error("Error in split_documents: %s", e, exc_info=True)
            raise RuntimeError(f"Error in split_documents: {e}") from e


if __name__ == "__main__":
    from src.file_processor import FileProcessor

    input_path = "./notes.pdf"  # Your file's path
    file_processor = FileProcessor()
    documents, num_files = file_processor.process(input_path)
    logger.info("Processed %d files from %s", num_files, input_path)

    chunker = Chunker(chunk_size=12000, chunk_overlap=1000)
    chunks = chunker.split_documents(documents)
    logger.info(
        "Total chunks created: %d with chunk_size=%d and overlap=%d",
        len(chunks),
        chunker.chunk_size,
        chunker.chunk_overlap,
    )
