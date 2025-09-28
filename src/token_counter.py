# src/token_counter.py
import tiktoken
from typing import List, Tuple
from langchain.schema import Document
from src.utils import get_logger

logger = get_logger(__name__)

class TokenCounter:
    """
    TokenCounter for estimating token usage per text or document.
    Default model: cl100k_base (similar to OpenAI/Gemini models)
    """
    def __init__(self, model: str = "cl100k_base"):
        self.model = model
        try:
            self.encoder = tiktoken.get_encoding(self.model)
            logger.info("TokenCounter initialized with model: %s", self.model)
        except Exception as e:
            logger.error("Failed to initialize TokenCounter with model %s: %s", self.model, e, exc_info=True)
            raise

    def count_tokens(self, text: str) -> int:
        """Return number of tokens in a string."""
        token_count = len(self.encoder.encode(text))
        logger.debug("Counted %d tokens for text: %s...", token_count, text[:50])
        return token_count

    def get_total_tokens(self, documents: List[Document]) -> Tuple[int, List[Tuple[int, int, str]]]:
        """
        Calculate total tokens across multiple documents.

        Returns:
            total_tokens (int): Total token count across all docs
            per_doc (List[Tuple[int, int, str]]): List of (doc_index, tokens, source)
        """
        total_tokens = 0
        per_doc = []

        for i, doc in enumerate(documents):
            tokens = self.count_tokens(doc.page_content)
            source = doc.metadata.get("source", "unknown")
            per_doc.append((i, tokens, source))
            total_tokens += tokens
            logger.debug("Doc %d (%s) has %d tokens", i, source, tokens)

        logger.info("Total tokens across %d documents: %d", len(documents), total_tokens)
        return total_tokens, per_doc


if __name__ == "__main__":
    token_counter = TokenCounter()
    sample_text = "Calculate total tokens across multiple documents"
    n_count = token_counter.count_tokens(sample_text)
    logger.info("Sample text token count: %d", n_count)
