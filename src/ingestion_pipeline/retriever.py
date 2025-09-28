from collections import defaultdict
import numpy as np
from typing import List, Tuple
from langchain_core.documents import Document
from src.utils import get_logger

logger = get_logger(__name__)

class PassageExtractor:
    """
    Ensemble retrieval system combining FAISS, similarity search,
    and Reciprocal Rank Fusion (RRF) fusion .
    """
    def __init__(self, faiss_index, doc_index):
        self.faiss_index = faiss_index
        self.doc_index = doc_index
        logger.info("PassageExtractor initialized with FAISS and doc index.")

    @staticmethod
    def reciprocal_rank_fusion(results_lists, k=60):
        """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        logger.debug("Starting Reciprocal Rank Fusion (RRF)...")
        scores, doc_mapping = defaultdict(float), {}

        for results in results_lists:
            for rank, (doc, _) in enumerate(results):
                doc_id = doc.metadata.get("id")
                scores[doc_id] += 1 / (k + rank + 1)
                doc_mapping[doc_id] = doc

        sorted_doc_ids = sorted(scores, key=scores.get, reverse=True)
        logger.debug(f"RRF fusion complete. Combined {len(sorted_doc_ids)} unique documents.")
        return [(doc_mapping[doc_id], scores[doc_id]) for doc_id in sorted_doc_ids]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents using FAISS and similarity search, with RRF fusion."""
        logger.info(f"Retrieving top-{k} documents for query: {query}")

        results_lists = []

        try:
            # Method 1: FAISS search
            logger.debug("Running FAISS search...")
            query_embedding = self.doc_index.embedding_function.embed_query(query)
            D, I = self.faiss_index.search(np.array([query_embedding], dtype=np.float32), k=k)
            faiss_results = []
            for idx, d in zip(I[0], D[0]):
                if idx >= 0:
                    doc_id = self.doc_index.index_to_docstore_id[idx]
                    doc = self.doc_index.docstore._dict[doc_id]
                    faiss_results.append((doc, float(-d)))
            results_lists.append(faiss_results)
            logger.debug(f"FAISS search returned {len(faiss_results)} results.")

            # Method 2: Similarity search
            logger.debug("Running similarity search...")
            sim_results = self.doc_index.similarity_search_with_score(query, k=k)
            results_lists.append([(doc, -score) for doc, score in sim_results])
            logger.debug(f"Similarity search returned {len(sim_results)} results.")

            # Fuse results using RRF
            fused_results = self.reciprocal_rank_fusion(results_lists)
            logger.info(f"Retrieved {len(fused_results)} fused results for query.")

            return fused_results[:k]

        except Exception as e:
            logger.error(f"Error during retrieval for query '{query}': {e}", exc_info=True)
            return []

    def extract(self, question: str) -> List[str]:
        """Extract top passages to answer the question."""
        logger.info(f"Extracting passages for question: {question}")
        results = self.retrieve(question)
        passages = [doc.page_content for doc, _ in results]
        logger.info(f"Extracted {len(passages)} passages for question.")
        return passages


if __name__ == "__main__":
    from src.ingestion_pipeline.ingest import VectorStoreFAISS
    from src.file_processor import FileProcessor
    from src.chunker import Chunker

    input_path = "./notes.pdf"
    file_processor = FileProcessor()
    documents, num_files = file_processor.process(input_path)
    logger.info(f"Processed {num_files} files from {input_path}")

    chunker = Chunker(chunk_size=600, chunk_overlap=90)
    chunks = chunker.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from documents.")

    vector_store = VectorStoreFAISS()
    db, stats = vector_store.create_new_faiss_index(chunks)
    logger.info(f"FAISS index created with stats: {stats}")

    passage_extractor = PassageExtractor(faiss_index=db.index, doc_index=db)
    question = "Continuous-Time Fourier Series (CTFS):** CTFS represents periodic continuous-time signals..."
    top_passages = passage_extractor.extract(question)

    logger.info(f"Top passages for question '{question}':")
    for i, passage in enumerate(top_passages):
        logger.info(f"Passage {i+1}:\n{passage}\n")
