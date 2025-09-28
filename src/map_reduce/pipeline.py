# src/map_reduce/pipeline.py
# This file implements the MapReduce pipeline for concept extraction and ranking
# It uses the ConceptMapper, ConceptCombiner, ConceptReducer, and ConceptRanker classes
# to process text chunks and extract ranked concepts.

from src.map_reduce.concept_mapper import ConceptMapper
from src.map_reduce.concept_combiner import ConceptCombiner
from src.map_reduce.concept_reducer import ConceptReducer
from src.map_reduce.concept_ranker import ConceptRanker

from src.llm_agent import LLMInvoke
from src.token_counter import TokenCounter
from src.utils import get_logger

logger = get_logger(__name__)


class ConceptPipeline:
    """A MapReduce pipeline for extracting and ranking concepts from text chunks."""    
    def __init__(self, llm=None, max_workers=8, max_tokens=100000):
        self.llm = llm or LLMInvoke()
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.counter = TokenCounter()

        # Initialize components
        self.mapper = ConceptMapper(self.llm, max_workers=self.max_workers)
        self.combiner = ConceptCombiner(self.llm, self.counter, self.max_tokens)
        self.reducer = ConceptReducer(self.llm)
        self.ranker = ConceptRanker(self.llm)

        logger.info("ConceptPipeline initialized with max_workers=%d, max_tokens=%d", 
                    self.max_workers, self.max_tokens)

    def run(self, chunks):
        """Run the full MapReduce pipeline on text chunks."""
        logger.info("Starting concept extraction pipeline with %d chunks.", len(chunks))

        try:
            # 1. Map
            logger.debug("Step 1: Mapping concepts from chunks...")
            concepts_map_list = self.mapper.extract(chunks)
            logger.info("Mapped %d concept groups.", len(concepts_map_list))

            # 2. Combine
            logger.debug("Step 2: Combining concepts...")
            final_main_ideas = self.combiner.combine_concepts(concepts_map_list)
            logger.info("Combined into %d main ideas.", len(final_main_ideas))

            # 3. Reduce
            logger.debug("Step 3: Reducing concepts...")
            reduced = self.reducer.reduce(final_main_ideas)
            logger.info("Reduced concepts to %d items.", len(reduced))

            # 4. Rank
            logger.debug("Step 4: Ranking concepts...")
            rankings = self.ranker.rank(reduced)
            ranked_concepts = self.ranker.get_ranked_concepts(reduced)
            logger.info("Ranked %d concepts.", len(ranked_concepts))

            return rankings, ranked_concepts

        except Exception as e:
            logger.error("Error in ConceptPipeline run: %s", e, exc_info=True)
            raise


# ===== Usage =====
if __name__ == "__main__":
    from src.file_processor import FileProcessor
    from src.chunker import Chunker

    input_path = "./notes.pdf"  # Your file path
    file_processor = FileProcessor()
    documents, num_files = file_processor.process(input_path)
    logger.info("Processed %d files from %s", num_files, input_path)

    chunker = Chunker(chunk_size=12000, chunk_overlap=1200)
    chunks = chunker.split_documents(documents)
    logger.info("Created %d chunks from documents.", len(chunks))

    pipeline = ConceptPipeline()
    rankings, ranked_concepts = pipeline.run(chunks)

    logger.info("Final Rankings: %s", rankings)
    logger.info("Concepts ranked by importance:")
    for i, concept in enumerate(ranked_concepts, 1):
        logger.info("%d. %s: %s", i, concept['concept'], concept['summary'])
