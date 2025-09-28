from typing import List
import math
from src.token_counter import TokenCounter
from src.file_processor import FileProcessor
from src.llm_agent import LLMInvoke
from src.chunker import Chunker
from src.map_reduce.concept_mapper import ConceptMapper

class ConceptCombiner:
    """
    Iteratively combines multiple concept lists into a single condensed set using LLM prompts,
    ensuring the total token count stays within the specified limit.
    """
    def __init__(self, llm_invoke, counter, max_tokens: int):
        self.llm_invoke = llm_invoke
        self.counter = counter
        self.max_tokens = max_tokens  # Maximum allowable tokens per LLM context

    def combine_concepts(self, concepts_list: List[str]) -> str:
        """
        Merge multiple concept strings into a single organized summary.
        Repeatedly batches and condenses until total tokens ≤ max_tokens.
        """
        current_sets = concepts_list.copy()

        while True:
            combined_text = "\n\n".join(current_sets)
            total_tokens = self.counter.count_tokens(combined_text)
            if total_tokens <= self.max_tokens:
                break  # Already within token limit

            # Heuristic batching: split roughly in half each iteration
            batch_size = max(1, math.ceil(len(current_sets) / 2))
            new_sets = []

            for i in range(0, len(current_sets), batch_size):
                batch = current_sets[i:i+batch_size]
                context = "\n\n".join(batch)
                prompt = f"""
                Instructions:
                You are combining multiple concept maps into a single, comprehensive summary while retaining all
                key ideas and details. Below are several lists of main ideas and concepts extracted from a larger
                document.
                
                Your task is to:
                1. Merge these lists into a single structured list, removing redundancies while keeping all unique
                and detailed information.
                2. Ensure all main ideas, relationships, and examples are preserved and clearly organized.
                
                Here are the concept maps to combine:
                Context:
                {context}
                
                Respond with the consolidated and organized list of main ideas and concepts."""

                result = self.llm_invoke.llm_response(prompt)
                new_sets.append(result["answer"])

            current_sets = new_sets  # Prepare for next iteration

        # Return final condensed concept set
        return "\n\n".join(current_sets)


# ===== Usage =====
if __name__ == "__main__":

    input_path = "./notes.pdf" # Your file 's path
    file_processor = FileProcessor()
    documents, num_files = file_processor.process(input_path)
    
    chunker = Chunker(chunk_size=12000, chunk_overlap=1200)
    chunks = chunker.split_documents(documents)
    
    llm_invoke = LLMInvoke()
    main_concept_extractor = ConceptMapper(llm_invoke, max_workers=8)
    concepts_Map_list = main_concept_extractor.extract(chunks)


    MAX_TOKENS = 100000  # Adjust for your LLM
    counter = TokenCounter()
    combiner = ConceptCombiner(llm_invoke, counter, MAX_TOKENS)
    final_main_ideas = combiner.combine_concepts(concepts_Map_list)

    print("Final condensed main concepts:")
    print(final_main_ideas)