import re
from typing import List, Dict

class ConceptRanker:

    """
    Ranks concepts based on importance using LLM prompts.
    Provides both numeric rankings and concepts sorted by importance.
    """
    def __init__(self, llm_invoke):
        self.llm_invoke = llm_invoke
    
    @staticmethod
    def format_concepts_for_ranking(dict_concepts: List[Dict[str, str]]) -> str:
        """
        Format the concept dictionaries into a readable string for the LLM prompt.
        """
        formatted_concepts = []
        for i, concept_dict in enumerate(dict_concepts, 1):
            concept_name = concept_dict.get("concept", "")
            summary = concept_dict.get("summary", "")
            
            if summary:
                formatted_concept = f"{i}. {concept_name}: {summary}"
            else:
                formatted_concept = f"{i}. {concept_name}"
            
            formatted_concepts.append(formatted_concept)
        
        return "\n".join(formatted_concepts)
    
    @staticmethod
    def parse_rankings(ranking_result: str) -> List[int]:
        """
        Parse the LLM ranking output to extract the ranking order.
        Expected format: [2, 1, 3] or similar numerical list.
        """
        # Look for patterns like [1, 2, 3] or Output: [1, 2, 3]
        bracket_pattern = r'\[([0-9,\s]+)\]'
        match = re.search(bracket_pattern, ranking_result)
        
        if match:
            # Extract numbers from the bracketed content
            numbers_str = match.group(1)
            rankings = [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
            return rankings
        
        # Fallback: look for individual numbers in the text
        numbers = re.findall(r'\b\d+\b', ranking_result)
        if numbers:
            return [int(x) for x in numbers]
        
        # If parsing fails, return sequential ranking as fallback
        return []
    
    def rank(self, dict_concepts: List[Dict[str, str]]) -> List[int]:
        """
        Rank the concepts using LLM and return the ranking order.
        
        Args:
            dict_concepts: List of dictionaries with 'concept' and 'summary' keys
            
        Returns:
            List of integers representing the ranking (e.g., [2, 1, 3] means 
            concept 2 is most important, concept 1 is second, concept 3 is third)
        """
        if not dict_concepts:
            return []
        
        # Format concepts for the prompt
        main_ideas = self.format_concepts_for_ranking(dict_concepts)
        
        # Create the ranking prompt
        prompt = f"""
        Instructions:
        Given the following groups of main ideas extracted from a text, rank them in order of importance,
        with the most important main idea receiving a rank of 1 and lower ranks for less important ideas.
        Focus on the most important aspects of the text and the main ideas that are critical to understanding
        the material. While sometimes important, background information or less critical ideas should be
        ranked lower.

        When ranking:
        - Assign a unique number to each main idea, starting from 1.
        - Ensure that the most important main idea is ranked first.
        - Rank the main ideas based on their relevance and significance.

        Example:
        Input: [Main Idea 1, Main Idea 2, Main Idea 3]
        Output: [2, 1, 3]

        Main Ideas:
        {main_ideas}

        Please provide your ranking as a list of numbers in brackets, like [rank_of_idea_1, rank_of_idea_2, rank_of_idea_3, ...]"""

        # Get LLM response
        result = self.llm_invoke.llm_response(prompt)
        ranking_output = result["answer"]
        
        # Parse the rankings
        rankings = self.parse_rankings(ranking_output)
        
        # Validate rankings - should have same length as input concepts
        if len(rankings) != len(dict_concepts):
            # Return sequential ranking as fallback
            rankings = list(range(1, len(dict_concepts) + 1))
        
        return rankings
    
    def get_ranked_concepts(self, dict_concepts: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Get concepts sorted by their importance ranking.
        
        Returns:
            List of concept dictionaries sorted by importance (most important first)
        """
        rankings = self.rank(dict_concepts)
        
        # Create tuples of (ranking, concept_dict) and sort by ranking
        ranked_pairs = list(zip(rankings, dict_concepts))
        ranked_pairs.sort(key=lambda x: x[0])  # Sort by ranking (1 = most important)
        
        # Return only the concept dictionaries in ranked order
        return [concept_dict for _, concept_dict in ranked_pairs]
    


from src.token_counter import TokenCounter
from src.file_processor import FileProcessor
from src.llm_agent import LLMInvoke
from src.chunker import Chunker
from src.map_reduce.concept_mapper import ConceptMapper
from src.map_reduce.concept_combiner import ConceptCombiner
from src.map_reduce.concept_reducer import ConceptReducer

# Usage example:
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


    parsed_reducer=ConceptReducer(llm_invoke)
    reduced=parsed_reducer.reduce(final_main_ideas)

    ranker = ConceptRanker(llm_invoke)
    rankings = ranker.rank(reduced)
    print(f"Rankings: {rankings}")
    
    ranked_concepts = ranker.get_ranked_concepts(reduced)
    print("Concepts ranked by importance:")
    for i, concept in enumerate(ranked_concepts, 1):
        print(f"{i}. {concept['concept']}: {concept['summary']}")