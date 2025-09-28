from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from langchain.schema import Document
from src.file_processor import FileProcessor
from src.llm_agent import LLMInvoke
from src.chunker import Chunker


class ConceptMapper:
    def __init__(self, llm_invoke, max_workers: int = 8):
        """
        Initialize the concept extractor with an LLM instance and thread pool size.
        """
        self.llm_invoke = llm_invoke
        self.max_workers = max_workers

    def process_chunk(self, chunk: Document) -> Dict:
        """
        Extract main concepts from a single document chunk using the LLM.
        Returns a dictionary with source and extracted concepts.
        """
        try:
            prompt = f"""
            You are an expert educator specializing in creating detailed concept maps from academic texts.
            Given the following excerpt from a longer document, extract the main ideas, detailed concepts, and supporting details critical to understanding the material.

            Focus on:
            - Key concepts or terms introduced in the text.
            - Definitions or explanations of these concepts.
            - Relationships between concepts.
            - Any examples or applications mentioned.

            Context:
            {chunk.page_content}

            Respond with a structured list of detailed main ideas and concepts."""

            result = self.llm_invoke.llm_response(prompt)
            return {"chunk_source": chunk.metadata.get("source", "unknown"), "concepts": result["answer"]}

        except Exception:
            return None

    def extract(self, chunks: List[Document]) -> List[Dict]:
        """
        Extract main concepts from multiple chunks in parallel.
        Returns a list of extracted concept dictionaries.
        """
        concepts = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {executor.submit(self.process_chunk, chunk): chunk for chunk in chunks}

            for future in as_completed(future_to_chunk):
                result = future.result()
                if result:
                    concepts.append(result)

        # Return only the concepts from each chunk
        return [c['concepts'] for c in concepts]


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

    print(f"Extracted concepts from {len(concepts_Map_list)} chunks")
    print("Extracted concepts:", concepts_Map_list)
