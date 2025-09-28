from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
import re
from src.utils import get_logger

logger = get_logger(__name__)

class QuestionGenerator:
    """
    Generates multiple-choice questions from ranked concepts using LLMs.
    Each question includes four answer options and a ground_truth reference.
    """
    def __init__(self, llm_invoke, passage_extractor, max_workers: int = 8):
        self.llm_invoke = llm_invoke
        self.passage_extractor = passage_extractor
        self.max_workers = max_workers
        logger.info("QuestionGenerator initialized with max_workers=%d", max_workers)

    def _generate_single(self, concept_dict: Dict, num_questions: int) -> (str, str):
        main_idea = f"{concept_dict['concept']}: {concept_dict['summary']}"
        logger.debug("Generating questions for main idea: %s", main_idea[:80])
        passages = self.passage_extractor.extract(main_idea)
        prompt = f"""
        Based on the following main idea and its relevant passages, create {num_questions}
        multiple-choice questions that require deep understanding, critical thinking, and detailed analysis.
        The questions should go beyond mere factual recall, involving higher-order thinking skills like analysis,
        synthesis, and evaluation.

        IMPORTANT INSTRUCTIONS:
        Do NOT include any lead-in phrases such as "Considering the following", "Based on the description", 
        "Analyzing the provided text", "Considering the given description", or "From the text above". 
        The question should start immediately with the core inquiry.
        Do not use the phrases "main idea" or "passages" in the question statement. Instead, directly address
        the content or concepts described.
        Provide four answer choices for each question:
        - The choices should start with A., B., C., and D.
        - One correct answer.
        - Three plausible distractors that are contextually appropriate, relevant to the content, and
        reflect common misunderstandings or errors without introducing contradictory or irrelevant
        information.
        - Return only a single valid JSON array (no prose, preamble, or code blocks).

        Main Idea:
        {main_idea}

        Passages:
        {passages}
        """
        try:
            result = self.llm_invoke.llm_response(prompt)
            logger.info("Generated raw output for main idea: %s", main_idea[:80])
            return result["answer"], main_idea
        except Exception as e:
            logger.error("Error generating questions for main idea: %s, Error: %s", main_idea[:80], e, exc_info=True)
            return "", main_idea

    @staticmethod
    def _clean_and_add_ground_truth(raw_output: str, main_idea: str) -> List[Dict]:
        """Remove code fences, parse JSON, and add ground_truth field to each question."""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group(0))
                for q in questions:
                    q["ground_truth"] = main_idea
                return questions
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON for main idea: %s, Error: %s", main_idea[:80], e)
                return []
        return []

    def generate_questions(self, ranked_concepts: List[Dict], num_questions: int = 2, save_path: str = None) -> Dict:
        all_questions_json = {"questions": []}
        logger.info("Starting generation of questions for %d concepts.", len(ranked_concepts))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._generate_single, concept_dict, num_questions)
                for concept_dict in ranked_concepts
            ]
            for future in as_completed(futures):
                raw_output, main_idea = future.result()
                parsed = self._clean_and_add_ground_truth(raw_output, main_idea)
                all_questions_json["questions"].extend(parsed)
                logger.debug("Added %d questions for main idea: %s", len(parsed), main_idea[:80])

        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump(all_questions_json, f, indent=2)
                logger.info("Saved generated questions to %s", save_path)
            except Exception as e:
                logger.error("Failed to save questions to %s: %s", save_path, e, exc_info=True)

        logger.info("Question generation completed. Total questions: %d", len(all_questions_json["questions"]))
        return all_questions_json






# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import List, Dict
# import json
# import re
# from src.utils import get_logger
# logger = get_logger(__name__)

# class QuestionGenerator:
#     """
#     Generates multiple-choice questions from ranked concepts using LLMs.
#     Each question includes four answer options and a ground_truth reference.
#     """
#     def __init__(self, llm_invoke, passage_extractor, max_workers: int = 8):
#         self.llm_invoke = llm_invoke
#         self.passage_extractor = passage_extractor
#         self.max_workers = max_workers

#     def _generate_single(self, concept_dict: Dict, num_questions: int) -> (str, str):
#         main_idea = f"{concept_dict['concept']}: {concept_dict['summary']}"
#         passages = self.passage_extractor.extract(main_idea)
#         prompt = f"""
        
#         Based on the following main idea and its relevant passages, create {num_questions}
#         multiple-choice questions that require deep understanding, critical thinking, and detailed analysis.
#         The questions should go beyond mere factual recall, involving higher-order thinking skills like analysis,
#         synthesis, and evaluation.

#         IMPORTANT INSTRUCTIONS:
#         Do NOT include any lead-in phrases such as "Considering the following", "Based on the description", "Analyzing the provided text", "Considering the given description"
#         or "From the text above". The question should start immediately with the core inquiry.
#         Do not use the phrases "main idea" or "passages" in the question statement. Instead, directly address
#         the content or concepts described.
#         Provide four answer choices for each question:
#         - The choices should start with A., B., C., and D.
#         - One correct answer.
#         - Three plausible distractors that are contextually appropriate, relevant to the content, and
#         reflect common misunderstandings or errors without introducing contradictory or irrelevant
#         information.
#         - Return only a single valid JSON array (no prose, preamble, or code blocks).

#         Main Idea:
#         {main_idea}

#         Passages:
#         {passages}
#         """

#         result = self.llm_invoke.llm_response(prompt)
#         return result["answer"], main_idea

#     @staticmethod
#     def _clean_and_add_ground_truth(raw_output: str, main_idea: str) -> List[Dict]:
#         """Remove code fences, parse JSON, and add ground_truth field to each question."""
#         cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_output.strip())
#         match = re.search(r"\[.*\]", cleaned, re.DOTALL)
#         if match:
#             try:
#                 questions = json.loads(match.group(0))
#                 for q in questions:
#                     q["ground_truth"] = main_idea
#                 return questions
#             except json.JSONDecodeError:
#                 return []
#         return []

#     def generate_questions(self, ranked_concepts: List[Dict], num_questions: int = 2, save_path: str = None) -> Dict:
#         all_questions_json = {"questions": []}

#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = [
#                 executor.submit(self._generate_single, concept_dict, num_questions)
#                 for concept_dict in ranked_concepts
#             ]
#             for future in as_completed(futures):
#                 raw_output, main_idea = future.result()
#                 parsed = self._clean_and_add_ground_truth(raw_output, main_idea)
#                 all_questions_json["questions"].extend(parsed)

#         if save_path:
#             with open(save_path, "w") as f:
#                 json.dump(all_questions_json, f, indent=2)

#         return all_questions_json

# # #Usage example:
# # if __name__ == "__main__":
# #     passage_extractor = PassageExtractor(faiss_index=db.index, doc_index=db)
# #     generator = QuestionGenerator(llm_invoke, passage_extractor)
# #     # test on subset by slicing for API Cost limitations :  ranked_concepts[:K] 
# #     questions_json = generator.generate_questions(ranked_concepts[:20], save_path="questions.json")
# #     print(json.dumps(questions_json, indent=2))
