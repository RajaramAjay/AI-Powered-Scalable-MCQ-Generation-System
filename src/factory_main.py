# src/factory_main.py
from src.mcq_generator import QuestionGenerator
from src.file_processor import FileProcessor
from src.chunker import Chunker
from src.ingestion_pipeline.ingest import VectorStoreFAISS
from src.ingestion_pipeline.retriever import PassageExtractor
from src.llm_agent import LLMInvoke
from src.map_reduce.pipeline import ConceptPipeline
from src.utils import get_logger

logger = get_logger(__name__)

class MCQPipelineFactory:
    def __init__(self, input_path: str):
        self.input_path = input_path

        # Initialize components
        self.vector_store = VectorStoreFAISS()
        self.file_processor = FileProcessor()
        self.chunker = Chunker(chunk_size=12000, chunk_overlap=1200)
        self.concept_pipeline = ConceptPipeline()
        self.llm = LLMInvoke()

        logger.info(f"MCQPipelineFactory initialized with input_path={input_path}")

    def run_pipeline(self):
        logger.info("Starting MCQ pipeline execution...")

        # Step 1: Ingest into vector store
        db, stats = self.vector_store.ingest(self.input_path)
        logger.info(f"FAISS index stats: {stats}")

        # Step 2: Process and chunk documents
        documents, num_files = self.file_processor.process(self.input_path)
        logger.info(f"Processed {num_files} files from {self.input_path}")

        chunks = self.chunker.split_documents(documents)
        logger.info(f"Total chunks created: {len(chunks)} (chunk size=12000, overlap=1200)")

        # Step 3: Extract passages and rank concepts
        passage_extractor = PassageExtractor(faiss_index=db.index, doc_index=db)
        rankings, ranked_concepts = self.concept_pipeline.run(chunks)
        logger.info(f"Rankings: {rankings}")
        logger.info("Concepts ranked by importance (top concepts):")

        for i, concept in enumerate(ranked_concepts, 1):
            logger.debug(f"{i}. {concept['concept']}: {concept['summary']}")

        # Step 4: Generate MCQs
        qg = QuestionGenerator(self.llm, passage_extractor=passage_extractor)
        questions_json = qg.generate_questions(ranked_concepts[:5], save_path="questions.json")
        logger.info("Generated questions saved to questions.json")

        logger.info("MCQ pipeline execution completed successfully")
        return questions_json
    def template_method_example(self):
        # Example of a template method that could be overridden in subclasses
        temp ={
                "questions": [
                    {
                    "question": "Given a signal x(t), how would the order of applying a time scaling by a factor of 'a' and a time shift by 'b' affect the final transformed signal if the intention is to achieve the form x(at - b)?",
                    "options": {
                        "A": "Applying the time shift first, followed by time scaling, results in x(a(t+b)) which is equivalent to x(at+ab).",
                        "B": "Applying the time scaling first, followed by the time shift, results in x(at - b).",
                        "C": "Applying the time shift first, followed by time scaling, results in x(at - b).",
                        "D": "The order of operations does not matter when performing time scaling and time shifting."
                    },
                    "correct_answer": "B",
                    "ground_truth": "Signal Transformations: ** These operations modify the independent variable of a signal, such as time scaling (`x(at)`), time shifting (`x(t-b)`), and reflection (`x(-t)`), altering the signal's shape or position along its independent axis, with the order of transformations being critical for accurate representation."
                    },
                    {
                    "question": "Analyzing Figure 2.32 and 2.33, which of the following statements best describes a fundamental difference in how transformations are applied to the independent variable, particularly concerning the effect of the order of operations?",
                    "options": {
                        "A": "Scaling the independent variable by 'a' and then shifting by 'b' (x(at - b)) is equivalent to shifting by 'b/a' and then scaling by 'a' (x(a(t - b/a))).",
                        "B": "In transformations like x(at + b), the scaling factor 'a' applies only to the original time variable 't' before any shift is considered, hence the structure x(a(t) + b).",
                        "C": "Reflecting the signal about the y-axis (x(-t)) and then time shifting by 'b' (x(-(t-b))) is identical to time shifting by 'b' and then reflecting (x(-(t+b))).",
                        "D": "The interpretation of 'x(at + b)' implies that 'a' scales the shifted time, meaning the shift is effectively by 'b/a' in the unscaled time domain."
                    },
                    "correct_answer": "D",
                    "ground_truth": "Signal Transformations: ** These operations modify the independent variable of a signal, such as time scaling (`x(at)`), time shifting (`x(t-b)`), and reflection (`x(-t)`), altering the signal's shape or position along its independent axis, with the order of transformations being critical for accurate representation."
                    },
                    {
                    "question": "Given a real signal $x(t)$, which of the following statements most accurately describes the relationship between its even component, $x_e(t)$, and its odd component, $x_o(t)$, when considering their behavior for positive and negative time intervals?",
                    "choices": {
                        "A": "$x_e(t)$ is necessarily an even function, and $x_o(t)$ is necessarily an odd function, meaning $x_e(t) = x_e(-t)$ and $x_o(t) = -x_o(-t)$ respectively, for all $t$.",
                        "B": "For any $t > 0$, $x_e(t)$ will have the same value as $x_o(-t)$, and $x_o(t)$ will have the same value as $-x_e(-t)$.",
                        "C": "$x_e(t)$ reflects the symmetry of $x(t)$ about the y-axis, while $x_o(t)$ captures the anti-symmetry of $x(t)$ about the origin, such that $x_e(t)$ is identical for $t$ and $-t$, and $x_o(t)$ is opposite for $t$ and $-t$.",
                        "D": "If $x(t)$ is an even signal, then $x_o(t)$ must be identically zero for all $t$, and if $x(t)$ is an odd signal, then $x_e(t)$ must be identically zero for all $t$."
                    },
                    "correct_answer": "C",
                    "ground_truth": "Even and Odd Signals: ** Real signals can be classified based on their symmetry: an even signal satisfies $x(t) = x(-t)$ (symmetric about the y-axis), while an odd signal satisfies $x(t) = -x(-t)$ (symmetric about the origin after reflection), and any real signal can be uniquely decomposed into its even and odd components."
                    },
                    {
                    "question": "Consider a real signal $x(t)$ that can be decomposed into its even and odd components, $x_e(t)$ and $x_o(t)$, such that $x(t) = x_e(t) + x_o(t)$. If we are given the values of $x(t)$ for $t > 0$, what is the minimum information required to fully reconstruct both $x_e(t)$ and $x_o(t)$ for all $t$?",
                    "choices": {
                        "A": "The values of $x(t)$ for $t > 0$ are sufficient, as this implicitly defines the symmetry properties needed for reconstruction.",
                        "B": "The values of $x_e(t)$ for $t > 0$ and $x_o(t)$ for $t < 0$ are needed to determine their respective behaviors across the entire time axis.",
                        "C": "The values of $x(t)$ for $t > 0$ and the fact that $x(0)=0$ are crucial to determine the origin's behavior, which impacts both components.",
                        "D": "The values of $x(t)$ for $t > 0$ and its integral over $t>0$ are necessary to infer the symmetry and antisymmetry over all time."
                    },
                    "correct_answer": "D",
                    "ground_truth": "Even and Odd Signals: ** Real signals can be classified based on their symmetry: an even signal satisfies $x(t) = x(-t)$ (symmetric about the y-axis), while an odd signal satisfies $x(t) = -x(-t)$ (symmetric about the origin after reflection), and any real signal can be uniquely decomposed into its even and odd components."
                    }
                ]
}

        return temp

    

if __name__ == "__main__":
    factory = MCQPipelineFactory(".assets/notes.pdf")
    factory.run_pipeline()

    

