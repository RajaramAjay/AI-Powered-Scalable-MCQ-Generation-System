# src/llm_agent.py
# This file implements the LLMInvoke class to interact with a Generative AI model
import google.generativeai as genai
from dotenv import load_dotenv
import os
from src.utils import get_logger    

logger = get_logger(__name__)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")  # Provide your Gemini API KEY
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully.")

class LLMInvoke:
    """
    Handles invoking a Generative AI model and fetching responses.
    """
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(model_name=self.model_name)
            logger.info("Initialized LLMInvoke with model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize LLM model %s: %s", self.model_name, e, exc_info=True)
            raise

    def llm_response(self, prompt, context=None):
        """
        Generates a response from the model for a given prompt.
        Returns a dictionary with the answer or an error message.
        """
        logger.debug("Generating response for prompt: %s", prompt[:100])  # log first 100 chars
        try:
            response = self.model.generate_content(prompt)
            logger.info("Generated LLM response successfully.")
            return {"answer": response.text}
        except Exception as e:
            logger.error("Error generating LLM response: %s", e, exc_info=True)
            return {"answer": f"Error processing query: {str(e)}"}


# Example usage
if __name__ == "__main__":
    llm_invoke = LLMInvoke()
    example_query = "LLM Quantization means?"
    result = llm_invoke.llm_response(example_query)
    logger.info("LLM response: %s", result)
