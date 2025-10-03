# Scalable AI-Powered Question Generation System (MVP)

This project is a **minimum viable product (MVP)** of a scalable pipeline that transforms raw documents (e.g., PDFs, text) into structured **Multiple-Choice Questions (MCQs)** for education, training, and assessments.

It leverages **LLMs** (Gemini / OpenAI / Anthropic), **Map-Reduce summarization**, vector databases, and embeddings to ensure **efficient document processing, concept extraction, and automated question generation**.

---

## 🚀 Features

* Extracts text from large documents (PDF/text).
* **Map-Reduce Summarization**: breaks down large content into chunks (Map step), summarizes each chunk individually, and combines the summaries into a coherent overall summary (Reduce step).
* Splits and preprocesses content using **LangChain RecursiveCharacterTextSplitter**.
* Embeds document chunks with **HuggingFace embeddings / SentenceTransformers**.
* Stores embeddings in **FAISS vector store** for efficient retrieval.
* Generates **MCQs** from the extracted knowledge using **LLMs (Gemini 1.5 Flash tested)**.
* Supports parallelism via **ThreadPoolExecutor** for scalability.

---

## 🛠 Tools & Requirements

**Language:** Python 3.9+

**LLM API:** Gemini API — can also use OpenAI / Anthropic

### Dependencies

```
langchain
langchain-community
pymupdf
google-generativeai
python-dotenv
tiktoken
faiss-cpu
sentence-transformers
```

---

## ⚙️ Pipeline Overview

1. **Document Loading** – Extract text from PDF/text files.
2. **Text Chunking** – Split into manageable segments with LangChain splitters.
3. **Map-Reduce Summarization** – Generate intermediate summaries for each chunk, then merge into a final condensed summary.
4. **Embeddings Generation** – Convert text chunks into vector embeddings.
5. **Vector Store (FAISS)** – Store and index embeddings for retrieval.
6. **LLM Question Generation** – Generate multiple-choice questions from summaries & content.
7. **Output JSON** – Store structured questions in JSON format.

---

## 📂 Project Structure

```
MVP.ipynb        # Jupyter Notebook with full pipeline
.env             # Store API keys (e.g., Gemini, OpenAI)
requirements.txt # Dependencies list
output.json      # Generated MCQs (example)
```

---

## ▶️ Usage

1. Clone the repo:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Add your API key in `.env`:

   ```
   GEMINI_API_KEY=your_key_here
   ```
4. Run the notebook:

   ```bash
   jupyter notebook MVP.ipynb
   ```
5. Check `output.json` for generated MCQs.

---

## 📌 Notes

* The MVP uses **Gemini 1.5 Flash** as the primary LLM.
* Modular design allows easy switching to **OpenAI** or **Anthropic** APIs.
* **Map-Reduce summarizer** ensures scalability for large documents where a single LLM call would exceed context limits.
* Scaling can be achieved with **distributed workers** and **larger vector DBs**.

## Sources

* [Savaal Documentation](https://dspace.mit.edu/bitstream/handle/1721.1/162563/chandler-jchand-meng-eecs-2025-thesis.pdf?sequence=1&isAllowed=y)
* [Savaal: Scalable Concept-Driven Question Generation to Enhance Human Learning](https://arxiv.org/abs/2502.12477)
