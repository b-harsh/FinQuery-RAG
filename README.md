# AI-Powered Financial Research Engine

A high-performance RAG (Retrieval-Augmented Generation) system built to analyze complex financial documents (10-K reports, research papers) with high accuracy and zero hallucination.

<img width="1397" height="760" alt="image" src="https://github.com/user-attachments/assets/0435c3a7-39ed-4a5f-bb23-f09b27da2e09" />


### Key Features

- **Hybrid RAG Pipeline:** Uses local HuggingFace embeddings (`all-MiniLM-l6-v2`) to minimize API costs and latency.
- **Lightning Fast Inference:** Integrated with **Groq (Llama-3)** for near-instant responses.
- **Context-Aware Retrieval:** Implements recursive chunking to maintain the integrity of financial tables and data.
- **Semantic Search:** Moves beyond keyword matching to understand financial concepts (e.g., "liquidity" vs "cash flow").

### Tech Stack

- **Orchestration:** LangChain (LCEL)
- **LLM:** Groq (Llama-3) / OpenAI (GPT-4o-mini)
- **Vector Store:** FAISS (Local)
- **Embeddings:** HuggingFace (Local)
- **Language:** Python 3.14+

### Quick Start

1. Clone the repo: `git clone [<your-repo-link>](https://github.com/b-harsh/FinQuery-RAG)`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GROQ_API_KEY` to a `.env` file.
4. Place your PDF as `report.pdf` in the root can delete if any existing already.
5. Run: `python main.py`
