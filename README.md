# rag-eval

A RAG (Retrieval-Augmented Generation) pipeline built with LlamaIndex that loads documents, chunks them, embeds them, and answers questions grounded strictly in the source material.


## What it does

- Loads PDF, Word, and HTML documents
- Chunks documents with configurable size and overlap
- Embeds chunks using OpenAI embeddings
- Persists the index to disk so embeddings aren't regenerated on every run
- Answers questions using only the context in your documents
- Refuses to answer questions not found in the documents

## Tech stack

- [LlamaIndex](https://docs.llamaindex.ai) — document loading, chunking, indexing, retrieval
- [OpenAI](https://platform.openai.com) — embeddings and LLM
- [uv](https://github.com/astral-sh/uv) — package management

## Setup

1. Clone the repo
2. Install dependencies with uv:
```bash
   uv sync
```
3. Create a `.env` file with your OpenAI key:
OPENAI_API_KEY=your-key-here

4. Add a PDF to the project folder and update the filename in `main.py`

## Usage

```bash
uv run main.py
```

Then type any question about your document. Type `quit` to exit.

## Next steps

- RAGAS evaluation for retrieval quality metrics
- LangSmith tracing for production observability
- Streamlit dashboard for visualization

## Evaluation

RAG quality is measured using [RAGAS](https://docs.ragas.io), an open-source evaluation framework for retrieval-augmented generation pipelines.

Run the evaluation:
```bash
uv run eval.py
```

**Metrics:**
- **Faithfulness** — measures whether answers are grounded in the retrieved context. A score of 1.0 means no hallucination detected.
- **Answer Relevancy** — measures how relevant the answer is to the question asked.

**Results on DiffMAS paper:**
| Metric | Score |
|---|---|
| Faithfulness | 1.0 |
| Answer Relevancy | 0.95 |