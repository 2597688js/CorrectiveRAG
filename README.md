# Corrective RAG (CRAG)

This project is an implementation of **Corrective Retrieval-Augmented Generation (CRAG)** using:

* LangGraph
* OpenAI
* FAISS
* Tavily

The system evaluates retrieved documents, decides whether retrieval is sufficient, and performs web search when additional information is needed.

## Environment Variables

Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run from CLI

```bash
uv run python code/main.py \
  --pdf \
  "code/documents/LoRA.pdf" \
  "code/documents/QLoRA.pdf" \
  --question "What is LoRA?"
```

## Run Streamlit UI

```bash
uv run streamlit run code/app.py
```

Upload one or more PDF documents and ask questions about their contents.
