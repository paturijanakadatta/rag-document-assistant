# RAG Document Assistant
### Intelligent Document Search & Q&A using Retrieval-Augmented Generation

> Built to solve a real problem: engineers spending 45+ minutes manually searching across mixed internal documentation — runbooks, process manuals, technical specs, and operational guides — with keyword-based search that had no semantic understanding.

This tool reduced average document query time from **~45 minutes to under 3 minutes** (~93% reduction) by combining semantic vector search with GPT-4 response generation, deployed as an internal Streamlit application for an observability engineering team.

---

## The Problem

| Pain Point | Impact |
|---|---|
| Manual search across large mixed-doc knowledge bases | ~45 min per query |
| Keyword search failing on context-dependent questions | Wrong or no results |
| Engineers context-switching during live incident investigations | Slower MTTD |
| No single interface across document types | Fragmented knowledge |

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG DOCUMENT ASSISTANT                       │
└─────────────────────────────────────────────────────────────────────┘

  INGESTION PIPELINE                        QUERY PIPELINE
  ─────────────────                         ──────────────

  ┌──────────────┐                          ┌──────────────┐
  │  Raw Docs    │                          │  User Query  │
  │ (PDF, DOCX,  │                          │  (Streamlit) │
  │  TXT, MD)    │                          └──────┬───────┘
  └──────┬───────┘                                 │
         │                                         ▼
         ▼                                  ┌──────────────┐
  ┌──────────────┐                          │   Embed      │
  │   Chunking   │                          │  User Query  │
  │ 512 tokens   │                          │ (ST Model)   │
  │ 64 overlap   │                          └──────┬───────┘
  └──────┬───────┘                                 │
         │                                         ▼
         ▼                                  ┌──────────────┐
  ┌──────────────┐    ┌──────────────┐      │ FAISS Vector │
  │  Embeddings  │───▶│ FAISS Index  │◀─────│   Search     │
  │  (Sentence   │    │  (Persisted) │      │  (Top-k=5)   │
  │ Transformers)│    └──────────────┘      └──────┬───────┘
  └──────────────┘                                 │
                                                   ▼
                                          ┌──────────────────┐
                                          │  Retrieved Chunks │
                                          │  (Cosine Sim.)   │
                                          └──────┬───────────┘
                                                 │
                                                 ▼
                                          ┌──────────────────┐
                                          │   Prompt Builder  │
                                          │  (Context +       │
                                          │   System Prompt)  │
                                          └──────┬───────────┘
                                                 │
                                                 ▼
                                          ┌──────────────────┐
                                          │   GPT-4 / 3.5    │
                                          │   (OpenAI API)   │
                                          └──────┬───────────┘
                                                 │
                                                 ▼
                                          ┌──────────────────┐
                                          │  Final Response  │
                                          │  (Streamlit UI)  │
                                          └──────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo / GPT-4 |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Orchestration | LangChain |
| UI | Streamlit |
| Language | Python 3.10+ |
| Document Parsing | PyMuPDF, python-docx, markdown |

---

## Key Design Decisions

### Chunking Strategy
- **Chunk size: 512 tokens** — balances sufficient context per chunk with retrieval precision
- **Overlap: 64 tokens** — prevents context loss at chunk boundaries, especially important for multi-step reasoning in runbooks
- Tested chunk sizes from 256–1024; 512 gave best retrieval precision on internal evaluation

### Retrieval
- **FAISS flat index** with cosine similarity — chosen for speed and accuracy on knowledge base sizes up to ~50k chunks
- **Top-k = 5** retrieved chunks passed to GPT-4 context window
- Cosine similarity threshold filtering (> 0.35) applied to prevent irrelevant chunks from polluting the prompt

### Prompt Engineering
Three specialised prompt templates:
- `qa_prompt` — direct factual Q&A with source grounding instructions
- `summary_prompt` — document/section summarisation with length control
- `decision_prompt` — decision-support queries requiring comparison across documents

All templates include explicit anti-hallucination instructions: *"Answer only from the provided context. If the answer is not in the context, say so clearly."*

---

## Project Structure

```
rag-document-assistant/
│
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
│
├── ingestion/
│   ├── loader.py           # Document loading (PDF, DOCX, TXT, MD)
│   ├── chunker.py          # Text chunking with overlap
│   └── embedder.py         # Embedding generation + FAISS index builder
│
├── retrieval/
│   ├── vector_store.py     # FAISS index load + similarity search
│   └── retriever.py        # Top-k retrieval with threshold filtering
│
├── generation/
│   ├── prompts.py          # Prompt templates (QA, summary, decision)
│   └── generator.py        # OpenAI API call + response handler
│
├── evaluation/
│   └── eval.py             # Retrieval precision + response quality checks
│
└── docs/
    ├── architecture.png    # Architecture diagram
    └── demo.gif            # (optional) UI walkthrough
```

---

## Features

- **Semantic search** — finds relevant documents even when the exact keywords aren't used
- **Multi-document type support** — ingests PDF, DOCX, TXT, and Markdown files
- **Three query modes** — Q&A, summarisation, decision support
- **Source attribution** — every response shows which document chunks were used
- **Anti-hallucination guardrails** — LLM is explicitly grounded to retrieved context only
- **Fast retrieval** — FAISS index enables millisecond-level vector search at scale
- **Simple UI** — Streamlit interface requires no specialist tooling to use

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/janaka-datta-paturi/rag-document-assistant.git
cd rag-document-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# 5. Add your documents to the /data folder
mkdir data
# Copy your PDF / DOCX / TXT / MD files into /data

# 6. Run ingestion to build the FAISS index
python ingestion/embedder.py

# 7. Launch the Streamlit app
streamlit run app.py
```

---

## Requirements

```
openai>=1.0.0
langchain>=0.1.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
streamlit>=1.30.0
pymupdf>=1.23.0
python-docx>=1.0.0
tiktoken>=0.5.0
python-dotenv>=1.0.0
```

---

## Results & Impact

| Metric | Before | After |
|---|---|---|
| Avg. document query time | ~45 minutes | ~3 minutes |
| Search success rate on context-dependent queries | Low (keyword mismatch) | High (semantic match) |
| Engineer adoption | Manual search only | Team-wide within 1 week |
| Query types supported | Keyword lookup | Q&A, summarisation, decision support |

---

## Evaluation Methodology

System performance was evaluated across 3 iteration cycles:

1. **Retrieval precision** — manually verified that the top-5 retrieved chunks were relevant to the query for a set of 30 representative test questions
2. **Response relevance** — rated LLM responses on a 1–5 scale for factual accuracy and grounding to retrieved context
3. **Iterative tuning** — chunk size, overlap, top-k, and similarity threshold were adjusted between cycles based on observed failure modes

---

## Author

**Janaka Datta Paturi** — Data Scientist & ML Engineer  
[LinkedIn](https://linkedin.com/in/janaka-datta-paturi)

---

## License

This project is for portfolio and demonstration purposes.
