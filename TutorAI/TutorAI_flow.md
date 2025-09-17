# TutorAI — End-to-End RAG Flow

> **Project:** TutorAI (RAG)  
> **Source of Truth:** *Storytelling with Data*  
> **Vector Store:** FAISS  
> **Embeddings:** `all-MiniLM-L6-v2` (normalized)  
> **Frontend:** Gradio

This document summarizes the full application pipeline, extracted from the HTML system flow  and aligned with the modularized Python package (`tutorai/`) and notebook (`TutorAI_RAG.ipynb`).

---

## High-Level Data Flow

![High-Level Flow Diagram](tutorai_flow.html)

1. **Utilities**
   - `process_book_url`
   - `download_pdf` → produces `pdf_bytes`
   - `read_pdf_pages` → list of `[page_number, text]`

2. **Chunking & Embeddings**
   - Uses `RecursiveCharacterTextSplitter`
   - Embedding model: `all-MiniLM-L6-v2` (see `utils.py`, `config.py`)
   - Documents tagged with page metadata

3. **Vector Store**
   - FAISS index built from chunks (`rag.py`)
   - Persistent local storage / loading

4. **Dashboard Controls**
   - Parameters exposed to UI (`ui_cli.py`, or extended to Gradio):
     - Model selection
     - Top-K retrieval
     - Temperature
     - Similarity cutoff

---

## Conversation Pipeline

- **Intent routing**  
  - *Chit-chat* → bypass RAG, no citations  
  - *Book_QA* → retrieve top-K, apply similarity cutoff

- **Steps**  
  - Retrieve relevant chunks (FAISS)  
  - Assemble context and user prompt (`rag.py::_build_prompt`)  
  - Generate answer with backend model (`generator_backend.py`)  
  - Apply guardrails (citations, refusal rules)

---

## Prompt Assembly

Prompt structure is built dynamically from system rules, chat history, user query, and retrieved context:

```text
<system>
You are TutorAI. Answer only from the provided book context.
If answer not in context, refuse. Include page citations like (p. N).
{system_description}

<history>
...prior user/assistant turns...

<user>
User query:
{user_query}

Book context (use ONLY this; cite every claim):
{context}

Guidelines:
{guidelines}
</user>
```

---

## Key Conditions (Routing & Guardrails)

- **Intent:**  
  - `chit_chat` → skip retrieval and citations  
  - `book_qa` → retrieval required; must cite

- **Similarity cutoff:**  
  - If all blocks filtered out → send empty context  
  - Model must refuse with a safe response

- **Guardrails:**  
  - If citations missing, append from retrieved blocks  
  - Enforce refusal if context is empty

- **History:**  
  - `ConversationBufferMemory` injects prior turns for continuity

**Success Criteria**
- Factual answers **must** include citations.  
- Chit-chat/meta answers **must not** include citations.  
- Empty context **must** yield refusal.

---

## Data Contracts

### Retrieved Block

```json
{
  "text": "...chunk content...",
  "page_number": 42,
  "score": 0.37,
  "metadata": {"chunk_id": "book|p42|c3", "source_id": "..."},
  "citation": "(p. 42)"
}
```

### Driver Response

```json
{
  "model": "phi-3-mini-4k-instruct",
  "answer": "...final text...",
  "citations_pages": [12, 42, 45],
  "retrieved_context": [...],
  "history_len": 7
}
```

---

## Typical Paths

- **Book_QA (happy path)**  
  1. User asks a question.  
  2. Intent: `book_qa` → retrieve K chunks → cutoff.  
  3. Prompt assembled → LLM generates answer.  
  4. Guardrail ensures citations.

- **Chit-Chat**  
  1. User says "hi".  
  2. Intent: `chit_chat`.  
  3. Retrieval bypassed → direct model reply.

- **No Relevant Context**  
  1. Query outside the book or cutoff too strict.  
  2. Context empty → model returns refusal.

---

## Inputs & Outputs

- **Inputs**
  - User question
  - Controls: top-K, temperature, similarity cutoff, model choice

- **Outputs**
  - Answer text
  - Citations: page numbers from retrieved blocks

---

## Present This Slide

Use this document as a **single-slide overview**.  
The diagram (`tutorai_flow.html`) visualizes the pipeline; sections here provide rules, prompt structure, and I/O schemas. This ties directly to:
- `TutorAI_RAG.ipynb` (original development notebook)
- `tutorai/` package modules (utilities, RAG, generator backend, UI)
- The generated HTML flow page (`tutorai_flow.html`)

---
