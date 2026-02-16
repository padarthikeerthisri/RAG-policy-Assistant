# RAG Policy Assistant

## 📌 Objective

This project implements a small **Retrieval-Augmented Generation (RAG)** system to answer questions about company policy documents.

The goal is to demonstrate:

* Prompt engineering and iteration
* Grounded LLM responses using retrieved context
* Hallucination avoidance
* Clear evaluation and reasoning

The system answers only from the provided policy documents and gracefully handles missing or out-of-scope queries.

---

## 🧩 Problem Overview

Given a set of company policy documents (e.g., Refund, Cancellation, Shipping, Payment), the system:

* Retrieves relevant policy content using semantic search
* Passes retrieved context to a local LLM
* Produces accurate, factual, non-hallucinated answers
* Explicitly refuses opinions, paraphrasing, and conversation-history questions

---

## 🏗️ Architecture Overview

```
User Question
      ↓
FastAPI Backend
      ↓
Embedding Model (Sentence Transformers)
      ↓
Vector Store (ChromaDB)
      ↓
Top-K Relevant Chunks
      ↓
Local LLM (Ollama)
      ↓
Final Answer
```

The system is **stateless** and does not retain conversation history.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Backend:** FastAPI
* **LLM:** Ollama (local, open-source)
* **Embeddings:** all-MiniLM-L6-v2
* **Vector Store:** ChromaDB
* **Frontend:** Minimal chat-style UI (HTML/CSS/JS)
* **Deployment:** Local server exposed via ngrok (demo only)

---

## 📁 Project Structure

```
rag-policy-assistant/
│
├── app.py                # FastAPI application
├── main.py               # RAG pipeline logic
├── requirements.txt
├── README.md
│
├── data/
│   └── policies/         # Policy documents (.txt)
│
├── prompts/
│   ├── prompt_v1_initial.txt
│   └── prompt_v2_improved.txt
│
├── responses/
│   ├── responses_v1_initial.txt
│   └── responses_v2_improved.txt
│
├── chroma_db/            # Vector DB persistence
│
└── templates/
    └── index.html        # Chat UI
```

---

## 📄 Data Preparation & Chunking

### Chunking Strategy

* Chunk size: **500 characters**
* Chunk overlap: **100 characters**

### Rationale

* Policy documents contain short, self-contained rules
* 500 characters preserves semantic completeness
* Overlap prevents boundary information loss
* Smaller chunks improve retrieval precision

Each chunk is augmented with its policy type to improve retrieval clarity.

---

## 🔗 RAG Pipeline

1. Load policy documents (.txt)
2. Chunk documents using recursive splitting
3. Generate embeddings using Sentence Transformers
4. Store embeddings in ChromaDB
5. Retrieve top-k relevant chunks per query
6. Inject retrieved context into a structured prompt
7. Generate response using a local LLM (Ollama)

---

## ✍️ Prompt Engineering

Prompt engineering was iterated to reduce hallucinations and enforce strict grounding.

### Prompt Versions

**Version 1 – Initial**

* Allowed opinions
* Inconsistent policy listing
* Occasional hallucinations

**Version 2 – Improved**

* Context-only answers
* Deterministic refusals
* No conversation memory
* No paraphrasing or opinions

---

## 🧪 Evaluation

### Sample Evaluation Questions

| Question                               | Expected Behavior      |
| -------------------------------------- | ---------------------- |
| What policies do you know about?       | List all policies      |
| Are digital products refundable?       | Answer from policy     |
| Can I cancel after shipping?           | Partial factual answer |
| What do you think about refund policy? | Refusal                |
| What was my previous question?         | Refusal                |
| Who is the CEO?                        | “I don’t know”         |

### Results

| Criterion               | Result        |
| ----------------------- | ------------- |
| Accuracy                | High          |
| Hallucination Avoidance | Strong        |
| Answer Clarity          | Clear         |
| Edge Case Handling      | Deterministic |

---

## 🚨 Edge Case Handling

The system explicitly handles:

* **No relevant documents**
  → “I don’t know based on the provided documents.”

* **Opinion / feedback questions**
  → Refused deterministically

* **Paraphrasing requests**
  → Refused

* **Conversation-history queries**
  → Refused (stateless system)

This behavior is enforced via prompt design, not hard-coded logic.

---

## 🌍 Running the Project

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Start Ollama

```
ollama run qwen2.5:0.5b
```

### 3️⃣ Start Server

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4️⃣ Access UI

```
http://127.0.0.1:8000
```

API Docs:

```
http://127.0.0.1:8000/docs
```

---

## ⚖️ Trade-offs & Future Improvements

### Trade-offs

* Lightweight embeddings → Faster, lower recall
* No reranking → Simpler pipeline
* Local LLM → No external API dependency

### Future Improvements

* Cross-encoder reranking
* Citation highlighting
* Automated evaluation
* Cloud deployment
* JSON schema validation

---
# RAG-policy-Assistant
