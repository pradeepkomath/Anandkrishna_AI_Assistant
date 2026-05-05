# 🏨 Anandakrishna Residency AI Assistant

## 📌 Overview

An AI-powered customer support assistant built using **CrewAI, LangChain, and Streamlit** to handle queries related to:

* Room tariffs and pricing
* Hotel facilities
* Guruvayoor Temple information
* Room availability (mock logic)
* Booking and escalation requests

The system integrates **RAG (Retrieval-Augmented Generation)**, **tool-based reasoning**, **short-term memory**, and **adaptive behaviour based on user feedback**.

---

## 🚀 Features

### 🔹 Intelligent Query Handling

* Uses **CrewAI agents** to route queries to appropriate tools
* Supports tariff lookup, informational queries, and availability checks

### 🔹 RAG-based Knowledge Retrieval

* Uses **FAISS vector store** for document retrieval
* Provides context-aware answers

### 🔹 Short-Term Memory (Phase 6)

* Maintains last 5 interactions
* Supports multi-turn conversations
* Includes memory reset functionality

### 🔹 Adaptive Behaviour (Phase 7)

* Collects user feedback (Helpful / Not Helpful)
* Adjusts response tone based on negative feedback
* Stores feedback in CSV logs

### 🔹 Streamlit UI (Phase 8)

* Chat-based interface
* Feedback buttons
* Debug trace visualization (optional)

### 🔹 Logging & Monitoring

* Interaction logs stored in `crew_agent_logs.csv`
* Feedback logs stored in `feedback_logs.csv`

### 🔹 Graceful Error Handling

* Handles API failures and file errors
* Displays user-friendly messages
* Technical errors shown only in debug mode

---

## 🗂 Project Structure

```
Phase8_UI_N_Packaging/
│
├── app.py                  # Streamlit UI
├── crew_agent.py           # CrewAI logic
├── tools.py                # Tool implementations
├── ingest.py               # Vector DB creation
├── requirements.txt        # Dependencies
├── .env.example            # Environment template
├── .streamlit/
│   └── config.toml         # Streamlit config
├── vectorstore/            # FAISS index
├── crew_agent_logs.csv     # Interaction logs
├── feedback_logs.csv       # Feedback logs
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Configure environment variables

Create `.env` file from template:

```bash
copy .env.example .env
```

Update `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
FAISS_INDEX_PATH=vectorstore/faiss_index
TOP_K=4
```

---

### 3️⃣ Run the application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 Debug Mode

* Enables visibility of **agent reasoning and tool usage**
* Displays execution trace in UI

---

## 🧠 Tools Used

| Tool                    | Purpose                |
| ----------------------- | ---------------------- |
| room_tariff_lookup_tool | Pricing & room details |
| rag_knowledge_tool      | Informational queries  |
| availability_check_tool | Mock availability      |
| human_escalation_tool   | Escalation handling    |

---

## 📊 Logs

* `crew_agent_logs.csv` → Stores user queries & responses
* `feedback_logs.csv` → Stores feedback for adaptive behaviour

---

## ⚠️ Limitations

* Availability check uses **mock logic** (no real backend)
* No real booking or transaction support
* Requires valid OpenAI API key
* FAISS index must be generated before use

---

## 🔒 Error Handling

* API failures handled gracefully
* File access errors (e.g., locked logs) managed safely
* User sees friendly messages instead of technical errors

---

## 📦 Deployment Readiness (Phase 8)

* Dependency management via `requirements.txt`
* Environment configuration via `.env`
* Local deployment using Streamlit
* Logging and tracing enabled
* Graceful failure handling implemented

---

## 👨‍💻 Author

Capstone Project – Agentic AI (IIT Madras)

---
