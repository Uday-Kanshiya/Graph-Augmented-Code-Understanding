# 🔮 Graph-Augmented Code Understanding

An advanced local-first context optimization engine designed to analyze codebases, generate semantic dependency graphs, and optimize LLM query contexts for QA and code rectifications. 

Equipped with a premium dark-themed Streamlit interface, token-saving analytics, and interactive AST call-graph visualizations.

---

## ✨ Features

- **📂 Multi-Format Ingestion**: Upload zipped repositories locally or import direct GitHub URLs for analysis.
- **🌳 AST-Based CodeGraph**: Custom Python static analyzer mapping structural hierarchy (`contains`), compilation dependencies (`imports`), execution paths (`calls`), and inheritance chains (`inherits`).
- **🧬 Graphify Explorer**: Macro-level design concept mapping to focus LLM context on high-level system components.
- **⚡ PageRank Hybrid Retrieval**: Selects the optimal codebase neighborhood for queries, avoiding token-heavy "whole codebase" dumps.
- **📊 Premium Token Analytics**: Real-time tracking of net token savings against a whole-codebase baseline.
- **🛠️ Automated Code Rectifier**: Applies proposed AI fixes and patches directly to your workspace with automatic background re-indexing.
- **🎨 Premium Dark Theme**: Beautiful modern slate & indigo theme with responsive metrics cards, sleek typography, and high-contrast Graphviz layout maps.

---

## 🚀 Quick Start (Streamlit App)

The fastest path to run the dashboard and start asking queries is through the Streamlit app. It runs all backend analysis services locally.

### 1. Installation
Ensure you have Python 3.10+ installed:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Keys
To use Gemini, create a `.streamlit/secrets.toml` file from the example (or define a `.env` file):

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your-api-key-here"
GEMINI_MODEL = "gemini-2.5-flash"
```

### 3. Run App
```powershell
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ Full Development Setup

For running the complete client-server application (FastAPI + Next.js frontend):

### 1. Copy Environment File
```powershell
Copy-Item .env.example .env
```

### 2. Start FastAPI Backend
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Start Next.js Frontend
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000).

---

## 🗺️ Project Architecture

```text
├── backend/                  # FastAPI API Server
│   ├── app/api/              # REST Endpoints
│   ├── app/models/           # Pydantic schemas (Graph, Queries, Repo)
│   └── app/services/         # Ingestion, parsing, retrieval, token, LLM
├── frontend/                 # Next.js Web App
│   ├── app/                  # App Router & page views
│   └── components/           # UI Elements & components
├── .streamlit/               # Streamlit config & secrets
├── streamlit_app.py          # Streamlit UI Dashboard
└── data/                     # Local file and database state storage
```

---

## 🧠 Technical Highlights

* **Tree-Sitter Parsing**: Multi-language parsing (Python, JS/TS, Go, Rust, Java, C/C++) to generate concrete syntax trees.
* **Neighborhood Maps**: Graphical context subgraphs rendered natively using Graphviz, highlighting primary query anchors vs. neighbor signatures.
* **Token Tracking**: Uses `tiktoken` locally and Gemini usage metadata to verify exact token metrics and showcase prompt size reductions.
* **Workspaces Auto-Sync**: The local file structure is synchronized upon any LLM-directed file rectification.
