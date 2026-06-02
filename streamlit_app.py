from __future__ import annotations

import os
import shutil
import sys
import re
import difflib
import hashlib
from pathlib import Path
from uuid import uuid4

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import GraphDocument, QueryRecord, RepoMetadata  # noqa: E402
from app.services.analysis_pipeline import AnalysisPipeline  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.codegraph_service import CodeGraphService  # noqa: E402
from app.services.file_utils import clean_repo_name, safe_extract_zip  # noqa: E402
from app.services.graph_retrieval_service import GraphRetrievalService  # noqa: E402
from app.services.graphify_service import GraphifyService  # noqa: E402
from app.services.llm.gemini import GeminiProvider  # noqa: E402
from app.services.repo_service import RepoService  # noqa: E402
from app.services.storage import LocalStorage  # noqa: E402
from app.services.token_service import TokenService  # noqa: E402
from app.services.tree_sitter_service import TreeSitterService  # noqa: E402


st.set_page_config(
    page_title="Context Optimization Engine",
    layout="wide",
)

# Inject custom modern dark UI styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Set base font-family on application wrapper without forcing it on icon fonts */
    .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Target only text containers specifically to keep them stylized but prevent breaking icon fonts */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    h1, h2, h3, h4, h5, h6,
    .gradient-text,
    div[data-testid="stMetricValue"] *, 
    div[data-testid="stMetricLabel"] *,
    div.stButton > button,
    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Title and gradients */
    .gradient-text {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Metrics Card styling */
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15) !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        color: #818cf8 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #9ca3af !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Sleek buttons styling */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #f3f4f6 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Primary buttons gradient */
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Modern Tabs */
    div[data-baseweb="tab-list"] {
        gap: 12px !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.06) !important;
        padding-bottom: 8px !important;
        margin-bottom: 24px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #9ca3af !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #f3f4f6 !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #a5b4fc !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #6366f1 !important;
        height: 3px !important;
        border-radius: 3px !important;
    }
    
    /* Selectboxes, Text Inputs, Text Areas */
    div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #f9fafb !important;
    }
    
    /* Dataframe table border and header radius */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background-color: #111827 !important;
    }
    
    /* Main area styling */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Expanders styling */
    div[data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        margin-bottom: 16px !important;
    }
    div[data-testid="stExpander"] > details {
        border: none !important;
    }
    div[data-testid="stExpander"] summary {
        color: #e5e7eb !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #6366f1 !important;
    }
    
    /* Code block styling */
    div[data-testid="stCodeBlock"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def get_secret(name: str, default: str | None = None) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    local_secrets = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    home_secrets = Path.home() / ".streamlit" / "secrets.toml"
    if not local_secrets.exists() and not home_secrets.exists():
        return default
    try:
        value = st.secrets.get(name)
        return str(value) if value else default
    except Exception:
        return default


@st.cache_resource
def services():
    data_dir_value = get_secret("CONTEXT_ENGINE_DATA_DIR") or os.getenv("CONTEXT_ENGINE_DATA_DIR")
    data_dir = Path(data_dir_value) if data_dir_value else PROJECT_ROOT / "data"
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    storage = LocalStorage(data_dir)
    token_service = TokenService()
    tree_sitter_service = TreeSitterService()
    codegraph_service = CodeGraphService()
    graphify_service = GraphifyService(storage=storage)
    pipeline = AnalysisPipeline(
        storage=storage,
        tree_sitter_service=tree_sitter_service,
        codegraph_service=codegraph_service,
        graphify_service=graphify_service,
        token_service=token_service,
    )
    repo_service = RepoService(storage=storage, analysis_pipeline=pipeline, max_upload_mb=200)
    llm_provider = GeminiProvider(
        api_key=get_secret("GEMINI_API_KEY"),
        model=get_secret("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash",
    )
    graph_retrieval_service = GraphRetrievalService(storage=storage, token_service=token_service)
    chat_service = ChatService(
        storage=storage,
        graph_retrieval_service=graph_retrieval_service,
        token_service=token_service,
        llm_provider=llm_provider,
        pipeline=pipeline,
    )

    return storage, pipeline, repo_service, chat_service, llm_provider


storage, pipeline, repo_service, chat_service, llm_provider = services()


def current_repo() -> RepoMetadata | None:
    repo_id = st.session_state.get("repo_id")
    if not repo_id:
        return None
    return storage.load_repo_metadata(repo_id)


def set_repo(repo: RepoMetadata) -> None:
    st.session_state.repo_id = repo.repo_id
    st.session_state.selected_file = None


def ingest_uploaded_zip(uploaded_file) -> RepoMetadata:
    repo_id = uuid4().hex
    repo_name = clean_repo_name(Path(uploaded_file.name).stem)
    upload_path = storage.uploads_dir / f"{repo_id}.zip"
    source_dir = storage.repo_source_dir(repo_id)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(uploaded_file.getbuffer())
    if source_dir.exists():
        shutil.rmtree(source_dir)
    safe_extract_zip(upload_path, source_dir)
    return pipeline.analyze_existing(name=repo_name, source_dir=source_dir, origin="upload", repo_id=repo_id)



def render_notice(message: str) -> None:
    if "Graphify" in message and "fallback" in message.lower():
        st.info(message)
        return
    st.warning(message)


def render_upload_import(repo: RepoMetadata | None) -> None:
    st.header("Upload Or Import")
    left, right = st.columns(2)
    with left:
        st.subheader("Upload zipped codebase")
        uploaded_file = st.file_uploader("Choose .zip file", type=["zip"])
        if st.button("Analyze upload", disabled=uploaded_file is None, type="primary"):
            with st.spinner("Extracting and analyzing repository..."):
                try:
                    repo = ingest_uploaded_zip(uploaded_file)
                    set_repo(repo)
                    st.success(f"Loaded {repo.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with right:
        st.subheader("Import GitHub URL")
        github_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
        if st.button("Clone and analyze", disabled=not github_url.strip()):
            with st.spinner("Cloning and analyzing repository..."):
                try:
                    repo = repo_service.import_github(github_url.strip())
                    set_repo(repo)
                    st.success(f"Loaded {repo.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.caption(
        "Supported languages: Python, JavaScript/TypeScript, Go, Rust, Java, and C/C++. "
        "The app automatically skips .git, virtual environments (.venv), node_modules, build outputs, and caches."
    )

    if repo:
        st.divider()
        st.subheader("Repository Summary")
        cols = st.columns(4)
        cols[0].metric("Total files", repo.stats.total_files)
        cols[1].metric("Python files", repo.stats.python_files)
        cols[2].metric("Total lines", repo.stats.total_lines)
        cols[3].metric("Python lines", repo.stats.python_lines)
        st.write(f"**Origin:** `{repo.origin}`")
        if repo.error:
            st.error(repo.error)
        
        # Display files by language in table format
        files = storage.load_files(repo.repo_id)
        if files:
            st.subheader("Files by Language")
            
            # Create a list of file data with path and language
            file_data = [
                {"File": file.path, "Language": file.language or "unknown"}
                for file in files
            ]
            
            # Display as table
            st.dataframe(file_data, use_container_width=True, hide_index=True)


def graph_to_dot(graph: GraphDocument, max_nodes: int = 80, max_edges: int = 160) -> str:
    visible_nodes = graph.nodes[:max_nodes]
    visible = {node.node_id for node in visible_nodes}
    lines = [
        "digraph G {", 
        "rankdir=LR;", 
        "bgcolor=transparent;",
        'node [shape=box, style="rounded,filled", fillcolor="#111827", color="#374151", fontcolor="#f9fafb", fontname="Plus Jakarta Sans", fontsize=10, penwidth=1.5];',
        'edge [color="#4b5563", fontcolor="#9ca3af", fontname="Plus Jakarta Sans", fontsize=8, penwidth=1.2];'
    ]
    for node in visible_nodes:
        label = f"[{node.node_type.upper()}]\\n{node.label}".replace('"', "'")
        lines.append(f'"{node.node_id}" [label="{label}"];')
    for edge in graph.edges:
        if edge.source_node in visible and edge.target_node in visible:
            label = edge.edge_type.replace('"', "'")
            lines.append(f'"{edge.source_node}" -> "{edge.target_node}" [label="{label}"];')
            max_edges -= 1
            if max_edges <= 0:
                break
    lines.append("}")
    return "\n".join(lines)


def render_graph_schematic(kind: str) -> None:
    st.markdown("---")
    st.subheader("🔮 Semantic Schema Guide")
    st.markdown(
        "This interactive blueprint explains the **graph model schema** and **metadata layers** "
        "captured in your graph database. This metadata is selectively loaded to optimize LLM query contexts."
    )
    
    if kind == "codegraph":
        t1, t2 = st.tabs(["📝 AST Node Types & Metadata", "🔗 AST Relationship Edges"])
        with t1:
            st.markdown(
                """
                | Node Type | Represents | Captured Information & Metadata Keys | Purpose in optimized QA |
                | :--- | :--- | :--- | :--- |
                | **`module`** | A file in the codebase (e.g. `.py`, `.ts`). | `file_path`, `total_lines`, `docstring_headers`, `imports` | Serves as the high-level anchor for structural organization. |
                | **`class`** | An OOP class definition. | `label` (class name), base classes, `line_start`, `line_end` | Defines data structures and component boundaries. |
                | **`function`** / **`method`** | Independent utility functions or class-bound methods. | `signature`, parameters, return type, docstrings, `source_snippet` | Holds the exact business logic and code execution body. |
                """
            )
            st.info(
                "💡 **How it saves tokens:** Instead of sending the full file, if a function is not matching the query, "
                "we only extract its **light signature** (signature + docstring header) in neighborhood traversal."
            )
        with t2:
            st.markdown(
                """
                | Edge Type | Connection | Meta Info Saved | Architectural Meaning |
                | :--- | :--- | :--- | :--- |
                | **`contains`** | `module ➔ class` or `class ➔ method` | Parent-child ownership, lexical nesting. | Map OOP structures and hierarchy. |
                | **`calls`** | `function ➔ function` or `method ➔ function` | Caller position, line number, frequency. | Traces dynamic execution paths and call graph. |
                | **`imports`** | `module ➔ module` | Imported entities, aliases, source line. | Maps compilation and module dependency chains. |
                | **`inherits`** | `class ➔ class` | Subclassing relations, base class names. | Identifies inheritance trees and behavioral overrides. |
                """
            )
    else: # graphify
        t1, t2 = st.tabs(["🏗️ Macro Design Nodes & Metadata", "🌊 Lifted Flow Edges"])
        with t1:
            st.markdown(
                """
                | Node Type | Represents | Captured Information & Metadata Keys | Purpose in optimized QA |
                | :--- | :--- | :--- | :--- |
                | **`file`** | A high-level module file. | `file_path`, dependency weights, import footprint | Maps structural macro organization. |
                | **`component`** | A macro-level class or concept design boundary. | Design role, interaction count, encapsulation level | Identifies core concepts and system boundaries. |
                """
            )
            st.info(
                "💡 **How it saves tokens:** Graphify **prunes all micro-nodes** (helper functions/methods) "
                "to keep the context focused on high-level system-level interactions and architecture."
            )
        with t2:
            st.markdown(
                """
                | Edge Type | Lifted Relation | Meta Info Saved | Architectural Meaning |
                | :--- | :--- | :--- | :--- |
                | **`part_of`** | `component ➔ file` | Structural containing, component ownership. | Maps component-to-file bundling. |
                | **`depends_on`** | `file ➔ file` | Compilation imports, import dependency trees. | Tracks high-level architectural layering. |
                | **`triggers`** | `component ➔ component` | Aggregated call flows, execution triggers. | Maps execution triggers across modular component bounds. |
                | **`extends`** | `component ➔ component` | Architectural inheritance, class expansions. | Tracks modular extension hierarchies. |
                """
            )
            
    with st.expander("🔬 View Raw JSON Database Schema Definitions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graph Node Schema Example:**")
            st.code(
                '''{
  "node_id": "codegraph:app/services/token_service.py:TokenService:estimate_tokens",
  "node_type": "method",
  "label": "estimate_tokens",
  "file_path": "backend/app/services/token_service.py",
  "line_start": 15,
  "line_end": 23,
  "source_snippet": "def estimate_tokens(self, text: str) -> int:...",
  "metadata": {
    "docstring_headers": ["Estimate tokens using tiktoken."],
    "parameters": ["self", "text"],
    "return_type": "int"
  }
}''',
                language="json"
            )
        with col2:
            st.markdown("**Graph Edge Schema Example:**")
            st.code(
                '''{
  "edge_id": "edge:token_service:calls:tiktoken",
  "edge_type": "calls",
  "source_node": "codegraph:app/services/token_service.py:TokenService:estimate_tokens",
  "target_node": "external:tiktoken:get_encoding",
  "score": 1.0,
  "metadata": {
    "line_number": 20,
    "alias": "tiktoken"
  }
}''',
                language="json"
            )


def render_graph(repo: RepoMetadata | None, kind: str) -> None:
    title = "CodeGraph Explorer" if kind == "codegraph" else "Graphify Explorer"
    st.header(title)
    if not repo:
        st.info("Load a repository first.")
        return
    graph = storage.load_codegraph(repo.repo_id) if kind == "codegraph" else storage.load_graphify(repo.repo_id)
    if not graph:
        st.error(f"{title} output not found.")
        return
    if graph.warnings:
        for warning in graph.warnings:
            render_notice(warning)
    if kind == "graphify" and graph.source == "graphify-fallback":
        st.info(
            "Native Graphify output is not available in this environment. This tab is showing the saved Graphify adapter output plus a clearly labeled fallback graph derived from CodeGraph."
        )
    cols = st.columns(4)
    cols[0].metric("Source", graph.source)
    cols[1].metric("Nodes", len(graph.nodes))
    cols[2].metric("Edges", len(graph.edges))
    cols[3].metric("Raw output", "saved" if graph.raw_output_path else "none")
    st.graphviz_chart(graph_to_dot(graph), use_container_width=True)
    with st.expander("Nodes"):
        st.dataframe([node.model_dump() for node in graph.nodes], use_container_width=True, hide_index=True)
    with st.expander("Edges"):
        st.dataframe([edge.model_dump() for edge in graph.edges], use_container_width=True, hide_index=True)
    if kind == "graphify" and graph.raw_output_path:
        with st.expander("Raw Graphify adapter output"):
            raw_path = Path(graph.raw_output_path)
            if raw_path.exists():
                raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
                st.code(raw_text[:16000], language="json")
                if len(raw_text) > 16000:
                    st.caption("Raw output truncated for display.")
            else:
                st.caption(f"Raw output path recorded but not found on disk: {graph.raw_output_path}")

    # Render semantic schema explanation view
    render_graph_schematic(kind)


def render_tokens(repo: RepoMetadata | None) -> None:
    st.header("Token Analytics")
    if not repo:
        st.info("Load a repository first.")
        return
        
    st.markdown(
        "### 🧠 LLM Prompt Token Analysis\n"
        "LLMs have strict **context window limits** and charge based on the number of input tokens. "
        "This dashboard tracks how **Graph-Optimized QA** successfully reduces the number of **LLM Prompt (input) tokens** "
        "compared to a **General Chatbot Baseline (Whole Codebase)** (sending 100% of all codebase lines/files directly to the LLM plus the query)."
    )

    # 1. Load all queries run so far
    import json
    queries_dir = storage.repo_state_dir(repo.repo_id) / "queries"
    queries = []
    if queries_dir.exists():
        for file_path in queries_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    queries.append(QueryRecord.model_validate(data))
            except Exception:
                pass
                
    # Sort queries by creation time
    queries = sorted(queries, key=lambda q: q.created_at)

    if queries:
        st.subheader("📊 Query-by-Query LLM Prompt Savings")
        
        # Build comparison rows against General Chatbot Baseline (Whole Codebase) for each graph_optimized query
        comparison_rows = []
        for q_rec in queries:
            if q_rec.mode != "graph_optimized":
                continue
            q_text = q_rec.query.strip()
            prompt_tokens_measurement = q_rec.token_usage.get("llm_prompt_tokens")
            prompt_tokens = prompt_tokens_measurement.tokens if prompt_tokens_measurement else 0
            
            baseline_measurement = q_rec.token_usage.get("whole_codebase_baseline")
            baseline_tokens = baseline_measurement.tokens if baseline_measurement else 0
            
            if baseline_tokens > 0:
                saved = baseline_tokens - prompt_tokens
                pct = (saved / baseline_tokens * 100) if baseline_tokens else 0
                comparison_rows.append({
                    "Query / Question": q_text,
                    "General Chatbot Baseline (Whole Codebase)": baseline_tokens,
                    "Graph-Optimized Prompt Tokens": prompt_tokens,
                    "Tokens Saved": saved,
                    "Savings %": f"{round(pct, 2)}%"
                })
                
        if comparison_rows:
            st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
            
            # Show aggregate savings
            total_baseline = sum(row["General Chatbot Baseline (Whole Codebase)"] for row in comparison_rows)
            total_graph = sum(row["Graph-Optimized Prompt Tokens"] for row in comparison_rows)
            total_saved = total_baseline - total_graph
            avg_pct = (total_saved / total_baseline * 100) if total_baseline else 0
            
            st.markdown("#### 📈 Cumulative Savings Against Whole Codebase Baseline")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Baseline Tokens", f"{total_baseline:,}")
            c2.metric("Total Optimized Tokens", f"{total_graph:,}")
            c3.metric("Total Tokens Saved", f"{total_saved:,}")
            c4.metric("Avg. Token Savings %", f"{round(avg_pct, 2)}%")
        else:
            st.info("Ask a question in **Graph QA** to see direct query-by-query token savings here!")
            
        # "All Query History & Token Usage" section removed as requested.
    else:
        st.info("No queries have been run in this session yet. Go to Graph QA to ask a question!")

    # Static ingestion & parsing pipeline measurements removed per user request.


def parse_and_render_code_fix(answer_text: str, repo_id: str) -> None:
    if not answer_text:
        st.markdown("_No answer generated._")
        return
        
    # Regex to extract <code_fix>...</code_fix>
    match = re.search(r"<code_fix>(.*?)</code_fix>", answer_text, re.DOTALL)
    if not match:
        st.markdown(answer_text)
        return
        
    # Exclude the code_fix tags from the conversational output
    conversation_text = answer_text.replace(match.group(0), "").strip()
    if conversation_text:
        st.markdown(conversation_text)
        
    # Parse inner XML tags
    inner = match.group(1)
    file_match = re.search(r"<filepath>(.*?)</filepath>", inner, re.DOTALL)
    orig_match = re.search(r"<original_code>(.*?)</original_code>", inner, re.DOTALL)
    repl_match = re.search(r"<replacement_code>(.*?)</replacement_code>", inner, re.DOTALL)
    
    if not file_match or not orig_match or not repl_match:
        st.warning("⚠️ A code fix was proposed but could not be parsed completely.")
        return
        
    filepath = file_match.group(1).strip()
    original_code = orig_match.group(1)
    if original_code.startswith("\n"):
        original_code = original_code[1:]
    if original_code.endswith("\n"):
        original_code = original_code[:-1]
        
    replacement_code = repl_match.group(1)
    if replacement_code.startswith("\n"):
        replacement_code = replacement_code[1:]
    if replacement_code.endswith("\n"):
        replacement_code = replacement_code[:-1]
        
    st.markdown("---")
    st.markdown(f"#### 🛠️ Proposed Fix for `{filepath}`")
    
    # Compute unified diff
    orig_lines = original_code.splitlines()
    repl_lines = replacement_code.splitlines()
    diff_generator = difflib.unified_diff(
        orig_lines, 
        repl_lines, 
        fromfile=f"a/{filepath}", 
        tofile=f"b/{filepath}", 
        lineterm=""
    )
    diff_text = "\n".join(list(diff_generator))
    
    st.code(diff_text, language="diff", line_numbers=True)
    
    # Render Apply Fix button
    button_key = f"apply_fix_btn_{hashlib.md5(filepath.encode()).hexdigest()}"
    if st.button("📁 Apply Fix directly to Workspace", key=button_key, type="primary"):
        with st.spinner("Applying codebase changes and re-ingesting..."):
            res = chat_service.apply_rectification(
                repo_id=repo_id,
                file_path=filepath,
                original_code=original_code,
                replacement_code=replacement_code
            )
            if res.get("status") == "success":
                st.success(res.get("message"))
                st.balloons()
                st.info("🔄 Repository has been successfully re-indexed in the background! Ask another question to see the updated graph.")
            else:
                st.error(f"❌ Error applying fix: {res.get('error')}")


def subgraph_to_dot(record: QueryRecord) -> str:
    nodes = record.selected_nodes
    edges = record.selected_edges
    
    lines = [
        "digraph G {", 
        "rankdir=LR;", 
        "bgcolor=transparent;",
        'node [shape=box, style="rounded,filled", fontname="Plus Jakarta Sans", fontsize=10];',
        'edge [fontname="Plus Jakarta Sans", fontsize=8, penwidth=1.2];'
    ]
    
    # Identify which nodes are anchors based on source snippets
    anchor_keys = set()
    for snippet in record.source_snippets:
        if snippet.source == "graph_anchor":
            anchor_keys.add((snippet.file_path, snippet.line_start))
            
    # Render nodes with distinct color highlights
    for node in nodes:
        is_anchor = (node.file_path, node.line_start) in anchor_keys
        node_id_escaped = node.node_id.replace('"', "'")
        label = f"[{node.node_type.upper()}]\\n{node.label}".replace('"', "'")
        
        if is_anchor:
            # Gold/amber premium outline for Primary Anchors (full bodies)
            lines.append(
                f'"{node_id_escaped}" [label="{label}", fillcolor="#fef3c7", color="#f59e0b", fontcolor="#1e293b", penwidth=2.5];'
            )
        else:
            # Sleek slate gray outline for Neighbor Nodes (signatures)
            lines.append(
                f'"{node_id_escaped}" [label="{label}", fillcolor="#111827", color="#4b5563", fontcolor="#cbd5e1", penwidth=1.5];'
            )
            
    # Render edges
    visible_node_ids = {node.node_id for node in nodes}
    for edge in edges:
        if edge.source_node in visible_node_ids and edge.target_node in visible_node_ids:
            label = edge.edge_type.replace('"', "'")
            lines.append(
                f'"{edge.source_node}" -> "{edge.target_node}" [label="{label}", color="#6366f1", fontcolor="#9ca3af"];'
            )
            
    lines.append("}")
    return "\n".join(lines)


def render_query_record(record: QueryRecord) -> None:
    if record.error:
        st.error(record.error)
    st.caption(f"{record.status} | {record.latency_ms} ms | query_id={record.query_id}")
    parse_and_render_code_fix(record.answer, record.repo_id)

    
    # Calculate Token and Context Size Analytics
    baseline_prompt = record.token_usage.get("whole_codebase_baseline")
    optimized_prompt = record.token_usage.get("llm_prompt_tokens")
    response_tokens = record.token_usage.get("llm_response_tokens")
    
    baseline_count = baseline_prompt.tokens if baseline_prompt else 0
    optimized_count = optimized_prompt.tokens if optimized_prompt else 0
    response_count = response_tokens.tokens if response_tokens else 0
    
    saved_tokens = max(0, baseline_count - optimized_count)
    saved_percent = round((saved_tokens / baseline_count * 100), 1) if baseline_count else 0.0
    
    st.markdown("---")
    st.subheader("📊 Token & Context Size Savings Analysis")
    
    if baseline_count > 0:
        cols = st.columns(4)
        cols[0].metric(
            "General Baseline", 
            f"{baseline_count:,} tokens",
            help="Total prompt tokens required if you uploaded the entire codebase folder directly to the LLM."
        )
        cols[1].metric(
            "Optimized Graph", 
            f"{optimized_count:,} tokens",
            help="Actual prompt tokens sent using your PageRank Hybrid Retrieval graph optimization."
        )
        cols[2].metric(
            "Net Input Saved", 
            f"{saved_tokens:,} tokens", 
            f"-{saved_percent}%" if saved_percent > 0 else "0.0%",
            help="Absolute and percentage reduction in LLM prompt (input) tokens."
        )
        cols[3].metric(
            "LLM Output Tokens", 
            f"{response_count:,} tokens",
            help="Actual response tokens generated by Gemini to answer your question."
        )
    else:
        st.info("Ingesting token summary... Ask a question to compute complete savings metrics!")
        
    with st.expander("🔍 Detailed Token Breakdown"):
        st.markdown(
            f"""
            * **General Chatbot Baseline (Whole Codebase):** `{baseline_count:,}` tokens
            * **Our Optimized Graph QA (Actual):** `{optimized_count:,}` tokens
            * **Net Input Tokens Saved:** `{saved_tokens:,}` tokens (**{saved_percent}% prompt size reduction**!)
            * **LLM Response Output size:** `{response_count:,}` tokens
            
            This means your PageRank Hybrid Retrieval algorithm saved **{saved_tokens:,} input tokens** on this single query!
            """
        )
        st.dataframe([value.model_dump() for value in record.token_usage.values()], use_container_width=True, hide_index=True)

    if record.selected_nodes:
        counts = graph_node_source_counts(record)
        cols = st.columns(3)
        cols[0].metric("CodeGraph nodes", counts["codegraph"])
        cols[1].metric("Native Graphify nodes", counts["graphify"])
        cols[2].metric("Fallback Graphify nodes", counts["graphify_fallback"])
        
        # Build raw text blocks passed to the prompt
        node_lines = [
            f"- {node.node_id} [{node.node_type}] {node.label} ({node.file_path or 'external'}:{node.line_start or '-'})"
            for node in record.selected_nodes
        ]
        edge_lines = [
            f"- {edge.source_node} --{edge.edge_type}--> {edge.target_node}"
            for edge in record.selected_edges
        ]
        snippet_lines = [
            f"### {snippet.file_path}:{snippet.line_start}-{snippet.line_end} ({snippet.source})\n{snippet.text}"
            for snippet in record.source_snippets
        ]
        exact_context = "\n".join(
            [
                "Graph-selected nodes:",
                *node_lines,
                "",
                "Graph relationships:",
                *edge_lines,
                "",
                "Source snippets:",
                *snippet_lines,
            ]
        )
        
        with st.expander("📄 View Exact Context Passed to LLM", expanded=False):
            t1, t2 = st.tabs(["📝 Context Text Block Only", "🚀 Complete Raw LLM Prompt Ingested"])
            with t1:
                st.markdown(
                    "This is the exact, optimized context text block that was appended to the LLM prompt "
                    "using your PageRank Hybrid Retrieval algorithm:"
                )
                st.text_area("LLM Prompt Context Text", exact_context, height=350)
            with t2:
                st.markdown(
                    "This is the **entire raw prompt** received by the Gemini model, including "
                    "system role guidelines, automated code rectifier instructions, your question, and the context block:"
                )
                rectify_str = ""
                if st.session_state.get("rectify_enabled", False):
                    rectify_str = (
                        "\n\nIMPORTANT: If you identify any bug/error and propose a code change, you MUST wrap the proposed fix exactly in "
                        "the following XML structure so the system can apply it automatically:\n"
                        "<code_fix>\n"
                        "  <filepath>relative/path/to/file.py</filepath>\n"
                        "  <original_code>\n"
                        "// Exact block of old code to replace (must match precisely including spacing)\n"
                        "  </original_code>\n"
                        "  <replacement_code>\n"
                        "// Exact block of new code to insert\n"
                        "  </replacement_code>\n"
                        "</code_fix>\n"
                        "Make sure that the <original_code> block you target matches the codebase content exactly, character-for-character."
                    )
                full_raw_prompt = (
                    "You are a graph-aware repository QA assistant. Use the selected CodeGraph and Graphify nodes, "
                    "relationships, and snippets to answer. Prefer precise relationships over broad guesses. "
                    "Cite files, lines, and relevant graph nodes. If the graph context is insufficient, say so. "
                    "Be extremely concise, direct, and brief in your answer. Avoid verbose explanations."
                    f"{rectify_str}\n\n"
                    f"Question:\n{record.query}\n\n"
                    f"Optimized graph context:\n{exact_context}"
                )
                st.text_area("Gemini Final Prompt", full_raw_prompt, height=450)

        # Graphical Subgraph Neighborhood representation
        st.markdown("#### 🗺️ Context Neighborhood Map")
        st.caption("Selected structural neighborhood map for this query. Primary Anchors (Full context) are highlighted in **yellow/orange**, and Neighbor Nodes (Signature context) are in **gray**.")
        dot = subgraph_to_dot(record)
        st.graphviz_chart(dot, use_container_width=True)

        with st.expander("Selected Graph Nodes"):
            st.dataframe([node.model_dump() for node in record.selected_nodes], use_container_width=True, hide_index=True)

        with st.expander("Selected Graph Edges"):
            if record.selected_edges:
                st.dataframe([edge.model_dump() for edge in record.selected_edges], use_container_width=True, hide_index=True)
            else:
                st.info("No graph edges were selected for this query.")
            
    with st.expander("Source Snippets", expanded=True):
        for snippet in record.source_snippets:
            st.caption(f"{snippet.file_path}:{snippet.line_start}-{snippet.line_end} | {snippet.source}")
            st.code(snippet.text, language="python", line_numbers=True)




def graph_node_source_counts(record: QueryRecord) -> dict[str, int]:
    counts = {"codegraph": 0, "graphify": 0, "graphify_fallback": 0}
    for node in record.selected_nodes:
        is_graphify = (
            node.node_id.startswith("graphify:") or 
            (node.metadata and (
                node.metadata.get("merged_from_graphify") is True or
                node.metadata.get("graphify_processed") is True
            ))
        )
        is_fallback = (
            node.node_id.startswith("graphify-fallback:") or 
            (node.metadata and node.metadata.get("fallback") is True)
        )
        
        if is_fallback:
            counts["graphify_fallback"] += 1
        elif is_graphify:
            counts["graphify"] += 1
            
        if node.node_id.startswith("codegraph:") or is_graphify:
            counts["codegraph"] += 1
    return counts


def qa_prompt_help() -> str:
    return "Ask about architecture, important functions, call paths, classes, imports, or implementation behavior."


def render_graph_qa(repo: RepoMetadata | None) -> None:
    st.header("Graph-Optimized Repo QA")
    if not repo:
        st.info("Load a repository first.")
        return
    codegraph = storage.load_codegraph(repo.repo_id)
    graphify = storage.load_graphify(repo.repo_id)
    codegraph_count = len(codegraph.nodes) if codegraph else 0
    graphify_count = len(graphify.nodes) if graphify else 0

    st.caption(
        f"Graph QA uses a single selected graph source. Choose CodeGraph or Graphify. "
        f"If the chosen source is unavailable, the app will report it instead of falling back."
    )
    st.info(
        f"Available sources: CodeGraph = {codegraph_count} nodes, Graphify = {graphify_count} nodes."
    )

    source_label = st.selectbox(
        "Retrieve Graph Context from:",
        ["CodeGraph", "Graphify"],
        index=0,
        key="graph_source_select"
    )
    source_selection = "codegraph" if "CodeGraph" in source_label else "graphify"

    budget_label = st.selectbox(
        "Context Size Budget:",
        ["Balanced (Recommended)", "Tight (Token Saver)", "Deep (Large Codebase Coverage)"],
        index=0,
        help="Adjust context bounds: Balanced is ideal for normal use, Tight minimizes token cost, and Deep is suited for large repositories with many files."
    )
    
    if budget_label == "Tight (Token Saver)":
        st.session_state.graph_max_nodes = 8
    elif budget_label == "Deep (Large Codebase Coverage)":
        st.session_state.graph_max_nodes = 24
    else: # Balanced
        st.session_state.graph_max_nodes = 14

    source_available = (source_selection == "codegraph" and codegraph_count > 0) or (
        source_selection == "graphify" and graphify_count > 0
    )

    if not source_available:
        st.error(f"{source_label} is not available for this repository.")

    query = st.text_area("Question", placeholder=qa_prompt_help(), key="graph_query")
    if st.button("Ask Graph QA", disabled=not query.strip() or not source_available, type="primary"):
        if not source_available:
            st.error(f"Cannot run Graph QA because {source_label} is unavailable.")
        else:
            with st.spinner("Selecting graph neighborhood and calling Gemini..."):
                record = chat_service.graph_optimized_qa(
                    repo.repo_id,
                    query.strip(),
                    st.session_state.session_id,
                    source_selection=source_selection,
                    max_nodes=st.session_state.get("graph_max_nodes", 8),
                )
                st.session_state.graph_record = record

    if "graph_record" in st.session_state:
        render_query_record(st.session_state.graph_record)





def main() -> None:
    st.markdown(
        '''
        <h1 class="gradient-text" style="font-size:42px; margin: 0; padding-bottom: 5px; line-height: 1.2;">
            Graph-Augmented Code Understanding
        </h1>
        <p style="font-size:15px; margin-top:8px; margin-bottom:24px; color: #9ca3af; font-weight: 500;">
            Evaluate token consumption, query performance, and repository comprehension with and without graphs.
        </p>
        ''',
        unsafe_allow_html=True,
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    repo = current_repo()

    tab_names = [
        "Upload / Import",
        "CodeGraph",
        "Graphify",
        "Graph QA",
        "Token Analytics",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_upload_import(repo)
    with tabs[1]:
        render_graph(repo, "codegraph")
    with tabs[2]:
        render_graph(repo, "graphify")
    with tabs[3]:
        render_graph_qa(repo)
    with tabs[4]:
        render_tokens(repo)




if __name__ == "__main__":
    main()
