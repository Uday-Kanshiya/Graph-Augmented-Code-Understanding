from __future__ import annotations

import os
import shutil
import sys
import re
import subprocess
from pathlib import Path
from uuid import uuid4

import streamlit as st
import networkx as nx
import time

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
    # Force reload cache to apply the updated GraphifyService path walking and diagnostics
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
    if not hasattr(ChatService, "get_rectify_instructions"):
        ChatService.get_rectify_instructions = lambda self: (
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
    # Clear last rectified files on repo change
    st.session_state.codegraph_last_rectified_file = None
    st.session_state.graphify_last_rectified_file = None


def ingest_uploaded_file(uploaded_file) -> RepoMetadata:
    repo_id = uuid4().hex
    repo_name = clean_repo_name(Path(uploaded_file.name).stem)
    source_dir = storage.repo_source_dir(repo_id)
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    
    if uploaded_file.name.lower().endswith(".zip"):
        upload_path = storage.uploads_dir / f"{repo_id}.zip"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(uploaded_file.getbuffer())
        safe_extract_zip(upload_path, source_dir)
    else:
        # Single code/programming file upload
        file_path = source_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())
        
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
        st.subheader("Upload zipped codebase or single file")
        uploaded_file = st.file_uploader("Choose .zip or coding file")
        if st.button("Analyze upload", disabled=uploaded_file is None, type="primary"):
            with st.spinner("Analyzing uploaded files..."):
                try:
                    repo = ingest_uploaded_file(uploaded_file)
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
        "Supported languages: Python, JavaScript/TypeScript, Go, Rust, Java, C/C++, HTML, CSS, Shell, and other text-based languages. "
        "The app automatically skips .git, virtual environments (.venv), node_modules, build outputs, caches, and binary files."
    )

    if repo:
        st.divider()
        st.subheader("Repository Summary")
        
        files = storage.load_files(repo.repo_id)
        from collections import Counter
        lang_counter = Counter()
        lang_lines = {}
        if files:
            for file in files:
                lang = file.language or "unknown"
                lang_counter[lang] += 1
                lang_lines[lang] = lang_lines.get(lang, 0) + file.line_count
                
        if lang_counter:
            top_lang, top_count = lang_counter.most_common(1)[0]
            top_lines = lang_lines.get(top_lang, 0)
        else:
            top_lang = "None"
            top_count = 0
            top_lines = 0

        cols = st.columns(4)
        cols[0].metric("Total files", repo.stats.total_files)
        cols[1].metric("Top Language (Files)", f"{top_lang.capitalize()} ({top_count})")
        cols[2].metric("Total lines", repo.stats.total_lines)
        cols[3].metric(f"{top_lang.capitalize()} lines", f"{top_lines:,}")
        st.write(f"**Origin:** `{repo.origin}`")
        if repo.error:
            st.error(repo.error)
        
        # Display files by language in table format
        files = storage.load_files(repo.repo_id)
        if files:
            st.subheader("Files by Language")
            
            # Create a list of file data with path, language and lines
            file_data = [
                {
                    "File": file.path, 
                    "Language": file.language or "unknown",
                    "Lines": file.line_count
                }
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


def generate_interactive_graph_html(graph: GraphDocument) -> str:
    import json
    
    # Colors for different node types
    color_map = {
        "file": {"background": "#1e1b4b", "border": "#6366f1"},       # Indigo
        "module": {"background": "#1e1b4b", "border": "#6366f1"},     # Indigo
        "class": {"background": "#3b0764", "border": "#a855f7"},      # Purple
        "component": {"background": "#3b0764", "border": "#a855f7"},  # Purple
        "function": {"background": "#022c22", "border": "#10b981"},   # Emerald
        "method": {"background": "#022c22", "border": "#10b981"},     # Emerald
        "concept": {"background": "#451a03", "border": "#f59e0b"},     # Amber
    }
    default_color = {"background": "#0c4a6e", "border": "#0ea5e9"}    # Sky
    
    nodes_json_list = []
    for node in graph.nodes:
        # Get color based on node type
        color = color_map.get(node.node_type.lower(), default_color)
        
        # Build node representation
        node_dict = {
            "id": node.node_id,
            "label": node.label,
            "title": f"[{node.node_type.upper()}] {node.label}",
            "node_type": node.node_type,
            "file_path": node.file_path,
            "line_start": node.line_start,
            "line_end": node.line_end,
            "source_snippet": node.source_snippet,
            "metadata": node.metadata or {},
            "color": {
                "background": color["background"],
                "border": color["border"],
                "highlight": {
                    "background": color["border"],
                    "border": "#ffffff"
                },
                "hover": {
                    "background": color["border"],
                    "border": color["border"]
                }
            }
        }
        nodes_json_list.append(node_dict)
        
    edges_json_list = []
    for edge in graph.edges:
        edges_json_list.append({
            "from": edge.source_node,
            "to": edge.target_node,
            "label": edge.edge_type,
            "title": f"{edge.edge_type} (score: {edge.score})",
            "weight": edge.score,
        })

    nodes_json_str = json.dumps(nodes_json_list)
    edges_json_str = json.dumps(edges_json_list)

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Interactive Graph View</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0b0f19;
    color: #f3f4f6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }
  #graph {
    flex: 1;
    height: 100%;
  }
  #sidebar {
    width: 320px;
    background: #111827;
    border-left: 1px solid #1f2937;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    padding: 20px;
    box-shadow: -4px 0 20px rgba(0, 0, 0, 0.4);
  }
  #search-wrap {
    margin-bottom: 20px;
  }
  #search {
    width: 100%;
    background: #1f2937;
    border: 1px solid #374151;
    color: #f3f4f6;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    outline: none;
    transition: all 0.2s ease;
  }
  #search:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
  }
  #search-results {
    max-height: 150px;
    overflow-y: auto;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    margin-top: 5px;
    display: none;
  }
  .search-item {
    padding: 8px 12px;
    cursor: pointer;
    font-size: 13px;
    border-bottom: 1px solid #1f2937;
  }
  .search-item:hover {
    background: #374151;
  }
  #info-panel {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  .card {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  .card-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9ca3af;
    margin-bottom: 8px;
    font-weight: 600;
  }
  .card-value {
    font-size: 15px;
    word-break: break-all;
    line-height: 1.5;
  }
  .badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    margin-top: 4px;
  }
  pre {
    background: #0b0f19;
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
    font-family: Consolas, Monaco, monospace;
    font-size: 12px;
    border: 1px solid #1f2937;
    margin-top: 8px;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .empty {
    color: #6b7280;
    font-style: italic;
    text-align: center;
    margin-top: 40px;
  }
</style>
</head>
<body>
<div id="graph"></div>
<div id="sidebar">
  <div id="search-wrap">
    <input id="search" type="text" placeholder="Search symbols..." autocomplete="off">
    <div id="search-results"></div>
  </div>
  <div id="info-panel">
    <div class="empty">Click on a node to view detailed architectural signatures.</div>
  </div>
</div>

<script>
  const nodesData = __NODES_PLACEHOLDER__;
  const edgesData = __EDGES_PLACEHOLDER__;

  const nodes = new vis.DataSet(nodesData);
  const edges = new vis.DataSet(edgesData);

  const container = document.getElementById('graph');
  const data = { nodes, edges };
  
  const options = {
    nodes: {
      shape: 'dot',
      size: 16,
      font: {
        color: '#f3f4f6',
        size: 12,
        face: 'system-ui'
      },
      borderWidth: 2,
      shadow: true
    },
    edges: {
      width: 1.5,
      color: { color: '#4b5563', highlight: '#6366f1', hover: '#818cf8' },
      arrows: { to: { enabled: true, scaleFactor: 0.8 } },
      smooth: { type: 'continuous', roundness: 0.5 },
      shadow: true
    },
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -50,
        centralGravity: 0.01,
        springLength: 100,
        springConstant: 0.08
      },
      stabilization: { iterations: 100 }
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
      navigationButtons: true,
      keyboard: true
    }
  };

  const network = new vis.Network(container, data, options);

  const searchInput = document.getElementById('search');
  const searchResults = document.getElementById('search-results');

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    if (!query) {
      searchResults.style.display = 'none';
      return;
    }
    const matches = nodesData.filter(n => 
      n.label.toLowerCase().includes(query) || 
      n.id.toLowerCase().includes(query)
    ).slice(0, 10);

    if (matches.length > 0) {
      searchResults.innerHTML = matches.map(n => 
        `<div class="search-item" data-id="${n.id}">${n.label} <span style="font-size: 10px; color: #6b7280;">(${n.node_type})</span></div>`
      ).join('');
      searchResults.style.display = 'block';
    } else {
      searchResults.innerHTML = '<div class="search-item" style="color: #6b7280; cursor: default;">No symbols found</div>';
      searchResults.style.display = 'block';
    }
  });

  searchResults.addEventListener('click', (e) => {
    const item = e.target.closest('.search-item');
    if (!item || !item.dataset.id) return;
    const nodeId = item.dataset.id;
    
    network.selectNodes([nodeId]);
    network.focus(nodeId, { scale: 1.2, animation: true });
    showNodeDetails(nodeId);
    
    searchInput.value = '';
    searchResults.style.display = 'none';
  });

  network.on("selectNode", function (params) {
    if (params.nodes.length > 0) {
      showNodeDetails(params.nodes[0]);
    }
  });

  network.on("deselectNode", function () {
    document.getElementById('info-panel').innerHTML = 
      '<div class="empty">Click on a node to view detailed architectural signatures.</div>';
  });

  function showNodeDetails(nodeId) {
    const node = nodesData.find(n => n.id === nodeId);
    if (!node) return;

    let metaHtml = '';
    if (node.metadata) {
      for (const [key, value] of Object.entries(node.metadata)) {
        if (typeof value === 'object') {
          metaHtml += `
            <div class="card">
              <div class="card-title">${key}</div>
              <pre>${JSON.stringify(value, null, 2)}</pre>
            </div>`;
        } else {
          metaHtml += `
            <div class="card">
              <div class="card-title">${key}</div>
              <div class="card-value">${value}</div>
            </div>`;
        }
      }
    }

    let codeHtml = '';
    if (node.source_snippet) {
      codeHtml = `
        <div class="card">
          <div class="card-title">Source Snippet</div>
          <pre>${escapeHtml(node.source_snippet)}</pre>
        </div>`;
    }

    document.getElementById('info-panel').innerHTML = `
      <div class="card">
        <div class="card-title">Symbol Name</div>
        <h2 style="font-size: 18px; font-weight: 700; color: #fff;">${node.label}</h2>
        <span class="badge" style="background: ${node.color.border}; color: #fff;">${node.node_type}</span>
      </div>
      <div class="card">
        <div class="card-title">File path</div>
        <div class="card-value" style="font-family: monospace; font-size: 12px; color: #a5b4fc;">${node.file_path || 'External/Built-in'}</div>
        ${node.line_start ? `<div style="font-size: 12px; color: #9ca3af; margin-top: 4px;">Line range: ${node.line_start} - ${node.line_end || node.line_start}</div>` : ''}
      </div>
      ${codeHtml}
      ${metaHtml}
    `;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
</script>
</body>
</html>
"""
    return html_template.replace("__NODES_PLACEHOLDER__", nodes_json_str).replace("__EDGES_PLACEHOLDER__", edges_json_str)


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
    cols = st.columns(3)
    cols[0].metric("Source", graph.source)
    cols[1].metric("Nodes", len(graph.nodes))
    cols[2].metric("Edges", len(graph.edges))

    html_path = None
    if kind == "graphify":
        repo_root = storage.repo_source_dir(repo.repo_id)
        if repo_root and repo_root.exists():
            possible_path = repo_root / "graphify-out" / "graph.html"
            if possible_path.exists():
                html_path = possible_path
        
        # Add selection toggle for Graphify visualization
        view_mode = st.radio(
            "Graph Visualization Mode:",
            ["🪐 Interactive HTML Graph", "📊 Static Schema (Graphviz)"],
            horizontal=True,
            key=f"{repo.repo_id}_graphify_view_mode"
        )
    else:
        view_mode = "📊 Static Schema (Graphviz)"

    rendered_interactive = False
    if view_mode == "🪐 Interactive HTML Graph":
        try:
            html_content = ""
            if html_path:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
            else:
                # Dynamically generate interactive HTML from the loaded graph document
                html_content = generate_interactive_graph_html(graph)
            
            if html_content:
                import base64
                import streamlit.components.v1 as components
                
                # Embed the vis-network graph directly in Streamlit page
                components.html(html_content, height=750, scrolling=False)
                rendered_interactive = True
                
                # Show standard action link/button to open in a new tab
                b64_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
                href = f"data:text/html;base64,{b64_html}"
                st.markdown(
                    """
                    <style>
                    .open-graph-btn {
                        display: inline-block;
                        padding: 0.5rem 1.5rem;
                        font-family: 'Plus Jakarta Sans', sans-serif;
                        font-size: 14px;
                        font-weight: 600;
                        color: #ffffff !important;
                        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
                        border: none;
                        border-radius: 10px;
                        text-decoration: none;
                        cursor: pointer;
                        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                        transition: all 0.2s ease;
                        margin-top: 10px;
                        margin-bottom: 20px;
                    }
                    .open-graph-btn:hover {
                        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
                        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
                        transform: translateY(-2px);
                        text-decoration: none;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<a href="{href}" target="_blank" class="open-graph-btn">🪐 Open Interactive Graphify (New Window)</a>',
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error rendering interactive graph HTML: {e}")
            rendered_interactive = False

    if not rendered_interactive:
        st.graphviz_chart(graph_to_dot(graph), use_container_width=True)
    with st.expander("Nodes"):
        st.dataframe([node.model_dump() for node in graph.nodes], use_container_width=True, hide_index=True)
    with st.expander("Edges"):
        st.dataframe([edge.model_dump() for edge in graph.edges], use_container_width=True, hide_index=True)

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
            
            # Determine query engine/source
            engine = "Graphify"
            if "codegraph_graphify_optimized_context" in q_rec.token_usage:
                engine = "CodeGraph"
            elif any("codegraph" in (sn.file_path or "") for sn in q_rec.source_snippets):
                engine = "CodeGraph"
            elif any("codegraph" in getattr(node, "node_id", "") for node in q_rec.selected_nodes):
                engine = "CodeGraph"
                
            if baseline_tokens > 0:
                saved = baseline_tokens - prompt_tokens
                pct = (saved / baseline_tokens * 100) if baseline_tokens else 0
                comparison_rows.append({
                    "Query / Question": q_text,
                    "Engine": engine,
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
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Usage", f"{total_graph:,}")
            c2.metric("Total Req for Raw Input", f"{total_baseline:,}")
            c3.metric("% Saved", f"{round(avg_pct, 2)}%")
        else:
            st.info("Ask a question in **Graphify QA** to see direct query-by-query token savings here!")
            
        # "All Query History & Token Usage" section removed as requested.
    else:
        st.info("No queries have been run in this session yet. Go to Graphify QA to ask a question!")

    # Static ingestion & parsing pipeline measurements removed per user request.





def parse_code_fix(text: str) -> dict | None:
    match = re.search(r"<code_fix>(.*?)</code_fix>", text, re.DOTALL)
    if not match:
        return None
    content = match.group(1)
    
    fp_match = re.search(r"<filepath>(.*?)</filepath>", content, re.DOTALL)
    orig_match = re.search(r"<original_code>(.*?)</original_code>", content, re.DOTALL)
    repl_match = re.search(r"<replacement_code>(.*?)</replacement_code>", content, re.DOTALL)
    
    if fp_match and orig_match and repl_match:
        return {
            "filepath": fp_match.group(1).strip(),
            "original_code": orig_match.group(1),
            "replacement_code": repl_match.group(1),
        }
    return None


def render_codegraph_qa(repo: RepoMetadata | None) -> None:
    st.header("CodeGraph QA")
    if not repo:
        st.info("Load a repository first.")
        return
        
    graph = storage.load_codegraph(repo.repo_id)
    if not graph or not graph.nodes:
        st.error("No CodeGraph graph found. Please build or import a repository with CodeGraph output first.")
        return
        
    left, right = st.columns([3, 1])
    with left:
        question = st.text_area(
            "Ask a question about codebase symbols and relationships:",
            placeholder="e.g. 'How does TokenService calculate prompt savings?'",
            key="codegraph_qa_question"
        )
    with right:
        max_nodes_input = st.text_input(
            "Max Nodes:",
            value="8",
            help="Limits context details sent to the LLM. Larger number provides more files but uses more tokens."
        )
        rectify_checked = st.checkbox(
            "Enable Error Rectification",
            value=False,
            key="codegraph_qa_rectify",
            help="If enabled, the LLM will check for errors and suggest a code fix."
        )
        
    max_nodes_val = 8
    if max_nodes_input.strip():
        try:
            max_nodes_val = int(max_nodes_input.strip())
        except ValueError:
            st.warning("Please enter a valid integer for Max Nodes.")
            
    if st.button("Ask CodeGraph QA", disabled=not question.strip(), type="primary"):
        st.session_state.codegraph_last_rectified_file = None  # Clear old rectified file on new query
        with st.spinner("Querying CodeGraph structure and calling LLM..."):
            started = time.perf_counter()
            try:
                # 1. Build context via GraphRetrievalService
                graph_context = chat_service.graph_retrieval_service.build_context(
                    repo.repo_id,
                    question.strip(),
                    max_nodes=max_nodes_val,
                    source_selection="codegraph"
                )
                
                # 2. Get LLM Prompt
                prompt = chat_service._graph_prompt(question.strip(), graph_context.context, rectify=rectify_checked)
                
                # 3. Call LLM
                response = llm_provider.generate_answer(prompt)
                
                # 4. Save query to storage so it shows in Token Analytics history
                from app.models.schemas import TokenMeasurement, CountType, QueryRecord
                token_usage = {
                    "codegraph_graphify_optimized_context": graph_context.token_measurement,
                    "llm_prompt_tokens": response.prompt_tokens,
                    "llm_response_tokens": response.response_tokens,
                    "total_per_query_tokens": response.total_tokens,
                }
                
                # Compute Whole Codebase Baseline
                repo_tokens = chat_service._get_total_repo_tokens(repo.repo_id)
                query_prompt = chat_service._standard_prompt(question.strip(), "[concatenated_files_placeholder]")
                query_tokens = chat_service._prompt_measurement(query_prompt).tokens
                raw_input_token_usage = repo_tokens + query_tokens
                
                token_usage["whole_codebase_baseline"] = TokenMeasurement(
                    stage="whole_codebase_baseline",
                    tokens=raw_input_token_usage,
                    count_type=CountType.exact if chat_service.token_service._encoding else CountType.estimated,
                    notes="Prompt tokens required if sending 100% of codebase files directly to LLM."
                )
                
                # Convert objects to dicts to avoid cross-module Pydantic model conflicts
                source_snippets_dicts = [sn.model_dump() if hasattr(sn, "model_dump") else sn for sn in graph_context.snippets]
                selected_nodes_dicts = [n.model_dump() if hasattr(n, "model_dump") else n for n in graph_context.selected_nodes]
                selected_edges_dicts = [e.model_dump() if hasattr(e, "model_dump") else e for e in graph_context.selected_edges]
                
                token_usage_dicts = {}
                for k, v in token_usage.items():
                    if hasattr(v, "model_dump"):
                        token_usage_dicts[k] = v.model_dump()
                    else:
                        token_usage_dicts[k] = v
                
                record = QueryRecord(
                    query_id=uuid4().hex,
                    repo_id=repo.repo_id,
                    session_id=st.session_state.session_id,
                    mode="graph_optimized",
                    query=question.strip(),
                    status="completed",
                    answer=response.text,
                    error=None,
                    source_snippets=source_snippets_dicts,
                    selected_nodes=selected_nodes_dicts,
                    selected_edges=selected_edges_dicts,
                    token_usage=token_usage_dicts,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                storage.save_query(record)
                
                st.session_state.codegraph_qa_answer = response.text
                st.session_state.codegraph_qa_prompt = prompt
                st.session_state.codegraph_qa_context = graph_context.context
                st.session_state.codegraph_qa_snippets = graph_context.snippets
                st.session_state.codegraph_qa_nodes = graph_context.selected_nodes
                st.session_state.codegraph_qa_edges = graph_context.selected_edges
                st.session_state.codegraph_qa_tokens = {
                    "prompt": response.prompt_tokens.tokens,
                    "response": response.response_tokens.tokens,
                    "total": response.total_tokens.tokens,
                    "raw_input": raw_input_token_usage,
                    "notes": response.total_tokens.notes
                }
                st.session_state.codegraph_qa_error = None
            except Exception as e:
                st.session_state.codegraph_qa_error = f"Error during CodeGraph QA generation: {e}"
                st.session_state.codegraph_qa_answer = None
                st.session_state.codegraph_qa_prompt = None
                st.session_state.codegraph_qa_context = None
                st.session_state.codegraph_qa_snippets = []
                st.session_state.codegraph_qa_nodes = []
                st.session_state.codegraph_qa_edges = []
                st.session_state.codegraph_qa_tokens = None
                
    if st.session_state.get("codegraph_qa_error"):
        st.error(st.session_state.codegraph_qa_error)
        
    if st.session_state.get("codegraph_qa_answer"):
        st.success("Answer synthesized successfully!")
        
        tokens_info = st.session_state.codegraph_qa_tokens
        if tokens_info:
            cols = st.columns(3)
            cols[0].metric("Current Token Usage", f"{tokens_info['prompt']:,}")
            cols[1].metric("Raw Input Token Usage", f"{tokens_info['raw_input']:,}")
            if tokens_info['raw_input'] > 0:
                pct_reduction = 100 - (tokens_info['prompt'] / tokens_info['raw_input']) * 100
                cols[2].metric("Token Reduction %", f"{round(pct_reduction, 2)}%")
            else:
                cols[2].metric("Token Reduction %", "0.00%")
                
        st.subheader("💡 Answer")
        st.markdown(st.session_state.codegraph_qa_answer)

        # Check for error rectification code fix
        fix = parse_code_fix(st.session_state.codegraph_qa_answer)
        if fix:
            st.divider()
            st.subheader("🛠️ Proposed Error Rectification")
            st.markdown(f"**Target File:** `{fix['filepath']}`")
            
            # Show original and replacement
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Code:**")
                st.code(fix["original_code"])
            with col2:
                st.markdown("**Proposed Replacement:**")
                st.code(fix["replacement_code"])
                
            # Render apply button
            if st.button("Apply Suggested Fix", key="apply_codegraph_fix_btn", type="primary"):
                with st.spinner("Applying fix and re-analyzing codebase..."):
                    res = chat_service.apply_rectification(
                        repo.repo_id,
                        fix["filepath"],
                        fix["original_code"],
                        fix["replacement_code"]
                    )
                    if res["status"] == "success":
                        st.success(res["message"])
                        
                        # Save the updated file info for download option
                        filepath_abs = storage.repo_source_dir(repo.repo_id) / fix["filepath"]
                        if filepath_abs.exists():
                            try:
                                updated_content = filepath_abs.read_text(encoding="utf-8")
                                st.session_state.codegraph_last_rectified_file = {
                                    "filename": Path(fix["filepath"]).name,
                                    "filepath": fix["filepath"],
                                    "content": updated_content
                                }
                            except Exception as e:
                                st.error(f"Error reading rectified file for download: {e}")
                        st.rerun()
                    else:
                        st.error(res["error"])
        
        with st.expander("🔬 Retrieved Subgraph Details and Raw LLM Prompt", expanded=False):
            is_cli = any(sn.file_path == "codegraph:cli" for sn in st.session_state.codegraph_qa_snippets)
            
            if is_cli:
                t1, t2 = st.tabs(["📝 CodeGraph Explore Output", "📄 Raw Input"])
                with t1:
                    st.markdown("**Context retrieved via `@colbymchenry/codegraph`:**")
                    st.markdown(st.session_state.codegraph_qa_context)
                with t2:
                    st.markdown("**Exact Prompt Sent to LLM:**")
                    st.text_area("LLM Final Prompt", st.session_state.codegraph_qa_prompt, height=400)
            else:
                t1, t2, t3 = st.tabs(["📝 Selected AST Nodes", "🔗 Selected AST Edges", "📄 Raw Input"])
                with t1:
                    nodes_list = st.session_state.codegraph_qa_nodes
                    st.markdown(f"**Retrieved {len(nodes_list)} AST Nodes:**")
                    for node in nodes_list:
                        node_label = node.label or node.node_id
                        node_file = node.file_path or "external"
                        node_loc = f"{node.line_start}-{node.line_end}" if node.line_start else "N/A"
                        st.markdown(f"  **NODE `{node_label}`** [src={node_file} loc={node_loc}] (Type: `{node.node_type}`)")
                        if node.source_snippet:
                            st.code(node.source_snippet[:2000])
                        st.divider()
                with t2:
                    edges_list = st.session_state.codegraph_qa_edges
                    st.markdown(f"**Retrieved {len(edges_list)} AST Edges:**")
                    for edge in edges_list:
                        st.markdown(f"  **EDGE** `{edge.source_node}` --`{edge.edge_type}`--> `{edge.target_node}`")
                with t3:
                    st.markdown("**Exact Prompt Sent to LLM:**")
                    st.text_area("LLM Final Prompt", st.session_state.codegraph_qa_prompt, height=400)

    # Render persistent download button at the bottom of the tab if a file was changed
    rectified_file = st.session_state.get("codegraph_last_rectified_file")
    if rectified_file:
        st.divider()
        st.subheader("📥 Download Rectified File")
        st.info(f"The file `{rectified_file['filepath']}` was successfully updated via error rectification.")
        st.download_button(
            label=f"Download {rectified_file['filename']}",
            data=rectified_file["content"],
            file_name=rectified_file["filename"],
            mime="text/plain",
            key="download_codegraph_rectified_file"
        )



def render_graphify_qa(repo: RepoMetadata | None) -> None:
    st.header("Graphify QA")
    if not repo:
        st.info("Load a repository first.")
        return
        
    graph = storage.load_graphify(repo.repo_id)
    if not graph or not graph.nodes:
        st.error("No Graphify graph found. Please build or import a repository with Graphify output first.")
        return
        

    
    left, right = st.columns([3, 1])
    with left:
        question = st.text_area("Ask a question about codebase architecture:", placeholder="e.g. 'How is TokenService connected to standard QA?'", key="graphify_qa_question")
    with right:
        mode_selection = st.radio(
            "Traversal Strategy:",
            ["BFS (Broad Context)", "DFS (Deep Chain)"],
            index=0,
            help="BFS finds broad nearest neighbors. DFS traces deep connection chains."
        )
        token_budget_input = st.text_input(
            "Max Token Usage (Budget):",
            placeholder="e.g. 2000",
            help="Limits context details sent to the LLM. If empty, no limit is applied."
        )
        rectify_checked = st.checkbox(
            "Enable Error Rectification",
            value=False,
            key="graphify_qa_rectify",
            help="If enabled, the LLM will check for errors and suggest a code fix."
        )
        
    token_budget_val = None
    if token_budget_input.strip():
        try:
            token_budget_val = int(token_budget_input.strip())
        except ValueError:
            st.warning("Please enter a valid integer for the token budget.")
            
    if st.button("Ask Graphify QA", disabled=not question.strip(), type="primary"):
        st.session_state.graphify_last_rectified_file = None  # Clear old rectified file on new query
        with st.spinner("Executing Graphify query and calling LLM..."):
            started = time.perf_counter()
            q_text = question.strip()
            response = None
            raw_input_token_usage = 0
            
            # 1. Try calling the graphify CLI first
            cli_available = False
            cli_output = ""
            try:
                repo_root = storage.repo_source_dir(repo.repo_id)
                if repo_root and repo_root.exists():
                    proc = subprocess.run(
                        ["graphify", "query", q_text],
                        cwd=str(repo_root),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if proc.returncode == 0:
                        cli_output = proc.stdout.strip()
                        if cli_output:
                            cli_available = True
            except Exception:
                pass

            try:
                if cli_available:
                    output = cli_output
                    budget_exceeded = False
                    if token_budget_val is not None:
                        char_budget = token_budget_val * 4
                        if len(output) > char_budget:
                            output = output[:char_budget] + f'\n... (truncated at ~{token_budget_val} token budget - use --budget N for more)'
                            budget_exceeded = True
                            
                    rectify_str = chat_service.get_rectify_instructions() if rectify_checked else ""
                    prompt = (
                        "You are a graph-aware repository QA assistant. Use the following Graphify output to answer the question.\n"
                        "Answer using ONLY what the graph context contains.\n\n"
                        f"Question: {q_text}\n\n"
                        f"Graph Context:\n{output}"
                        f"{rectify_str}"
                    )
                    
                    response = llm_provider.generate_answer(prompt)
                    
                    st.session_state.graphify_qa_answer = response.text
                    st.session_state.graphify_qa_nodes = []
                    st.session_state.graphify_qa_edges = []
                    st.session_state.graphify_qa_start_nodes = []
                    st.session_state.graphify_qa_prompt = prompt
                    st.session_state.graphify_qa_cli_output = cli_output
                    st.session_state.graphify_qa_using_cli = True
                    
                    repo_tokens = chat_service._get_total_repo_tokens(repo.repo_id)
                    query_prompt = chat_service._standard_prompt(q_text, "[concatenated_files_placeholder]")
                    query_tokens = chat_service._prompt_measurement(query_prompt).tokens
                    raw_input_token_usage = repo_tokens + query_tokens
                    
                    st.session_state.graphify_qa_tokens = {
                        "prompt": response.prompt_tokens.tokens,
                        "response": response.response_tokens.tokens,
                        "total": response.total_tokens.tokens,
                        "raw_input": raw_input_token_usage,
                        "notes": response.total_tokens.notes
                    }
                    st.session_state.graphify_qa_error = None
                    st.session_state.graphify_qa_budget_exceeded = budget_exceeded
                    st.session_state.graphify_qa_budget_limit = token_budget_val
                    st.session_state.graphify_qa_total_discovered = 0
                    st.session_state.graphify_qa_total_included = 0
                else:
                    # Fallback: custom python AST traversal using NetworkX
                    G = nx.Graph()
                    for node in graph.nodes:
                        G.add_node(
                            node.node_id,
                            label=node.label,
                            node_type=node.node_type,
                            file_path=node.file_path,
                            line_start=node.line_start,
                            line_end=node.line_end,
                            source_snippet=node.source_snippet,
                            metadata=node.metadata
                        )
                    for edge in graph.edges:
                        G.add_edge(
                            edge.source_node,
                            edge.target_node,
                            edge_type=edge.edge_type,
                            edge_id=edge.edge_id,
                            score=edge.score,
                            metadata=edge.metadata
                        )
                    
                    terms = [t.lower() for t in q_text.split() if len(t) > 3]
                    
                    scored = []
                    for nid, ndata in G.nodes(data=True):
                        label = ndata.get('label', '').lower()
                        score = sum(1 for t in terms if t in label)
                        if score > 0:
                            scored.append((score, nid))
                    scored.sort(reverse=True)
                    start_nodes = [nid for _, nid in scored[:3]]
                    
                    if not start_nodes:
                        st.session_state.graphify_qa_error = f"No matching starting nodes found for query terms: {terms}"
                        st.session_state.graphify_qa_answer = None
                        st.session_state.graphify_qa_nodes = []
                        st.session_state.graphify_qa_edges = []
                        st.session_state.graphify_qa_start_nodes = []
                        st.session_state.graphify_qa_tokens = None
                        st.session_state.graphify_qa_budget_exceeded = False
                    else:
                        mode = "dfs" if "DFS" in mode_selection else "bfs"
                        subgraph_nodes = []
                        subgraph_edges = []
                        
                        if mode == "dfs":
                            visited = set()
                            stack = [(n, 0) for n in reversed(start_nodes)]
                            while stack:
                                node, depth = stack.pop()
                                if node in visited or depth > 6:
                                    continue
                                visited.add(node)
                                subgraph_nodes.append(node)
                                if not G.has_node(node):
                                    continue
                                for neighbor in G.neighbors(node):
                                    if neighbor not in visited:
                                        stack.append((neighbor, depth + 1))
                                        subgraph_edges.append((node, neighbor))
                        else:
                            frontier = set(start_nodes)
                            subgraph_nodes = list(start_nodes)
                            visited = set(start_nodes)
                            for _ in range(3):
                                next_frontier = set()
                                for n in frontier:
                                    if not G.has_node(n):
                                        continue
                                    for neighbor in G.neighbors(n):
                                        if neighbor not in visited:
                                            visited.add(neighbor)
                                            next_frontier.add(neighbor)
                                            subgraph_nodes.append(neighbor)
                                            subgraph_edges.append((n, neighbor))
                                frontier = next_frontier
                                
                        def relevance(nid):
                            label = G.nodes[nid].get('label', '').lower()
                            return sum(1 for t in terms if t in label)

                        ranked_nodes = sorted(subgraph_nodes, key=relevance, reverse=True)

                        lines = [f'Traversal: {mode.upper()} | Start: {[G.nodes[n].get("label", n) for n in start_nodes]} | {len(subgraph_nodes)} nodes']
                        node_details = []
                        for nid in ranked_nodes:
                            d = G.nodes[nid]
                            src = d.get("source_file") or d.get("file_path") or ""
                            loc = d.get("source_location") or d.get("line_start") or ""
                            node_str = f'  NODE {d.get("label", nid)} [src={src} loc={loc}]'
                            lines.append(node_str)
                            node_details.append(node_str)
                            
                        edge_details = []
                        for u, v in subgraph_edges:
                            if u in subgraph_nodes and v in subgraph_nodes:
                                d = G[u][v]
                                rel = d.get("relation") or d.get("edge_type") or "connected_to"
                                conf = d.get("confidence") or d.get("score") or "1.0"
                                edge_str = f'  EDGE {G.nodes[u].get("label", u)} --{rel} [{conf}]--> {G.nodes[v].get("label", v)}'
                                lines.append(edge_str)
                                edge_details.append(edge_str)

                        output = '\n'.join(lines)
                        
                        budget_exceeded = False
                        if token_budget_val is not None:
                            char_budget = token_budget_val * 4
                            if len(output) > char_budget:
                                output = output[:char_budget] + f'\n... (truncated at ~{token_budget_val} token budget - use --budget N for more)'
                                budget_exceeded = True
                                
                        rectify_str = chat_service.get_rectify_instructions() if rectify_checked else ""
                        prompt = (
                            "You are a graph-aware repository QA assistant. Use the following Graphify nodes and relationships to answer the question.\n"
                            "Cite files and lines when they are available in the node details. Answer using ONLY what the graph contains. Quote source locations when citing specific facts.\n"
                            "If the graph lacks enough information to answer, say so - do not hallucinate edges or facts.\n\n"
                            f"Question: {q_text}\n\n"
                            f"Graph Context:\n{output}"
                            f"{rectify_str}"
                        )
                        
                        response = llm_provider.generate_answer(prompt)
                        
                        st.session_state.graphify_qa_answer = response.text
                        st.session_state.graphify_qa_nodes = node_details
                        st.session_state.graphify_qa_edges = edge_details
                        st.session_state.graphify_qa_start_nodes = start_nodes
                        st.session_state.graphify_qa_prompt = prompt
                        st.session_state.graphify_qa_cli_output = None
                        st.session_state.graphify_qa_using_cli = False
                        
                        repo_tokens = chat_service._get_total_repo_tokens(repo.repo_id)
                        query_prompt = chat_service._standard_prompt(q_text, "[concatenated_files_placeholder]")
                        query_tokens = chat_service._prompt_measurement(query_prompt).tokens
                        raw_input_token_usage = repo_tokens + query_tokens

                        st.session_state.graphify_qa_tokens = {
                            "prompt": response.prompt_tokens.tokens,
                            "response": response.response_tokens.tokens,
                            "total": response.total_tokens.tokens,
                            "raw_input": raw_input_token_usage,
                            "notes": response.total_tokens.notes
                        }
                        st.session_state.graphify_qa_error = None
                        st.session_state.graphify_qa_budget_exceeded = budget_exceeded
                        st.session_state.graphify_qa_budget_limit = token_budget_val
                        st.session_state.graphify_qa_total_discovered = len(subgraph_nodes)
                        st.session_state.graphify_qa_total_included = len(subgraph_nodes)

                if response is not None:
                    # Save query to storage so it shows in Token Analytics history
                    from app.models.schemas import TokenMeasurement, CountType, QueryRecord
                    token_usage = {
                        "llm_prompt_tokens": TokenMeasurement(
                            stage="llm_prompt_tokens",
                            tokens=response.prompt_tokens.tokens,
                            count_type=CountType.exact,
                            provider="gemini",
                            model=llm_provider.model,
                        ),
                        "llm_response_tokens": TokenMeasurement(
                            stage="llm_response_tokens",
                            tokens=response.response_tokens.tokens,
                            count_type=CountType.exact,
                            provider="gemini",
                            model=llm_provider.model,
                        ),
                        "total_per_query_tokens": TokenMeasurement(
                            stage="total_per_query_tokens",
                            tokens=response.total_tokens.tokens,
                            count_type=CountType.exact,
                            provider="gemini",
                            model=llm_provider.model,
                        ),
                        "whole_codebase_baseline": TokenMeasurement(
                            stage="whole_codebase_baseline",
                            tokens=raw_input_token_usage,
                            count_type=CountType.exact if chat_service.token_service._encoding else CountType.estimated,
                            provider="gemini",
                            model=llm_provider.model,
                            notes="Raw Input token usage baseline"
                        )
                    }
                    token_usage_dicts = {}
                    for k, v in token_usage.items():
                        if hasattr(v, "model_dump"):
                            token_usage_dicts[k] = v.model_dump()
                        else:
                            token_usage_dicts[k] = v
                            
                    record = QueryRecord(
                        query_id=uuid4().hex,
                        repo_id=repo.repo_id,
                        session_id=st.session_state.session_id,
                        mode="graph_optimized",
                        query=q_text,
                        status="completed",
                        answer=response.text,
                        error=None,
                        source_snippets=[],
                        selected_nodes=[],
                        selected_edges=[],
                        token_usage=token_usage_dicts,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                    storage.save_query(record)
            except Exception as e:
                st.session_state.graphify_qa_error = f"Error during QA generation: {e}"
                st.session_state.graphify_qa_answer = None
                st.session_state.graphify_qa_nodes = []
                st.session_state.graphify_qa_edges = []
                st.session_state.graphify_qa_start_nodes = []
                st.session_state.graphify_qa_tokens = None
                st.session_state.graphify_qa_budget_exceeded = False
                st.session_state.graphify_qa_prompt = None
                st.session_state.graphify_qa_cli_output = None
                st.session_state.graphify_qa_using_cli = False

    if st.session_state.get("graphify_qa_error"):
        st.error(st.session_state.graphify_qa_error)
        
    if st.session_state.get("graphify_qa_answer"):
        st.success("Answer synthesized successfully!")
        
        tokens_info = st.session_state.graphify_qa_tokens
        if tokens_info:
            cols = st.columns(3)
            cols[0].metric("Current Token Usage", f"{tokens_info['prompt']:,}")
            cols[1].metric("Raw Input Token Usage", f"{tokens_info['raw_input']:,}")
            if tokens_info['raw_input'] > 0:
                pct_reduction = 100 - (tokens_info['prompt'] / tokens_info['raw_input']) * 100
                cols[2].metric("Token Reduction %", f"{round(pct_reduction, 2)}%")
            else:
                cols[2].metric("Token Reduction %", "0.00%")
            
        st.subheader("💡 Answer")
        st.markdown(st.session_state.graphify_qa_answer)
        
        # Check for error rectification code fix
        fix = parse_code_fix(st.session_state.graphify_qa_answer)
        if fix:
            st.divider()
            st.subheader("🛠️ Proposed Error Rectification")
            st.markdown(f"**Target File:** `{fix['filepath']}`")
            
            # Show original and replacement
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Code:**")
                st.code(fix["original_code"])
            with col2:
                st.markdown("**Proposed Replacement:**")
                st.code(fix["replacement_code"])
                
            # Render apply button
            if st.button("Apply Suggested Fix", key="apply_graphify_fix_btn", type="primary"):
                with st.spinner("Applying fix and re-analyzing codebase..."):
                    res = chat_service.apply_rectification(
                        repo.repo_id,
                        fix["filepath"],
                        fix["original_code"],
                        fix["replacement_code"]
                    )
                    if res["status"] == "success":
                        st.success(res["message"])
                        
                        # Save the updated file info for download option
                        filepath_abs = storage.repo_source_dir(repo.repo_id) / fix["filepath"]
                        if filepath_abs.exists():
                            try:
                                updated_content = filepath_abs.read_text(encoding="utf-8")
                                st.session_state.graphify_last_rectified_file = {
                                    "filename": Path(fix["filepath"]).name,
                                    "filepath": fix["filepath"],
                                    "content": updated_content
                                }
                            except Exception as e:
                                st.error(f"Error reading rectified file for download: {e}")
                        st.rerun()
                    else:
                        st.error(res["error"])
        
        with st.expander("🔬 Retrieved Subgraph Details and Raw LLM Prompt", expanded=False):
            if st.session_state.get("graphify_qa_using_cli"):
                t1, t2 = st.tabs(["📝 Graphify CLI Output", "📄 Raw Input"])
                with t1:
                    st.markdown("**Context retrieved via Graphify CLI:**")
                    st.code(st.session_state.get("graphify_qa_cli_output", ""), language="text")
                with t2:
                    st.markdown("**Exact Prompt Sent to LLM:**")
                    st.text_area("LLM Final Prompt", st.session_state.get("graphify_qa_prompt", ""), height=400)
            else:
                st.markdown(f"**Identified Start Nodes (Matching Query Terms):** {', '.join([f'`{n}`' for n in st.session_state.graphify_qa_start_nodes])}")
                
                t1, t2, t3 = st.tabs(["📝 Selected Graph Nodes", "🔗 Selected Graph Edges", "📄 Raw Input"])
                with t1:
                    st.markdown(f"**Retrieved {len(st.session_state.graphify_qa_nodes)} Nodes:**")
                    for node_str in st.session_state.graphify_qa_nodes:
                        st.markdown(node_str)
                        st.divider()
                with t2:
                    st.markdown(f"**Retrieved {len(st.session_state.graphify_qa_edges)} Edges:**")
                    for edge_str in st.session_state.graphify_qa_edges:
                        st.markdown(edge_str)
                with t3:
                    st.markdown("**Exact Prompt Sent to LLM:**")
                    st.text_area("LLM Final Prompt", st.session_state.get("graphify_qa_prompt", ""), height=400)

    # Render persistent download button at the bottom of the tab if a file was changed
    rectified_file = st.session_state.get("graphify_last_rectified_file")
    if rectified_file:
        st.divider()
        st.subheader("📥 Download Rectified File")
        st.info(f"The file `{rectified_file['filepath']}` was successfully updated via error rectification.")
        st.download_button(
            label=f"Download {rectified_file['filename']}",
            data=rectified_file["content"],
            file_name=rectified_file["filename"],
            mime="text/plain",
            key="download_graphify_rectified_file"
        )



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
        "CodeGraph QA",
        "Graphify",
        "Graphify QA",
        "Token Analytics",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_upload_import(repo)
    with tabs[1]:
        render_graph(repo, "codegraph")
    with tabs[2]:
        render_codegraph_qa(repo)
    with tabs[3]:
        render_graph(repo, "graphify")
    with tabs[4]:
        render_graphify_qa(repo)
    with tabs[5]:
        render_tokens(repo)




if __name__ == "__main__":
    main()
