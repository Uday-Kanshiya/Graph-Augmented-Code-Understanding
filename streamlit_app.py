from __future__ import annotations

import os
import shutil
import sys
import textwrap
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

from app.models.schemas import GraphDocument, QueryRecord, RepoMetadata, TreeNode  # noqa: E402
from app.services.analysis_pipeline import AnalysisPipeline  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.codegraph_service import CodeGraphService  # noqa: E402
from app.services.file_utils import clean_repo_name, safe_extract_zip  # noqa: E402
from app.services.graph_retrieval_service import GraphRetrievalService  # noqa: E402
from app.services.graphify_service import GraphifyService  # noqa: E402
from app.services.llm.gemini import GeminiProvider  # noqa: E402
from app.services.repo_service import RepoService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402
from app.services.storage import LocalStorage  # noqa: E402
from app.services.token_service import TokenService  # noqa: E402
from app.services.tree_sitter_service import TreeSitterService  # noqa: E402


st.set_page_config(
    page_title="Context Optimization Engine",
    layout="wide",
    initial_sidebar_state="expanded",
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
    chat_service = ChatService(
        storage=storage,
        retrieval_service=RetrievalService(storage=storage, token_service=token_service),
        graph_retrieval_service=GraphRetrievalService(storage=storage, token_service=token_service),
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


def metric_row(repo: RepoMetadata) -> None:
    cols = st.columns(4)
    cols[0].metric("Total files", repo.stats.total_files)
    cols[1].metric("Python files", repo.stats.python_files)
    cols[2].metric("Total lines", repo.stats.total_lines)
    cols[3].metric("Python lines", repo.stats.python_lines)


def render_status(repo: RepoMetadata | None) -> None:
    model_info = llm_provider.get_model_info()
    st.sidebar.subheader("Status")
    st.sidebar.write(f"Repo: **{repo.name if repo else 'none'}**")
    st.sidebar.write(f"Pipeline: **{repo.status if repo else 'idle'}**")
    st.sidebar.write(f"Model: **{model_info.model}**")
    st.sidebar.write(f"Gemini key: **{'configured' if model_info.configured else 'missing'}**")
    
    st.sidebar.subheader("Context Size Budget")
    budget_label = st.sidebar.selectbox(
        "Retrieval Scale Selector",
        ["Balanced (Recommended)", "Tight (Token Saver)", "Deep (Large Codebase Coverage)"],
        index=0,
        help="Adjust context bounds: Balanced is ideal for normal use, Tight minimizes token cost, and Deep is suited for large repositories with many files."
    )
    
    if budget_label == "Tight (Token Saver)":
        st.session_state.graph_max_nodes = 8
        anchors, neighbors = 2, 4
    elif budget_label == "Deep (Large Codebase Coverage)":
        st.session_state.graph_max_nodes = 24
        anchors, neighbors = 8, 16
    else: # Balanced
        st.session_state.graph_max_nodes = 14
        anchors, neighbors = 4, 8
        
    st.sidebar.caption(
        f"**Active Limits:** Anchors (Full): `{anchors}` | Neighbors (Signature): `{neighbors}`"
    )

    
    st.sidebar.subheader("Codebase Rectifier")
    st.session_state.rectify_enabled = st.sidebar.checkbox(
        "🔍 Enable Error Rectification",
        value=False,
        help="When enabled, the assistant will propose direct code modifications when bugs or errors are identified, allowing you to overwrite files with a single click."
    )

    
    if repo and repo.warnings:
        with st.sidebar.expander("Warnings", expanded=True):
            for warning in repo.warnings:
                render_notice(warning)


def render_notice(message: str) -> None:
    if "Graphify" in message and "fallback" in message.lower():
        st.info(message)
        return
    st.warning(message)


def render_logs(repo: RepoMetadata | None) -> None:
    st.subheader("Developer / Debug Logs")
    if not repo:
        st.info("Load a repository to see pipeline logs.")
        return
    logs = storage.load_logs(repo.repo_id)
    if not logs:
        st.info("No logs yet.")
        return
    st.dataframe(logs, use_container_width=True, hide_index=True)


def render_upload_import() -> None:
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


def render_repo_analysis(repo: RepoMetadata | None) -> None:
    st.header("Repo Analysis")
    if not repo:
        st.info("Load a repository first.")
        return
    metric_row(repo)
    st.write(f"Origin: `{repo.origin}`")
    if repo.error:
        st.error(repo.error)
    files = storage.load_files(repo.repo_id)
    st.subheader("Python Files")
    st.dataframe([file.model_dump() for file in files], use_container_width=True, hide_index=True)
    render_logs(repo)


def node_label(node: TreeNode) -> str:
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    suffix = f" | {node.text_preview}" if node.text_preview else ""
    return f"{node.type} [{start_line}:{node.start_point[1]} - {end_line}:{node.end_point[1]}]{suffix}"


def collect_tree_rows(
    node: TreeNode,
    depth: int,
    max_depth: int,
    budget: list[int],
    rows: list[dict[str, str | int | bool | None]],
) -> None:
    if budget[0] <= 0:
        return
    budget[0] -= 1
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    rows.append(
        {
            "tree": f"{'  ' * depth}{node.type}",
            "depth": depth,
            "named": node.named,
            "start": f"{start_line}:{node.start_point[1]}",
            "end": f"{end_line}:{node.end_point[1]}",
            "preview": node.text_preview,
        }
    )
    if depth < max_depth:
        for child in node.children:
            collect_tree_rows(child, depth + 1, max_depth, budget, rows)


def flatten_named_nodes(node: TreeNode, output: list[TreeNode], limit: int = 500) -> None:
    if len(output) >= limit:
        return
    if node.named:
        output.append(node)
    for child in node.children:
        flatten_named_nodes(child, output, limit)


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', "'").replace("\n", "\\n")


def tree_to_dot(root: TreeNode, max_depth: int = 5, max_nodes: int = 180) -> tuple[str, bool]:
    lines = [
        "digraph TreeSitter {",
        "rankdir=TB;",
        'node [shape=box, style="rounded,filled", fillcolor="#fbfcfa", color="#bfc8c2", fontsize=10];',
        'edge [color="#6d4b7d"];',
    ]
    counter = {"value": 0}
    truncated = {"value": False}

    def visit(node: TreeNode, depth: int) -> str | None:
        if counter["value"] >= max_nodes:
            truncated["value"] = True
            return None
        current_id = f"n{counter['value']}"
        counter["value"] += 1
        start_line = node.start_point[0] + 1
        label = dot_escape(f"{node.type}\\nL{start_line}")
        fill = "#e6f4f1" if node.named else "#f7f8f6"
        lines.append(f'"{current_id}" [label="{label}", fillcolor="{fill}"];')
        if depth < max_depth:
            for child in node.children:
                child_id = visit(child, depth + 1)
                if child_id:
                    lines.append(f'"{current_id}" -> "{child_id}";')
        elif node.children:
            truncated["value"] = True
        return current_id

    visit(root, 0)
    lines.append("}")
    return "\n".join(lines), truncated["value"]


def render_tree_sitter(repo: RepoMetadata | None) -> None:
    st.header("Tree-sitter Explorer")
    if not repo:
        st.info("Load a repository first.")
        return
    files = storage.load_files(repo.repo_id)
    if not files:
        st.info("No Python files available.")
        return
    selected = st.selectbox(
        "File",
        [file.path for file in files],
        index=0,
        key="tree_file_select",
    )
    document = storage.load_tree_sitter(repo.repo_id, selected)
    if not document:
        st.error("Tree-sitter output not found.")
        return
    if document.warnings:
        for warning in document.warnings:
            st.warning(warning)
    if document.parse_error:
        st.error(document.parse_error)
        st.code(document.source, language="python")
        return

    left, right = st.columns([0.42, 0.58])
    with left:
        st.subheader("Parse Tree")
        max_depth = st.slider("Expansion depth", 2, 8, 5)
        max_nodes = st.slider("Rendered nodes", 50, 1000, 300, step=50)
        if document.root:
            rows: list[dict[str, str | int | bool | None]] = []
            collect_tree_rows(document.root, 0, max_depth, [max_nodes], rows)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.subheader("Parse Tree Graph")
            dot, truncated = tree_to_dot(document.root, max_depth=max_depth, max_nodes=min(max_nodes, 220))
            if truncated:
                st.info("Graph view is truncated by the depth/node controls to keep the page responsive.")
            st.graphviz_chart(dot, use_container_width=True)
    with right:
        st.subheader("Source Span")
        if document.root:
            named_nodes: list[TreeNode] = []
            flatten_named_nodes(document.root, named_nodes)
            labels = [node_label(node) for node in named_nodes]
            selected_index = st.selectbox("Highlight node", range(len(labels)), format_func=lambda index: labels[index])
            chosen = named_nodes[selected_index]
            start_line = chosen.start_point[0] + 1
            end_line = chosen.end_point[0] + 1
            lines = document.source.splitlines()
            snippet = "\n".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])
            st.caption(f"{selected}:{start_line}-{end_line}")
            st.code(snippet or document.source, language="python", line_numbers=True)


def graph_to_dot(graph: GraphDocument, max_nodes: int = 80, max_edges: int = 160) -> str:
    visible_nodes = graph.nodes[:max_nodes]
    visible = {node.node_id for node in visible_nodes}
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#f7f8f6", color="#bfc8c2"];']
    for node in visible_nodes:
        label = f"{node.node_type}\\n{node.label}".replace('"', "'")
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
            
        st.subheader("📜 All Query History & Token Usage")
        history_rows = []
        for q_rec in queries:
            prompt_meas = q_rec.token_usage.get("llm_prompt_tokens")
            resp_meas = q_rec.token_usage.get("llm_response_tokens")
            total_meas = q_rec.token_usage.get("total_per_query_tokens")
            
            history_rows.append({
                "Time": q_rec.created_at.strftime("%H:%M:%S") if q_rec.created_at else "N/A",
                "Mode": "Graph-Optimized" if q_rec.mode == "graph_optimized" else "Standard",
                "Question": q_rec.query,
                "LLM Prompt (Input)": prompt_meas.tokens if prompt_meas else 0,
                "LLM Response (Output)": resp_meas.tokens if resp_meas else 0,
                "Total Query Tokens": total_meas.tokens if total_meas else 0,
                "Latency": f"{q_rec.latency_ms} ms"
            })
        st.dataframe(history_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No queries have been run in this session yet. Go to Graph QA to ask a question!")

    # Collapsible Ingestion Pipeline Measurements
    summary = storage.load_token_summary(repo.repo_id)
    if summary:
        with st.expander("🛠️ Static Ingestion & Parsing Pipeline Measurements (Raw Code & Graph Build)"):
            st.markdown(
                "These tokens represent the static sizes of the files and graphs during ingestion and parsing. "
                "They are stored locally and are **not** sent to the LLM on every query."
            )
            pipeline_rows = [measurement.model_dump() for measurement in summary.stages.values()]
            st.dataframe(pipeline_rows, use_container_width=True, hide_index=True)
            
            if summary.cumulative_session_usage:
                st.markdown("#### Cumulative Raw Stage Measurements")
                st.dataframe(
                    [{"stage": key, "tokens": value} for key, value in summary.cumulative_session_usage.items()],
                    use_container_width=True,
                    hide_index=True,
                )


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
        'node [shape=box, style="rounded,filled", fontname="Courier New", fontsize=9];'
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
            # Gold premium outline for Primary Anchors (full bodies)
            lines.append(
                f'"{node_id_escaped}" [label="{label}", fillcolor="#fff3bf", color="#f08c00", penwidth=2.5];'
            )
        else:
            # Sleek slate gray outline for Neighbor Nodes (signatures)
            lines.append(
                f'"{node_id_escaped}" [label="{label}", fillcolor="#f1f3f5", color="#adb5bd", penwidth=1.2];'
            )
            
    # Render edges
    visible_node_ids = {node.node_id for node in nodes}
    for edge in edges:
        if edge.source_node in visible_node_ids and edge.target_node in visible_node_ids:
            label = edge.edge_type.replace('"', "'")
            lines.append(
                f'"{edge.source_node}" -> "{edge.target_node}" [label="{label}", color="#495057", fontcolor="#495057", fontsize=8];'
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


def render_standard_qa(repo: RepoMetadata | None) -> None:
    st.header("Standard Repo QA")
    if not repo:
        st.info("Load a repository first.")
        return
    query = st.text_area("Question", placeholder=qa_prompt_help(), key="standard_query")
    if st.button("Ask Standard QA", disabled=not query.strip(), type="primary"):
        with st.spinner("Building chunk context and calling Gemini..."):
            record = chat_service.standard_qa(
                repo.repo_id,
                query.strip(),
                st.session_state.session_id,
                limit=st.session_state.get("standard_limit", 8),
                rectify=st.session_state.get("rectify_enabled", False)
            )
            st.session_state.standard_record = record
    if "standard_record" in st.session_state:
        render_query_record(st.session_state.standard_record)


def render_graph_qa(repo: RepoMetadata | None) -> None:
    st.header("Graph-Optimized Repo QA")
    if not repo:
        st.info("Load a repository first.")
        return
    codegraph = storage.load_codegraph(repo.repo_id)
    graphify = storage.load_graphify(repo.repo_id)
    st.caption(
        f"Graph QA retrieves from CodeGraph ({len(codegraph.nodes) if codegraph else 0} nodes) plus "
        f"{graphify.source if graphify else 'missing Graphify'} ({len(graphify.nodes) if graphify else 0} nodes)."
    )
    
    # Expose graph source selection dropdown
    source_label = st.selectbox(
        "Retrieve Graph Context from:",
        ["Merged CodeGraph + Graphify", "CodeGraph (Static Structure) Only", "Graphify (Flow Analysis) Only"],
        index=0,
        key="graph_source_select"
    )
    source_selection = "merged"
    if "Static Structure" in source_label:
        source_selection = "codegraph"
    elif "Flow Analysis" in source_label:
        source_selection = "graphify"

    query = st.text_area("Question", placeholder=qa_prompt_help(), key="graph_query")
    if st.button("Ask Graph QA", disabled=not query.strip(), type="primary"):
        with st.spinner("Selecting graph neighborhood and calling Gemini..."):
            record = chat_service.graph_optimized_qa(
                repo.repo_id,
                query.strip(),
                st.session_state.session_id,
                source_selection=source_selection,
                max_nodes=st.session_state.get("graph_max_nodes", 8),
                rectify=st.session_state.get("rectify_enabled", False)
            )
            st.session_state.graph_record = record

    if "graph_record" in st.session_state:
        render_query_record(st.session_state.graph_record)


def render_compare(repo: RepoMetadata | None) -> None:
    st.header("Compare Baseline vs Graph-Optimized")
    if not repo:
        st.info("Load a repository first.")
        return
    st.markdown(
        "Run a comparison to see how much **LLM Prompt (input) tokens** are reduced using Graph-Optimized QA compared to Standard Retrieval QA. "
        "Reducing prompt tokens directly helps you stay within LLM context window limits and saves cost."
    )
    
    # Expose graph source selection dropdown for comparison
    source_label = st.selectbox(
        "Retrieve Graph Context from:",
        ["Merged CodeGraph + Graphify", "CodeGraph (Static Structure) Only", "Graphify (Flow Analysis) Only"],
        index=0,
        key="compare_source_select"
    )
    source_selection = "merged"
    if "Static Structure" in source_label:
        source_selection = "codegraph"
    elif "Flow Analysis" in source_label:
        source_selection = "graphify"

    query = st.text_area("Question", placeholder="Run the same query through both modes.", key="compare_query")
    if st.button("Run comparison", disabled=not query.strip(), type="primary"):
        with st.spinner("Running both QA modes..."):
            st.session_state.compare_result = chat_service.compare(
                repo.repo_id,
                query.strip(),
                st.session_state.session_id,
                source_selection=source_selection,
                limit=st.session_state.get("standard_limit", 8),
                max_nodes=st.session_state.get("graph_max_nodes", 8)
            )
    result = st.session_state.get("compare_result")
    if not result:
        return
        
    st.subheader("LLM Prompt Token Reduction Analysis")
    
    # Read the values safely with fallback for older query records
    is_new_format = "baseline_prompt_tokens" in result.token_savings
    
    baseline_val = result.token_savings.get("baseline_prompt_tokens") if is_new_format else result.token_savings.get("baseline_context_tokens", 0)
    optimized_val = result.token_savings.get("optimized_prompt_tokens") if is_new_format else result.token_savings.get("optimized_context_tokens", 0)
    saved_val = result.token_savings.get("saved_prompt_tokens") if is_new_format else result.token_savings.get("saved_context_tokens", 0)
    saved_pct = result.token_savings.get("saved_percent", 0)
    
    cols = st.columns(4)
    cols[0].metric("Baseline LLM Prompt", f"{baseline_val:,} tokens" if isinstance(baseline_val, int) else f"{baseline_val} tokens")
    cols[1].metric("Optimized LLM Prompt", f"{optimized_val:,} tokens" if isinstance(optimized_val, int) else f"{optimized_val} tokens")
    cols[2].metric("Saved Prompt Tokens", f"{saved_val:,} tokens" if isinstance(saved_val, int) else f"{saved_val} tokens")
    cols[3].metric("Saved %", f"{saved_pct}%")
    
    if is_new_format:
        st.caption(
            f"**Context alone:** Standard context was {result.token_savings.get('baseline_context_tokens', 0):,} tokens vs "
            f"Graph-optimized context of {result.token_savings.get('optimized_context_tokens', 0):,} tokens "
            f"(Context tokens reduced by **{result.token_savings.get('saved_context_percent', 0)}%**)."
        )
    else:
        st.caption("Showing estimated context tokens (fallback for older records).")
        
    left, right = st.columns(2)
    with left:
        st.subheader("Standard QA (No Graph)")
        render_query_record(result.standard)
    with right:
        st.subheader("Graph-Optimized QA")
        render_query_record(result.graph_optimized)


def render_architecture_note() -> None:
    with st.expander("What this public demo is doing"):
        st.markdown(
            textwrap.dedent(
                """
                This Streamlit app reuses the same Stage 1 engine:

                - Tree-sitter parses Python files with real source spans.
                - CodeGraph is built from Python AST relationships.
                - Graphify is attempted through a local CLI and falls back transparently when unavailable.
                - Token counts are labeled exact or estimated.
                - Gemini calls use secrets or environment variables, never hardcoded keys.

                On Streamlit Community Cloud, local storage is ephemeral. That is fine for a mentor demo, but a production version
                should move artifacts to durable storage.
                """
            )
        )


def main() -> None:
    st.title("Context Optimization Engine")
    st.caption("Stage 1 public demo: Python repo ingestion, Tree-sitter, CodeGraph, Graphify, token accounting, Gemini QA.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

    repo = current_repo()
    render_status(repo)
    render_architecture_note()

    tab_names = [
        "Upload / Import",
        "Repo Analysis",
        "Tree-sitter",
        "CodeGraph",
        "Graphify",
        "Graph QA",
        "Token Analytics",
    ]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_upload_import()
    with tabs[1]:
        render_repo_analysis(repo)
    with tabs[2]:
        render_tree_sitter(repo)
    with tabs[3]:
        render_graph(repo, "codegraph")
    with tabs[4]:
        render_graph(repo, "graphify")
    with tabs[5]:
        render_graph_qa(repo)
    with tabs[6]:
        render_tokens(repo)




if __name__ == "__main__":
    main()
