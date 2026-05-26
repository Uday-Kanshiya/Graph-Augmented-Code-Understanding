from __future__ import annotations

import os
import shutil
import sys
import textwrap
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
        st.subheader("Upload zipped Python repo")
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
        "Stage 1 is Python-focused. The app skips .git, virtualenvs, node_modules, build outputs, caches, and bytecode folders."
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


def render_tokens(repo: RepoMetadata | None) -> None:
    st.header("Token Analytics")
    if not repo:
        st.info("Load a repository first.")
        return
    summary = storage.load_token_summary(repo.repo_id)
    if not summary:
        st.error("Token summary not found.")
        return
    rows = [measurement.model_dump() for measurement in summary.stages.values()]
    st.subheader("Pipeline Token Measurements")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.subheader("Cumulative Session Usage")
    if summary.cumulative_session_usage:
        st.dataframe(
            [{"stage": key, "tokens": value} for key, value in summary.cumulative_session_usage.items()],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No query usage yet.")


def render_query_record(record: QueryRecord) -> None:
    if record.error:
        st.error(record.error)
    st.caption(f"{record.status} | {record.latency_ms} ms | query_id={record.query_id}")
    st.markdown(record.answer or "_No answer generated._")
    st.subheader("Token Usage")
    st.dataframe([value.model_dump() for value in record.token_usage.values()], use_container_width=True, hide_index=True)
    if record.selected_nodes:
        counts = graph_node_source_counts(record)
        cols = st.columns(3)
        cols[0].metric("CodeGraph nodes", counts["codegraph"])
        cols[1].metric("Native Graphify nodes", counts["graphify"])
        cols[2].metric("Fallback Graphify nodes", counts["graphify_fallback"])
        with st.expander("Selected Graph Nodes"):
            st.dataframe([node.model_dump() for node in record.selected_nodes], use_container_width=True, hide_index=True)
    with st.expander("Source Snippets", expanded=True):
        for snippet in record.source_snippets:
            st.caption(f"{snippet.file_path}:{snippet.line_start}-{snippet.line_end} | {snippet.source}")
            st.code(snippet.text, language="python", line_numbers=True)


def graph_node_source_counts(record: QueryRecord) -> dict[str, int]:
    counts = {"codegraph": 0, "graphify": 0, "graphify_fallback": 0}
    for node in record.selected_nodes:
        if node.node_id.startswith("graphify-fallback:"):
            counts["graphify_fallback"] += 1
        elif node.node_id.startswith("graphify:"):
            counts["graphify"] += 1
        elif node.node_id.startswith("codegraph:"):
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
            record = chat_service.standard_qa(repo.repo_id, query.strip(), st.session_state.session_id)
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
    query = st.text_area("Question", placeholder=qa_prompt_help(), key="graph_query")
    if st.button("Ask Graph QA", disabled=not query.strip(), type="primary"):
        with st.spinner("Selecting graph neighborhood and calling Gemini..."):
            record = chat_service.graph_optimized_qa(repo.repo_id, query.strip(), st.session_state.session_id)
            st.session_state.graph_record = record
    if "graph_record" in st.session_state:
        render_query_record(st.session_state.graph_record)


def render_compare(repo: RepoMetadata | None) -> None:
    st.header("Compare Baseline vs Graph-Optimized")
    if not repo:
        st.info("Load a repository first.")
        return
    query = st.text_area("Question", placeholder="Run the same query through both modes.", key="compare_query")
    if st.button("Run comparison", disabled=not query.strip(), type="primary"):
        with st.spinner("Running both QA modes..."):
            st.session_state.compare_result = chat_service.compare(repo.repo_id, query.strip(), st.session_state.session_id)
    result = st.session_state.get("compare_result")
    if not result:
        return
    cols = st.columns(4)
    cols[0].metric("Baseline context", result.token_savings.get("baseline_context_tokens", 0))
    cols[1].metric("Optimized context", result.token_savings.get("optimized_context_tokens", 0))
    cols[2].metric("Saved tokens", result.token_savings.get("saved_context_tokens", 0))
    cols[3].metric("Saved %", f"{result.token_savings.get('saved_percent', 0)}%")
    left, right = st.columns(2)
    with left:
        st.subheader("Standard")
        render_query_record(result.standard)
    with right:
        st.subheader("Graph Optimized")
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
        "Token Analytics",
        "Standard QA",
        "Graph QA",
        "Compare",
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
        render_tokens(repo)
    with tabs[6]:
        render_standard_qa(repo)
    with tabs[7]:
        render_graph_qa(repo)
    with tabs[8]:
        render_compare(repo)


if __name__ == "__main__":
    main()
