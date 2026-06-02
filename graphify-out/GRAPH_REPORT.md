# Graph Report - Context-Optimization  (2026-06-02)

## Corpus Check
- 40 files · ~16,701 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 323 nodes · 558 edges · 25 communities (16 shown, 9 thin omitted)
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 115 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]

## God Nodes (most connected - your core abstractions)
1. `LocalStorage` - 43 edges
2. `ChatService` - 21 edges
3. `CodeGraphService` - 18 edges
4. `GraphRetrievalService` - 16 edges
5. `compilerOptions` - 16 edges
6. `AnalysisPipeline` - 14 edges
7. `GeminiProvider` - 13 edges
8. `services()` - 12 edges
9. `TokenMeasurement` - 11 edges
10. `main()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `services()` --calls--> `LocalStorage`  [INFERRED]
  streamlit_app.py → backend/app/services/storage.py
- `services()` --calls--> `CodeGraphService`  [INFERRED]
  streamlit_app.py → backend/app/services/codegraph_service.py
- `services()` --calls--> `GeminiProvider`  [INFERRED]
  streamlit_app.py → backend/app/services/llm/gemini.py
- `services()` --calls--> `ChatService`  [INFERRED]
  streamlit_app.py → backend/app/services/chat_service.py
- `services()` --calls--> `RetrievalService`  [INFERRED]
  streamlit_app.py → backend/app/services/retrieval_service.py

## Communities (25 total, 9 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (33): BaseModel, Enum, LLMConfigurationError, LLMProvider, LLMResponse, client(), GeminiProvider, LLMProvider (+25 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (23): TabId, tabs, GraphView(), typeColors, TokenTable(), TreeExplorer(), api, CompareResult (+15 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (15): get_secret(), services(), AnalysisPipeline, clean_repo_name(), _flatten_single_top_level_dir(), is_ignored(), iter_code_files(), iter_python_files() (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (9): compare_chat(), graph_optimized_chat(), import_github(), repo_files(), standard_chat(), upload_repo(), ChatService, RectificationService (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (28): collect_tree_rows(), current_repo(), flatten_named_nodes(), graph_node_source_counts(), graph_to_dot(), ingest_uploaded_zip(), main(), metric_row() (+20 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (23): dependencies, lucide-react, next, react, react-dom, @xyflow/react, devDependencies, autoprefixer (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.10
Nodes (19): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (16): Architecture, code:powershell (python -m venv .venv), code:toml (GEMINI_API_KEY = "your_key_here"), code:powershell (Copy-Item .env.example .env), code:text (GEMINI_API_KEY=your_key_here), code:powershell (cd backend), code:powershell (cd frontend), code:text (backend/) (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.50
Nodes (3): BaseSettings, get_settings(), Settings

## Knowledge Gaps
- **60 isolated node(s):** `nextConfig`, `name`, `version`, `private`, `dev` (+55 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LocalStorage` connect `Community 5` to `Community 0`, `Community 10`, `Community 2`, `Community 3`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `services()` connect `Community 2` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `ChatService` connect `Community 3` to `Community 0`, `Community 10`, `Community 2`, `Community 5`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `LocalStorage` (e.g. with `AnalysisPipeline` and `ChatService`) actually correct?**
  _`LocalStorage` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `ChatService` (e.g. with `CompareResult` and `CountType`) actually correct?**
  _`ChatService` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `CodeGraphService` (e.g. with `AnalysisPipeline` and `GraphDocument`) actually correct?**
  _`CodeGraphService` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `str` (e.g. with `get_secret()` and `render_upload_import()`) actually correct?**
  _`str` has 15 INFERRED edges - model-reasoned connections that need verification._