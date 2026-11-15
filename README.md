# Helix — Multi-Agent GraphRAG Research Console

**Helix** is a production-oriented multi-agent system built on **LangGraph**.  
It combines hierarchical RAG, knowledge-graph reasoning, episodic memory of past tool executions, secure browser automation, and full multi-interrupt human-in-the-loop control — exposed through a hardened NiceGUI console and a streaming FastAPI bridge.

---

## Why Helix is different

Most agent demos stop at "LLM + a few tools". Helix goes further in concrete, engineering-level ways:

| Capability | Typical agents | Helix |
|---|---|---|
| **Document understanding** | Flat chunking + vector search | Hierarchical parent/child chunking → hybrid dense+sparse retrieval → parent expansion |
| **Knowledge** | Pure vector *or* pure graph | Dual-store GraphRAG (Qdrant + Neo4j) with grounding verification, fuzzy + structural entity resolution |
| **Memory** | Conversation buffer / simple vector store | Episodic memory of *successful tool-call sequences* with multi-strategy recall |
| **Browser use** | Unrestricted Playwright / Selenium | AST-validated Playwright code inside a disposable, capability-dropped Docker sandbox |
| **Control** | Fully autonomous or fully scripted | True multi-HITL: parallel tool interrupts, batch approval, argument editing, final-answer gate |
| **Architecture** | Single agent or simple router | Explicit supervisor + specialist workers with parallel dispatch |
| **Frontend** | Basic chat or Streamlit | Hardened NiceGUI console that stays alive when the backend drops, recovers automatically, and handles multi-interrupt cycles cleanly |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  NiceGUI Console (frontend/app.py)  →  http://localhost:8080     │
│  • Chat + thread management                                       │
│  • File upload linked to thread state                             │
│  • Multi-interrupt approval UI (Approve / Reject / Edit args)     │
│  • Live health indicator + automatic recovery                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼─────────────────────────────────────┐
│  FastAPI HITL Bridge (api/main.py)  →  http://localhost:8008      │
│  • Streaming chat + resume                                        │
│  • Parallel multi-interrupt collection & batch approval           │
│  • Thread / state / history / upload endpoints                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │ LangGraph SDK
┌────────────────────────────▼─────────────────────────────────────┐
│  LangGraph Agent (agent/)  →  http://localhost:2024               │
│                                                               │
│   SUPERVISOR                                                  │
│     ├── MEMER          (long-term + episodic memory)          │
│     ├── RAGGER         (hierarchical RAG + GraphRAG)          │
│     ├── WEB_SCRAPER    (live search + secure Playwright)      │
│     └── FINALIZER      (answer synthesis + final gate)        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ MCP (Streamable HTTP)
┌────────────────────────────▼─────────────────────────────────────┐
│  MCP Tool Server (mcp_server/)  →  http://localhost:8000          │
│  retrieve · clean_doc · ingest_or_list · run_playwright ·        │
│  search · recall · download_files                                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     Qdrant             Neo4j            Playwright Sandbox
  (hybrid vectors)   (knowledge +      (AST-validated,
                      episodes)         isolated browser)
```

---

## Advanced features in depth

### 1. Hierarchical RAG with parent expansion
- Title-aware chunking → recursive token-aware splitting into **parent** (context) and **child** (precise) chunks.
- Hybrid retrieval: dense (BGE) + sparse (BM25) with Reciprocal Rank Fusion.
- On hit, child chunks are expanded back to their parent chunks so the LLM receives rich surrounding context instead of isolated fragments.

### 2. GraphRAG with grounded extraction & entity resolution
- LLM extracts entities and relations **only** when explicitly supported by a source span.
- Grounding verification filters out ungrounded claims.
- Duplicate detection combines fuzzy name matching with structural neighbor-overlap scoring.
- Entities are indexed in Qdrant for semantic entry-point discovery; multi-hop traversal runs from the resolved nodes.

### 3. Episodic agent memory (procedure memory)
Successful runs are parsed into a lean graph:

```
Task ─[:AGENT_USED]→ Agent ─[:INVOKED]→ ToolCall ─[:USED]→ Tool
```

- `ToolCall` nodes are **events** (not shared hubs), preserving step order, arguments, status, and output previews.
- Ingestion is idempotent via an `Episode` ledger and deterministic Qdrant point IDs.
- Recall is multi-strategy:
  1. Vector similarity on query-shaped embeddings
  2. Lexical full-text fallback
  3. Exact graph blueprints of past successful paths
  4. Aggregate tool reliability statistics

This lets the agent reuse *working strategies*, not just remembered facts.

### 4. Secure Playwright sandbox
- User (or LLM-generated) code is first validated by a strict AST visitor:
  - Allowed: `asyncio`, `playwright.async_api`, and the injected `page` / `context` / `browser` objects
  - Blocked: `eval`, `exec`, `open`, private attributes, class/function definitions, while-loops, etc.
- Validated code runs inside a disposable container with:
  - dropped capabilities, memory/CPU/PID limits, no-new-privileges, tmpfs
  - optional noVNC view for human observation
- The host never executes arbitrary browser code.

### 5. True multi-HITL (parallel interrupts)
- Specialist agents can hit tool interrupts **in parallel**.
- The FastAPI bridge collects *all* pending interrupts from the stream **and** from authoritative thread state.
- The NiceGUI console presents a unified approval panel:
  - **Approve** / **Reject** / **Edit arguments** per tool call
  - Batch submission of the complete decision map
- A final-answer gate lets the human accept or request changes before the run is marked successful.
- The UI is hardened: it stays alive when the backend drops, shows clear online/offline status, and recovers automatically.

### 6. Specialist supervisor architecture
The supervisor never answers the user directly. It only dispatches fully self-contained tasks to:

| Specialist | Responsibility |
|---|---|
| **Memer** | Load previous threads + episodic blueprints |
| **Ragger** | Clean → ingest → hybrid retrieve over uploaded documents |
| **Web Scraper** | Live web search + optional sandboxed Playwright |
| **Finalizer** | Synthesize evidence into the final answer |

Independent tasks are dispatched together so they run in parallel.

---

## Project structure

```
.
├── README.md
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt                  # pointers to per-service files
├── requirements/
│   ├── agent.txt
│   ├── mcp_server.txt
│   ├── sandbox.txt
│   ├── api.txt
│   └── frontend.txt
│
├── frontend/                         # NiceGUI console
│   ├── app.py
│   └── requirements.txt
│
├── api/                              # FastAPI multi-HITL bridge
│   ├── main.py
│   └── requirements.txt
│
├── agent/                            # LangGraph multi-agent runtime
│   ├── graph.py · node.py · state.py · llm.py
│   ├── memer.py · rager.py · web_scraper.py
│   ├── agent_session.py · agent_memory.py · mockllm.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── mcp_server/                       # Tools + data pipelines
│   ├── tools.py                      # FastMCP server
│   ├── data_prap.py                  # hierarchical cleaning & chunking
│   ├── vec1.py                       # hybrid vector store
│   ├── gdb.py                        # GraphRAG + entity resolution
│   ├── experiences.py                # episodic memory
│   ├── loader.py · prac.py · px1.py
│   ├── Dockerfile
│   └── requirements.txt
│
└── sand_box/                         # Isolated browser environment
    ├── server.py · sandbox_runner.py
    ├── start.sh · Dockerfile
    └── requirements.txt
```

---

## Quick start

### Prerequisites
- Docker + Docker Compose
- Neo4j and Qdrant (local or remote)
- Google AI API key (or swap the LLM client)

### 1. Environment

```bash
cp .env.example .env
# fill in GOOGLE_API_KEY, Neo4j, Qdrant, etc.
```

### 2. Launch the core stack

```bash
docker compose up --build
```

| Service | Port |
|---|---|
| LangGraph Agent | 2024 |
| MCP Tool Server | 8000 |
| Playwright sandbox (noVNC) | 6080 |
| Postgres | 5433 |

### 3. Start the HITL bridge + UI

```bash
# Terminal 1 – FastAPI bridge
cd api && pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8008

# Terminal 2 – NiceGUI console
cd frontend && pip install -r requirements.txt
python app.py
# → http://localhost:8080
```

Open the console, create a thread, upload documents if needed, and start chatting. Every tool call will pause for your approval.

---

## Typical use cases

- **Research assistant** that reads your private documents *and* searches the live web, then cross-links facts through a knowledge graph.
- **Document analyst** that preserves hierarchical context instead of losing it in flat chunks.
- **Safe browser automation agent** that can run LLM-generated Playwright scripts without exposing the host.
- **Long-running agent** that remembers *how* it solved similar tasks (procedure memory), not just what the user said.
- **Human-in-the-loop workflows** where every external action is reviewable and editable before execution.

---

## Design principles

1. **Specialists over generalists** — narrow, focused system prompts and clear responsibilities.
2. **Evidence over hallucination** — final answers are synthesized only from retrieved or scraped material.
3. **Memory of procedure** — successful tool sequences are stored and reusable.
4. **Defense in depth for code execution** — AST validation + container isolation + resource limits.
5. **Human remains in the loop** — interrupts on every tool call and a final approval gate.
6. **UI resilience** — the console never crashes when the backend is offline; it recovers automatically.

---

## Roadmap

- [ ] Replace mock LLMs with production models and full streaming
- [ ] Persist episodic memory across restarts with stronger schema migrations
- [ ] Multi-user / multi-tenant isolation
- [ ] Additional document types and OCR
- [ ] Optional web UI polish with the original Helix brand assets

---

## License

Add your preferred license (MIT or Apache-2.0 recommended).

---

Built with LangGraph, Qdrant, Neo4j, FastEmbed, Playwright, FastAPI, NiceGUI, and careful systems engineering.
