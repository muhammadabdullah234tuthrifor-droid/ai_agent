"""
experiences.py — Episodic agent memory over Neo4j (graph) + Qdrant (vectors). v2

WHAT CHANGED VS v1 (short version):
 1. Tool calls are now EVENT nodes (:ToolCall), one per invocation.
    -> No more shared-hub fan-out, args are never overwritten, step order is kept.
 2. Result nodes are gone. Status/output live ON the ToolCall node (leaner graph).
 3. Episode ledger (:Episode) makes ingestion IDEMPOTENT (re-runs never duplicate).
 4. Qdrant point ids are deterministic (uuid5 of episode_id) -> upserts, not dupes.
 5. Embedding text is query-shaped (user query + tools), not a dict repr.
 6. Retrieval is now MULTI-STRATEGY: vector (with score threshold + payload
    filters), lexical fulltext fallback, exact graph recall by id, and
    aggregate tool-reliability stats. Composed in recall_similar_executions().
 7. All writes are UNWIND-batched inside ONE transaction per episode (atomic).
 8. Uniqueness constraints + fulltext index created by ensure_schema().
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Optional

from dotenv import load_dotenv
from fastembed import TextEmbedding
from neo4j import Driver, GraphDatabase, ManagedTransaction
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from logger import get_logger

logger = get_logger(name="agent_memory")


# ==========================================================
# 1. PYDANTIC MODELS
# ==========================================================
class NodeType(str, Enum):
    TASK = "Task"
    AGENT = "Agent"
    TOOL = "Tool"
    CALL = "ToolCall"  # NEW: one node per tool invocation (an "event")


class RelationType(str, Enum):
    AGENT_USED = "AGENT_USED"  # (Task)->(Agent)   props: role, reasoning
    INVOKED = "INVOKED"        # (Agent)->(ToolCall) props: seq, episode_id
    USED = "USED"              # (ToolCall)->(Tool)  props: episode_id


class ExtractedEntity(BaseModel):
    id: str
    type: NodeType
    props: dict[str, Any] = Field(default_factory=dict)


class ExtractedRelation(BaseModel):
    source_id: str
    target_id: str
    type: RelationType
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    episode_id: str
    succeeded: bool
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# ==========================================================
# 2. SMALL HELPERS
# ==========================================================
def _short_hash(*parts: str, length: int = 12) -> str:
    """Stable, collision-resistant id fragment from arbitrary strings."""
    return sha256("::".join(parts).encode("utf-8")).hexdigest()[:length]


def _args_hash(args: Any) -> str:
    return _short_hash(json.dumps(args, sort_keys=True, default=str))


def _truncate(text: Any, limit: int = 200) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    return s if len(s) <= limit else s[: limit - 3] + "..."


_LUCENE_SPECIAL = re.compile(r'[+\-!(){}\[\]^"~*?:\\/&|]+')


def _to_lucene(text: str, max_terms: int = 8) -> str:
    """Turn free text into a safe Lucene OR-query for the fulltext fallback."""
    cleaned = _LUCENE_SPECIAL.sub(" ", text or "")
    terms = [t for t in cleaned.split() if len(t) > 2]
    return " OR ".join(terms[:max_terms])


def sanitize_for_neo4j(value: Any) -> Any:
    """
    Neo4j property values may NOT be nested maps. Flat lists of primitives are
    native; anything nested is serialized to a JSON string so we never crash
    on write and can always json.loads() it back on read.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, list) and all(
        isinstance(x, (str, int, float, bool, type(None))) for x in value
    ):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def sanitize_props(props: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not props:
        return {}
    return {k: sanitize_for_neo4j(v) for k, v in props.items()}


# ==========================================================
# 3. PARSING LOGS -> GRAPH STRUCTURE
# ==========================================================
def _load_raw_logs(raw_logs_path: str) -> list[Any]:
    with open(raw_logs_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _extract_episode_id(raw_logs: list[Any], content_fallback: str) -> str:
    """Prefer an explicit episode id from the log; else hash the content so
    re-ingesting the SAME file is idempotent."""
    for item in raw_logs:
        if isinstance(item, dict):
            for key in ("episode_id", "run_id", "session_id"):
                if item.get(key):
                    return str(item[key])
    return _short_hash(content_fallback or json.dumps(raw_logs, default=str), length=16)


def _episode_succeeded(raw_logs: list[Any]) -> bool:
    """Scan from the END for the first dict carrying a result key.
    v1 broke on the first dict containing 'Result'; we keep scanning backwards
    until we actually find a result-bearing entry."""
    for item in reversed(raw_logs):
        if not isinstance(item, dict):
            continue
        for key in ("Result", "result", "final_result"):
            if key in item:
                return str(item[key]).strip().lower() in {
                    "success", "succeeded", "ok", "true",
                }
    return False


def _call_status(item: dict) -> str:
    """Per-tool status if the log provides one; episode gate already passed,
    so default to success rather than dropping information."""
    for key in ("status", "tool_status"):
        if key in item:
            return "success" if str(item[key]).strip().lower() in {
                "success", "succeeded", "ok", "true",
            } else "failure"
    return "success"


def parse_agent_logs_to_graph(raw_logs_path: str) -> ExtractionResult:
    """
    Parse one episode of agent logs into entities+relations.
    Failed episodes return an EMPTY result with succeeded=False (memory
    poisoning guard). Tool calls become ToolCall event nodes with seq order,
    args, args_hash, status, output preview and repeat_count.
    """
    with open(raw_logs_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    raw_logs = _load_raw_logs(raw_logs_path)
    episode_id = _extract_episode_id(raw_logs, content)
    succeeded = _episode_succeeded(raw_logs)

    result = ExtractionResult(episode_id=episode_id, succeeded=succeeded)
    if not succeeded:
        logger.info(
            "Episode %s did not succeed -> skipping to prevent memory poisoning.",
            episode_id,
        )
        return result

    entities: dict[str, ExtractedEntity] = {}
    relations: list[ExtractedRelation] = []

    task_id = f"task_{episode_id}"
    entities[task_id] = ExtractedEntity(
        id=task_id,
        type=NodeType.TASK,
        props={
            "user_query": "",
            "description": "Overall user request processed by super agent",
            "final_outcome": "Success",
            "episode_id": episode_id,
            "log_source": raw_logs_path,
        },
    )

    agent_rel_index: dict[str, ExtractedRelation] = {}
    seen_calls: dict[str, str] = {}  # (agent, tool, args_hash) -> ToolCall id
    seq = 0

    for item in raw_logs:
        if not isinstance(item, dict):
            continue

        # ---------- Task delegation: (Task)-[:AGENT_USED]->(Agent) ----------
        if "tasks" in item:
            reasoning = item.get("reasoning")
            user_query = item.get("task_description")
            if user_query:
                entities[task_id].props["user_query"] = str(user_query)
            for sub_task in item["tasks"]:
                if not isinstance(sub_task, dict):
                    continue
                agent_name = sub_task.get("agent")
                if not agent_name:
                    continue
                if agent_name not in entities:
                    entities[agent_name] = ExtractedEntity(
                        id=agent_name,
                        type=NodeType.AGENT,
                        props={"name": agent_name},
                    )
                if agent_name in agent_rel_index:
                    rel = agent_rel_index[agent_name]
                    if reasoning and reasoning not in rel.properties.setdefault(
                        "reasonings", []
                    ):
                        rel.properties["reasonings"].append(reasoning)
                    continue
                rel = ExtractedRelation(
                    source_id=task_id,
                    target_id=agent_name,
                    type=RelationType.AGENT_USED,
                    properties={
                        "role": sub_task.get("task") or "",
                        "reasoning": reasoning or "",
                        "episode_id": episode_id,
                    },
                )
                agent_rel_index[agent_name] = rel
                relations.append(rel)

        # ---------- Tool execution: (Agent)-[:INVOKED]->(ToolCall)-[:USED]->(Tool) ----------
        elif "tool_name" in item:
            tool_name = item["tool_name"]
            caller = item.get("caller_agent") or ""
            args = item.get("tool_args", {}) or {}
            status = _call_status(item)
            output = item.get("output", item.get("observation", item.get("tool_output")))
            ahash = _args_hash(args)
            dedupe_key = f"{caller}|{tool_name}|{ahash}"

            if dedupe_key in seen_calls:
                # Same agent, same tool, same args = redundant retry.
                call = entities[seen_calls[dedupe_key]]
                call.props["repeat_count"] = call.props.get("repeat_count", 1) + 1
                if status == "success":
                    call.props["status"] = "success"
                    if output:
                        call.props["output_preview"] = _truncate(output)
                continue

            seq += 1
            call_id = f"call_{episode_id}_{seq:03d}"
            seen_calls[dedupe_key] = call_id
            entities[call_id] = ExtractedEntity(
                id=call_id,
                type=NodeType.CALL,
                props={
                    "tool": tool_name,
                    "agent": caller,
                    "args": args,               # sanitized (JSON string) at write time
                    "arg_keys": sorted(args.keys()) if isinstance(args, dict) else [],
                    "args_hash": ahash,
                    "status": status,
                    "output_preview": _truncate(output) if output else "",
                    "seq": seq,
                    "repeat_count": 1,
                    "episode_id": episode_id,
                },
            )
            if tool_name not in entities:
                entities[tool_name] = ExtractedEntity(
                    id=tool_name,
                    type=NodeType.TOOL,
                    props={"name": tool_name},
                )
            if caller:
                relations.append(
                    ExtractedRelation(
                        source_id=caller,
                        target_id=call_id,
                        type=RelationType.INVOKED,
                        properties={"seq": seq, "episode_id": episode_id},
                    )
                )
            relations.append(
                ExtractedRelation(
                    source_id=call_id,
                    target_id=tool_name,
                    type=RelationType.USED,
                    properties={"episode_id": episode_id},
                )
            )

    result.entities = list(entities.values())
    result.relations = relations
    return optimize_execution_path(result)


def optimize_execution_path(result: ExtractionResult) -> ExtractionResult:
    """
    v2 meaning of 'optimize':
      - drop duplicate identical relations,
      - prune orphan entities (anything except the Task that no relation touches).
    Call-level dedup already happened during parsing via repeat_count.
    """
    seen_rel: set[tuple] = set()
    relations: list[ExtractedRelation] = []
    for rel in result.relations:
        key = (rel.source_id, rel.target_id, rel.type, json.dumps(rel.properties, sort_keys=True, default=str))
        if key in seen_rel:
            continue
        seen_rel.add(key)
        relations.append(rel)

    referenced = {r.source_id for r in relations} | {r.target_id for r in relations}
    entities = [
        e for e in result.entities if e.type == NodeType.TASK or e.id in referenced
    ]
    logger.info(
        "Optimized episode %s: %d->%d relations, %d->%d entities",
        result.episode_id, len(result.relations), len(relations),
        len(result.entities), len(entities),
    )
    result.relations = relations
    result.entities = entities
    return result


def parse_and_optimize_agent_logs(raw_logs_path: str) -> ExtractionResult | None:
    result = parse_agent_logs_to_graph(raw_logs_path)
    if not result.succeeded or not result.entities:
        return None  # failed episode -> caller must skip ingestion entirely
    return result


# ==========================================================
# 4. SCHEMA, INDEXES, MIGRATION
# ==========================================================
_CONSTRAINTS = [
    "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (n:Task) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (n:Agent) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT tool_id IF NOT EXISTS FOR (n:Tool) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT call_id IF NOT EXISTS FOR (n:ToolCall) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (n:Episode) REQUIRE n.id IS UNIQUE",
]
_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX task_text IF NOT EXISTS "
    "FOR (n:Task) ON EACH [n.user_query, n.description]"
)


def ensure_schema(
    driver: Driver,
    qdrant_client: QdrantClient,
    vector_size: int = 384,
    collection_name: str = "agent_task_memory",
) -> None:
    """Idempotent bootstrap: constraints (also give you index-backed MATCH),
    a fulltext index for lexical fallback, and the Qdrant collection."""
    with driver.session() as session:
        for stmt in _CONSTRAINTS:
            session.run(stmt) # type: ignore
        session.run(_FULLTEXT_INDEX)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
    logger.info("Schema ensured (constraints, fulltext index, collection).")


def migrate_legacy_graph(driver: Driver) -> dict[str, int]:
    """
    ONE-TIME cleanup of v1 damage: Result hubs and merged TOOL_USED edges.
    Honest note: v1 never stored per-call detail, so that information is
    unrecoverable from the graph — re-ingest raw logs if you still have them.
    """
    with driver.session() as session:
        edges = session.run("MATCH ()-[r:TOOL_USED]->() DELETE r RETURN count(r) AS n").single()["n"] # type: ignore
        results = session.run("MATCH (r:Result) DETACH DELETE r RETURN count(r) AS n").single()["n"] # type: ignore
        leftover = session.run("MATCH ()-[r:RESULTED_IN]->() DELETE r RETURN count(r) AS n").single()["n"] # type: ignore
    report = {"tool_used_edges": edges, "result_nodes": results, "resulted_in_edges": leftover}
    logger.info("Legacy migration: %s", report)
    return report


# ==========================================================
# 5. LEAN, ATOMIC, IDEMPOTENT INGESTION
# ==========================================================
_NODE_WRITE_CYPHER = {
    nt: (
        f"UNWIND $rows AS row "
        f"MERGE (n:{nt.value} {{id: row.id}}) "
        f"SET n += row.props, n.updated_at = timestamp()"
    )
    for nt in NodeType
}

_REL_LABELS = {
    RelationType.AGENT_USED: ("Task", "Agent"),
    RelationType.INVOKED: ("Agent", "ToolCall"),
    RelationType.USED: ("ToolCall", "Tool"),
}
_RELATION_WRITE_CYPHER = {
    rt: (
        f"UNWIND $rows AS row "
        f"MATCH (a:{src} {{id: row.src}}) "
        f"MATCH (b:{dst} {{id: row.dst}}) "
        f"MERGE (a)-[r:{rt.value}]->(b) "
        f"SET r += row.props"
    )
    for rt, (src, dst) in _REL_LABELS.items()
}

_TOOL_STATS_CYPHER = """
UNWIND $tool_ids AS tid
MATCH (t:Tool {id: tid})
OPTIONAL MATCH (tc:ToolCall)-[:USED]->(t)
WITH t, collect(tc) AS calls
SET t.call_count = size(calls),
    t.success_count = size([c IN calls WHERE c.status = 'success']),
    t.last_used_at = timestamp()
"""

_DELETE_EPISODE_CYPHER = """
MATCH (t:Task {episode_id: $ep}) DETACH DELETE t
WITH 1 AS _
MATCH (tc:ToolCall {episode_id: $ep}) DETACH DELETE tc
WITH 1 AS _
MATCH (e:Episode {id: $ep}) DELETE e
"""


def _write_episode_tx(tx: ManagedTransaction, result: ExtractionResult, source: str) -> None:
    # Ledger first: this transaction only runs for fresh (or force-deleted) episodes.
    tx.run(
        "MERGE (e:Episode {id: $id}) "
        "ON CREATE SET e.ingested_at = timestamp(), e.source = $source",
        id=result.episode_id,
        source=source,
    )
    by_type: dict[NodeType, list[dict]] = {}
    for e in result.entities:
        by_type.setdefault(e.type, []).append({"id": e.id, "props": sanitize_props(e.props)})
    for nt, rows in by_type.items():
        tx.run(_NODE_WRITE_CYPHER[nt], rows=rows) # type: ignore

    by_rel: dict[RelationType, list[dict]] = {}
    for r in result.relations:
        by_rel.setdefault(r.type, []).append(
            {"src": r.source_id, "dst": r.target_id, "props": sanitize_props(r.properties)}
        )
    for rt, rows in by_rel.items():
        tx.run(_RELATION_WRITE_CYPHER[rt], rows=rows) # type: ignore

    tool_ids = sorted({e.id for e in result.entities if e.type == NodeType.TOOL})
    if tool_ids:
        tx.run(_TOOL_STATS_CYPHER, tool_ids=tool_ids)


def ingest_agent_memory(
    driver: Driver,
    qdrant_client: QdrantClient,
    embedder: TextEmbedding,
    result: ExtractionResult | None,
    collection_name: str = "agent_task_memory",
    source: str = "unknown",
    force: bool = False,
) -> bool:
    """
    Returns True if something was written. Skips failed/empty episodes and
    already-ingested episodes (unless force=True, which deletes then rewrites).
    """
    if result is None or not result.succeeded or not result.entities:
        logger.info("Nothing to ingest (failed or empty episode).")
        return False

    with driver.session() as session:
        exists = session.run(
            "MATCH (e:Episode {id: $id}) RETURN count(e) > 0 AS x", id=result.episode_id
        ).single()["x"] # type: ignore
        if exists and not force:
            logger.info("Episode %s already ingested -> skipping (idempotent).", result.episode_id)
            return False
        if exists and force:
            session.execute_write(lambda tx: tx.run(_DELETE_EPISODE_CYPHER, ep=result.episode_id))
        session.execute_write(_write_episode_tx, result, source)

    # ---- Vector side: deterministic point id => true upsert ----
    task = next(e for e in result.entities if e.type == NodeType.TASK)
    user_query = str(task.props.get("user_query") or task.props.get("description") or "")
    tools: list[str] = []
    for e in result.entities:
        if e.type == NodeType.CALL and e.props.get("tool") not in tools:
            tools.append(e.props["tool"])
    agents = sorted({r.target_id for r in result.relations if r.type == RelationType.AGENT_USED})

    # Query-shaped embedding text: what a FUTURE request will look like.
    embed_text = f"{user_query}\nTools: {', '.join(tools)}"
    embedding = list(embedder.embed(embed_text))[0]
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant:{collection_name}:{result.episode_id}"))

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "task_id": task.id,
                    "episode_id": result.episode_id,
                    "user_query": user_query,
                    "summary": embed_text,
                    "outcome": task.props.get("final_outcome"),
                    "agents": agents,
                    "tools": tools,
                    "n_steps": sum(1 for e in result.entities if e.type == NodeType.CALL),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    logger.info("Ingested episode %s (%d entities).", result.episode_id, len(result.entities))
    return True


# ==========================================================
# 6. MULTI-STRATEGY RECALL
# ==========================================================
def search_episodes_vector(
    qdrant_client: QdrantClient,
    embedder: TextEmbedding,
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.40,
    collection_name: str = "agent_task_memory",
    required_tools: Optional[list[str]] = None,
) -> list[dict]:
    """Strategy 1 — semantic. Thresholded cosine + optional payload filter
    (e.g. only episodes that used tools you plan to use)."""
    emb = list(embedder.embed(query))[0]
    flt = None
    if required_tools:
        flt = models.Filter(
            must=[models.FieldCondition(key="tools", match=models.MatchAny(any=required_tools))]
        )
    res = qdrant_client.query_points(
        collection_name=collection_name,
        query=emb.tolist(),
        limit=top_k,
        query_filter=flt,
        with_payload=True,
    )
    hits = []
    for p in res.points:
        if p.score is not None and p.score < score_threshold:
            continue  # refuse to inject weak matches — silence beats noise
        payload = p.payload or {}
        hits.append(
            {
                "task_id": payload.get("task_id"),
                "episode_id": payload.get("episode_id"),
                "score": round(float(p.score), 4),
                "source": "vector",
                "payload": payload,
            }
        )
    return hits


def search_episodes_lexical(driver: Driver, query: str, top_k: int = 5) -> list[dict]:
    """Strategy 2 — lexical (BM25-ish) fallback for exact names/IDs that
    embeddings blur (tool names, product codes, people)."""
    q = _to_lucene(query)
    if not q:
        return []
    cypher = (
        'CALL db.index.fulltext.queryNodes("task_text", $query, {limit: $limit}) '
        "YIELD node, score "
        "RETURN node.id AS task_id, node.episode_id AS episode_id, score"
    )
    with driver.session() as session:
        return [
            {"task_id": r["task_id"], "episode_id": r["episode_id"],
             "score": round(float(r["score"]), 4), "source": "lexical", "payload": {}}
            for r in session.run(cypher, query=q, limit=top_k) # type: ignore
        ]


_BLUEPRINT_CYPHER = """
UNWIND $tasks AS row
MATCH (t:Task {id: row.task_id})
OPTIONAL MATCH (t)-[r:AGENT_USED]->(a:Agent)
OPTIONAL MATCH (a)-[inv:INVOKED]->(tc:ToolCall)-[:USED]->(tool:Tool)
WHERE tc.episode_id = t.episode_id
RETURN row.score AS score,
       row.source AS source,
       t.id AS task_id,
       t.episode_id AS episode_id,
       t.user_query AS user_query,
       t.final_outcome AS outcome,
       a.id AS agent,
       r.role AS role,
       r.reasoning AS reasoning,
       tc.seq AS step,
       tc.tool AS tool,
       tc.agent AS call_agent,
       tc.args AS args,
       tc.status AS status,
       tc.output_preview AS output,
       tc.repeat_count AS repeats
ORDER BY task_id, step
"""


def fetch_episode_blueprints(driver: Driver, hits: list[dict]) -> list[dict]:
    """Strategy 3 — exact graph recall. Because ToolCall is an event node,
    this traversal CANNOT leak calls/results from other episodes (the v1 bug)."""
    if not hits:
        return []
    rows = [{"task_id": h["task_id"], "score": h.get("score"), "source": h.get("source")} for h in hits]
    with driver.session() as session:
        records = [rec.data() for rec in session.run(_BLUEPRINT_CYPHER, tasks=rows)]

    bps: dict[str, dict] = {}
    order: list[str] = []
    for rec in records:
        tid = rec.get("task_id")
        if not tid:
            continue
        if tid not in bps:
            bps[tid] = {
                "task_id": tid,
                "episode_id": rec.get("episode_id"),
                "score": rec.get("score"),
                "source": rec.get("source"),
                "user_query": rec.get("user_query"),
                "outcome": rec.get("outcome"),
                "agents": {},
                "steps": [],
            }
            order.append(tid)
        bp = bps[tid]
        agent = rec.get("agent")
        if agent and agent not in bp["agents"]:
            bp["agents"][agent] = {"role": rec.get("role"), "reasoning": rec.get("reasoning")}
        if rec.get("step") is not None:
            bp["steps"].append(
                {
                    "step": rec["step"],
                    "agent": rec.get("call_agent"),
                    "tool": rec.get("tool"),
                    "args": rec.get("args"),
                    "status": rec.get("status"),
                    "output": rec.get("output"),
                    "repeats": rec.get("repeats"),
                }
            )
    return [bps[t] for t in order]


_TOOL_STATS_CYPHER_READ = """
MATCH (t:Tool)
WHERE size($tool_ids) = 0 OR t.id IN $tool_ids
RETURN t.id AS tool,
       coalesce(t.call_count, 0) AS calls,
       coalesce(t.success_count, 0) AS successes,
       CASE WHEN coalesce(t.call_count, 0) = 0 THEN 0.0
            ELSE 1.0 * t.success_count / t.call_count END AS success_rate
ORDER BY success_rate DESC, calls DESC
"""


def tool_reliability(driver: Driver, tool_ids: Optional[list[str]] = None) -> list[dict]:
    """Strategy 4 — aggregate statistics over ALL remembered episodes.
    Answers 'which tools historically actually work?' Pass [] for all tools."""
    with driver.session() as session:
        return [r.data() for r in session.run(_TOOL_STATS_CYPHER_READ, tool_ids=tool_ids or [])]


def recall_structured(
    driver: Driver,
    qdrant_client: QdrantClient,
    embedder: TextEmbedding,
    new_task_description: str,
    top_k: int = 3,
    score_threshold: float = 0.40,
    collection_name: str = "agent_task_memory",
    required_tools: Optional[list[str]] = None,
) -> dict:
    """Hybrid recall: vector first, lexical backfill, then graph expansion
    and tool stats. Returns raw parts for programmatic use."""
    hits = search_episodes_vector(
        qdrant_client, embedder, new_task_description,
        top_k=top_k, score_threshold=score_threshold,
        collection_name=collection_name, required_tools=required_tools,
    )
    if len(hits) < top_k:  # hybrid backfill for exact-term matches
        have = {h["task_id"] for h in hits}
        for h in search_episodes_lexical(driver, new_task_description, top_k=top_k):
            if h["task_id"] not in have:
                hits.append(h)
                have.add(h["task_id"])
            if len(hits) >= top_k:
                break
    blueprints = fetch_episode_blueprints(driver, hits)
    tools = sorted({s["tool"] for bp in blueprints for s in bp["steps"] if s.get("tool")})
    stats = tool_reliability(driver, tools) if tools else []
    return {"hits": hits, "blueprints": blueprints, "tool_stats": stats}


def render_blueprints(blueprints: list[dict], tool_stats: Optional[list[dict]] = None, max_steps: int = 12) -> str:
    blocks = []
    for bp in blueprints:
        lines = [
            f' PAST TASK (similarity={bp.get("score")}, via {bp.get("source")}): "{bp.get("user_query")}"',
            f'🏁 OUTCOME: {bp.get("outcome")}   (episode {bp.get("episode_id")})',
        ]
        reasoning = next((a["reasoning"] for a in bp["agents"].values() if a.get("reasoning")), None)
        if reasoning:
            lines.append(f'💡 WHY THIS PLAN: {reasoning}')
        if bp["agents"]:
            lines.append("🤖 Crew: " + ", ".join(f"{n} ({a.get('role') or 'role n/a'})" for n, a in bp["agents"].items()))
        lines.append("🪜 Execution steps:")
        if not bp["steps"]:
            lines.append("    (no tool calls recorded)")
        for s in bp["steps"][:max_steps]:
            rep = f" ×{s['repeats']}" if (s.get("repeats") or 0) > 1 else ""
            icon = "✅" if s.get("status") == "success" else "❌"
            lines.append(f"    {s['step']:>2}. {icon} {s.get('agent') or '?'} → {s.get('tool')}{rep}  args={_truncate(s.get('args'), 160)}")
            if s.get("output"):
                lines.append(f"        └─ out: {s['output']}")
        if len(bp["steps"]) > max_steps:
            lines.append(f"    … +{len(bp['steps']) - max_steps} more steps")
        blocks.append("\n".join(lines))

    if tool_stats:
        lines = ["🧰 TOOL RELIABILITY (all remembered episodes):"]
        for ts in tool_stats:
            note = "   (small sample — trust lightly)" if ts["calls"] < 5 else ""
            lines.append(f"    - {ts['tool']}: {ts['successes']}/{ts['calls']} success ({ts['success_rate']:.0%}){note}")
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks)


def recall_similar_executions(
    driver: Driver,
    qdrant_client: QdrantClient,
    embedder: TextEmbedding,
    new_task_description: str,
    top_k: int = 3,
    score_threshold: float = 0.40,
    collection_name: str = "agent_task_memory",
    required_tools: Optional[list[str]] = None,
) -> str:
    """Drop-in replacement for v1's recall, now hybrid + thresholded + ordered."""
    parts = recall_structured(
        driver, qdrant_client, embedder, new_task_description,
        top_k=top_k, score_threshold=score_threshold,
        collection_name=collection_name, required_tools=required_tools,
    )
    if not parts["blueprints"]:
        return (
            f"No reliable past executions found (score threshold={score_threshold}). "
            "Plan from scratch; do not imitate unknown history."
        )
    return render_blueprints(parts["blueprints"], parts["tool_stats"])


# ==========================================================
# 7. USAGE EXAMPLE
# ==========================================================
if __name__ == "__main__":
    load_dotenv()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "neo4j://localhost"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "your_password_here")),
    )
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    ensure_schema(driver, qdrant_client, vector_size=384)
    # migrate_legacy_graph(driver)   # run ONCE if upgrading from v1

    result = parse_and_optimize_agent_logs("agent_logs.jsonl")
    ingest_agent_memory(driver, qdrant_client, embedder, result, source="agent_logs.jsonl")

    print(recall_similar_executions(driver, qdrant_client, embedder, "maya", top_k=3))
    print(tool_reliability(driver))
