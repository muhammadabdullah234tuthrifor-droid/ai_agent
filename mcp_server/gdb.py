import uuid
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from itertools import combinations

from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai
from neo4j import Driver, GraphDatabase, ManagedTransaction
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from rapidfuzz import fuzz

from logger import get_logger

logger = get_logger(name="gdb")
# client = genai.Client()

def index_entity_for_retrieval(
    dense_embedder,
    entity_name: str,
    node_type: str,
    entity_collection_name: str = "entity_collection",
):
    client = QdrantClient(url="http://localhost:6333")

    embedding = list(dense_embedder.embed(entity_name))
    vector_size = len(embedding[0]) if embedding else 384
    print("present")
    if not client.collection_exists(entity_collection_name):
        print("creating...")
        client.create_collection(
            collection_name=entity_collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    client.upsert(
        collection_name=entity_collection_name,
        points=[
            models.PointStruct(
                id=uuid.uuid4(),
                vector=embedding[0],
                payload={"text": entity_name, "node_type": node_type},
            )
        ],
    )


class NodeType(str, Enum):
    PERSON = "Person"
    COMPANY = "Company"
    PRODUCT = "Product"
    FUNDING_ROUND = "FundingRound"


class RelationType(str, Enum):
    FOUNDED = "FOUNDED"
    WORKS_AT = "WORKS_AT"
    ACQUIRED = "ACQUIRED"
    INVESTED_IN = "INVESTED_IN"
    PRODUCES = "PRODUCES"
    COMPETES_WITH = "COMPETES_WITH"
    RAISED = "RAISED"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    RELATED_TO = "RELATED_TO"  # fallback


class ExtractedEntity(BaseModel):
    id: str = Field(
        description="A short, stable local identifier for this entity within this extraction, e.g. 'satya_nadella'. Used to link relationships to entities."
    )
    name: str = Field(description="The canonical name as it appears in the text")
    type: NodeType
    source_span: str = Field(
        description="The exact sentence or phrase from the source text that mentions this entity"
    )


class ExtractedRelation(BaseModel):
    source_id: str = Field(description="id of the source entity")
    target_id: str = Field(description="id of the target entity")
    type: RelationType
    properties: dict[str, str] = Field(
        default_factory=dict,
        description="Additional attributes like year, amount, role",
    )
    source_span: str = Field(
        description="The exact sentence from the source text that supports this relationship"
    )
    description: str = Field(
        default="",
        description="Only used when type is RELATED_TO — free-text description of the actual relationship",
    )


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


load_dotenv()

client = genai.Client()

SYSTEM_PROMPT = """You extract structured knowledge graph data from text.

CRITICAL RULES:
1. Only extract entities and relationships that are EXPLICITLY stated in the provided text.
   Do not use outside knowledge, even if you are confident it is true.
2. If a fact is not directly supported by a specific sentence in the text, do not include it.
3. Every entity and relation must include the exact source_span (sentence or phrase)
   that justifies it.
4. Use the RELATED_TO type with a description only when no other relation type fits —
   do not invent new relation types.
5. If the text mentions an entity but no clear relationship to another extracted entity,
   still include the entity on its own.
"""

EXTRACTION_TOOL = {
    "type": "function",
    "name": "extract_graph",
    "description": "Extract entities and relationships from the text into the fixed ontology.",
    "parameters": ExtractionResult.model_json_schema(),
}


def extract_graph(data: str):  # 1


    response = client.interactions.create(
        model="gemini-3.7-flash",
        system_instruction=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        input=f"Extract the knowledge graph from this text: {data}.",
        generation_config={
            "max_output_tokens": 2048,
            "tool_choice": {
                "allowed_tools": {"mode": "any", "tools": ["extract_graph"]}
            },
        },
    )
    response = response.steps[-1].arguments # type: ignore
    logger.debug("Extracted nodes and relations: %s", response)
    response = ExtractionResult.model_validate(response)
    return response


txt = """
Years after she founded the ambitious software enterprise **Aether Dynamics**, **Elena Vance** stood in the glass-walled sanctuary of her top-floor office, contemplating the sprawling web of connections that had built her empire. What began as a mere whisper of an idea in a cramped workshop had rapidly matured into an industry powerhouse that produces NeuralCore, a revolutionary artificial intelligence chip that reshaped real-time computing standards across the globe. The journey, however, was defined by relentless pressure; from its earliest days, Aether fiercely competed with Apex Robotics, a well-entrenched rival that constantly threatened to steal their technological momentum. To secure their survival and scale operations, Aether successfully raised a pivotal Series A funding round, drawing elite institutional investors who eagerly participated in the deal and invested in the company's bold hardware roadmap. This massive influx of capital enabled brilliant chief architect Marcus Thorne, who works at Aether, to solve complex system bottlenecks and accelerate product manufacturing. With their position fortified, Aether pulled off a masterstroke the following year when it strategically acquired Apex Robotics, effectively turning their most aggressive competitor into a vital internal division. Looking back over the ledger of achievements, strategic pivots, and technological breakthroughs, Elena realized that every triumph was inextricably related to a singular, enduring mission: bridging the gap between human curiosity and machine intelligence.
"""


def verify_grounding(
    result: ExtractionResult, path, threshold: float = 0.85
) -> ExtractionResult:
    with open(path, "r") as f:
        source_text = f.read()

    def is_grounded(span: str) -> bool:
        if span in source_text:
            return True
        best_ratio = max(
            SequenceMatcher(None, span, source_text[i : i + len(span) + 10]).ratio()
            for i in range(0, max(len(source_text) - len(span), 1), 20)
        )
        return best_ratio >= threshold

    grounded_entities = [e for e in result.entities if is_grounded(e.source_span)]
    valid_ids = {e.id for e in grounded_entities}
    grounded_relations = [
        r
        for r in result.relations
        if is_grounded(r.source_span)
        and r.source_id in valid_ids
        and r.target_id in valid_ids
    ]
    result = ExtractionResult(entities=grounded_entities, relations=grounded_relations)
    logger.debug(
        "Verified extracted facts by checking if ids of src and target ids in relations match with entities and if src span exist: %s",
        result,
    )
    return result


def block_candidates(
    entities: list[ExtractedEntity], threshold: int = 85
) -> list[tuple[ExtractedEntity, ExtractedEntity]]:
    candidates = []
    for a, b in combinations(entities, 2):
        if a.type != b.type:
            continue
        score = fuzz.token_sort_ratio(a.name.lower(), b.name.lower())
        if score >= threshold:
            candidates.append((a, b))
    logger.debug(
        "Nodes that have the same type and almost the same names and therfore suspected to be the same node: %s",
        candidates,
    )
    return candidates


def relationship_overlap_score(
    entities: list[tuple[ExtractedEntity, ExtractedEntity]],
    relations: list[ExtractedRelation],
    threshold: float = 0.85,
) -> list[tuple[ExtractedEntity, ExtractedEntity]]:
    final = []

    neighbors = defaultdict(set)
    for r in relations:
        if r.source_id != r.target_id:
            neighbors[r.source_id].add(r.target_id)
            neighbors[r.target_id].add(r.source_id)

    for a, b in entities:
        candidates = {a.id, b.id}

        a_neighbors = neighbors.get(a.id, set()) - candidates
        logger.debug("neighbours of one node: %s", a_neighbors)

        b_neighbors = neighbors.get(b.id, set()) - candidates
        logger.debug("neighbours of another node: %s", b_neighbors)

        if not a_neighbors or not b_neighbors:
            continue

        overlap = len(a_neighbors & b_neighbors)
        if overlap / min(len(a_neighbors), len(b_neighbors)) >= threshold:
            final.append((a, b))

    logger.debug(
        "Previous Candidate nodes that have almost the same neighbors: %s", final
    )

    return final


def detect_dupies(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    threshold_name: int = 85,
    threshold_nebor: float = 0.85,
) -> list[tuple[ExtractedEntity, ExtractedEntity]]:
    candidates = block_candidates(entities, threshold_name)
    candidates = relationship_overlap_score(candidates, relations, threshold_nebor)
    return candidates


def build_id_to_name_map(entities: list[ExtractedEntity]) -> dict[str, str]:
    return {entity.id: entity.name for entity in entities}


@dataclass
class GraphWriteOp:
    cypher: str
    params: dict


def entity_to_write_op(entity: ExtractedEntity, doc_id: str) -> GraphWriteOp:
    return GraphWriteOp(
        cypher=f"""
        MERGE (n:{entity.type.value} {{name: $name}})
        ON CREATE SET n.created_at = timestamp(), n.source_docs = [$doc_id]
        ON MATCH SET n.source_docs = CASE
            WHEN NOT $doc_id IN n.source_docs
            THEN n.source_docs + $doc_id
            ELSE n.source_docs
        END
        SET n.last_span = $source_span
        """,
        params={
            "name": entity.name,
            "doc_id": doc_id,
            "source_span": entity.source_span,
        },
    )


def relation_to_write_op(
    relation: ExtractedRelation, id_to_name: dict[str, str], doc_id: str
) -> GraphWriteOp:
    rel_type = relation.type.value
    return GraphWriteOp(
        cypher=f"""
        MATCH (a {{name: $source_name}}), (b {{name: $target_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        ON CREATE SET r += $properties, r.source_docs = [$doc_id], r.created_at = timestamp()
        ON MATCH SET r.source_docs = CASE
            WHEN NOT $doc_id IN r.source_docs
            THEN r.source_docs + $doc_id
            ELSE r.source_docs
        END
        SET r.last_span = $source_span
        """,
        params={
            "source_name": id_to_name[relation.source_id],
            "target_name": id_to_name[relation.target_id],
            "properties": relation.properties,
            "doc_id": doc_id,
            "source_span": relation.source_span,
        },
    )


def write_batch(tx: ManagedTransaction, ops: list[GraphWriteOp]):
    for op in ops:
        tx.run(op.cypher, op.params) # type: ignore


def ingest_extraction_with_reporting(
    dense_embedder,
    driver,
    entity_collection_name,
    result: ExtractionResult,
    doc_id: str,
    batch_size: int = 200,
) -> dict:
    print("cLLWS")
    id_to_name = build_id_to_name_map(result.entities)
    logger.debug("id to name map of entities: %s", id_to_name)
    entity_ids = set(id_to_name.keys())

    valid_relations = [
        r
        for r in result.relations
        if r.source_id in entity_ids and r.target_id in entity_ids
    ]

    entity_ops = [entity_to_write_op(e, doc_id) for e in result.entities]
    logger.debug("Cypher for entities: %s", entity_ops)
    relation_ops = [
        relation_to_write_op(r, id_to_name, doc_id) for r in valid_relations
    ]
    logger.debug("Cypher for relations: %s", relation_ops)

    stats = {
        "entities_attempted": len(entity_ops),
        "relations_attempted": len(relation_ops),
        "errors": [],
    }

    with driver.session() as session:
        try:
            for i in range(0, len(entity_ops), batch_size):
                session.execute_write(write_batch, entity_ops[i : i + batch_size])
        except Exception as e:
            stats["errors"].append(f"entity batch failed: {e}")
            return stats
        for entity in result.entities:
            index_entity_for_retrieval(
                dense_embedder=dense_embedder,
                entity_collection_name=entity_collection_name,
                entity_name=entity.name,
                node_type=entity.type.value,
            )
        try:
            for i in range(0, len(relation_ops), batch_size):
                session.execute_write(write_batch, relation_ops[i : i + batch_size])
        except Exception as e:
            stats["errors"].append(f"relation batch failed: {e}")

    return stats


def run_query(driver: Driver, query: str, params: dict | None = None) -> list[dict]:

    params = params or {}
    with driver.session() as session:
        try:
            result = session.run(query, params) # type: ignore
            return [record.data() for record in result]
        except Exception as e:
            raise RuntimeError(
                f"Query failed: {e}\nQuery: {query}\nParams: {params}"
            ) from e


def get_relationship_overlap(
    driver,
    entities: list[tuple[ExtractedEntity, ExtractedEntity]],
    threshold: float = 0.85,
) -> list[tuple[ExtractedEntity, ExtractedEntity]]:
    final = []
    for a, b in entities:
        name_a = a.name
        name_b = b.name

        result = run_query(
            driver,
            """
            MATCH (a {name: $name_a})-[]-(neighbor_a)
            WITH a, collect(DISTINCT neighbor_a.name) AS a_neighbors
            MATCH (b {name: $name_b})-[]-(neighbor_b)
            WITH a_neighbors, collect(DISTINCT neighbor_b.name) AS b_neighbors
            RETURN a_neighbors, b_neighbors
            """,
            {"name_a": name_a, "name_b": name_b},
        )
        # print(result)
        logger.debug("result from dupies in graph: %s", result)
        if not result:
            continue
        a_neighbors = set(result[0]["a_neighbors"])
        b_neighbors = set(result[0]["b_neighbors"])
        if not a_neighbors or not b_neighbors:
            continue
        overlap = len(a_neighbors & b_neighbors)
        if overlap / min(len(a_neighbors), len(b_neighbors)) >= threshold:
            final.append((a, b))
    return final


def merge_entities_in_graph(
    driver, canonical_name: str, alias_name: str, node_type: str
):
    safe_label = node_type.replace("`", "")

    query = f"""
    MATCH (canonical:`{safe_label}`)
    WHERE canonical.name = $canonical_name

    MATCH (alias:`{safe_label}`)
    WHERE alias.name = $alias_name AND elementId(alias) <> elementId(canonical)

    SET canonical.aliases = coalesce(canonical.aliases, []) + alias.name
    WITH canonical, alias
    CALL apoc.refactor.mergeNodes([canonical, alias], {{properties: "combine", mergeRels: true}})
    YIELD node
    RETURN node
    """

    run_query(
        driver, query, {"canonical_name": canonical_name, "alias_name": alias_name}
    )


@dataclass
class EntryPoint:
    node_name: str
    node_type: str
    confidence: float


def find_graph_entry_points(
    dense_embedder, entity_collection_name, query_text: str, top_k: int = 3
) -> list[EntryPoint]:
    query_embedding = next(iter(dense_embedder.embed(query_text)))


    client = QdrantClient(url="http://localhost:6333")

    result = client.query_points(
        collection_name=entity_collection_name,
        query=query_embedding,
        limit=top_k,
    )

    entry_points = []
    for distance, name, node_type in zip(
        [point.score for point in result.points],
        [point.payload.get("text") for point in result.points], # type: ignore
        [point.payload.get("node_type") for point in result.points], # type: ignore
    ):
        entry_points.append(
            EntryPoint(
                node_name=str(name), node_type=str(node_type), confidence=1 - distance
            )
        )
    logger.debug("entities found in chroma: %s", entry_points)
    return entry_points


def traverse_from_entity(
    driver, entity_name: str, entity_type: str, max_hops: int = 1
) -> list[dict]:

    results = run_query(
        driver,
        f"""
        MATCH path = (start: `{entity_type}` {{name: $name}})-[*1..{max_hops}]->(connected)
        RETURN
            [node in nodes(path) | node.name]           AS node_names,
            [rel  in relationships(path) | type(rel)]   AS rel_types,
            [node in nodes(path) | labels(node)[0]] AS node_types,
            connected.name                              AS endpoint_name,
            labels(connected)[0]                        AS endpoint_type,
            length(path)                                AS hops
        ORDER BY hops ASC
        LIMIT 50
        """,
        {"name": entity_name},
    )
    logger.debug("neigbhors found for the query entity: %s", results)
    return results


def traverse_to_entity(
    driver, entity_name: str, entity_type: str, max_hops: int = 1
) -> list[dict]:

    results = run_query(
        driver,
        f"""
        MATCH path = (start: `{entity_type}` {{name: $name}})<-[*1..{max_hops}]-(connected)
        RETURN
            [node in nodes(path) | node.name]           AS node_names,
            [rel  in relationships(path) | type(rel)]   AS rel_types,
            [node in nodes(path) | labels(node)[0]] AS node_types,
            connected.name                              AS endpoint_name,
            labels(connected)[0]                        AS endpoint_type,
            length(path)                                AS hops
        ORDER BY hops ASC
        LIMIT 50
        """,
        {"name": entity_name},
    )
    logger.debug("neigbhors found for the query entity: %s", results)
    return results


def traverse_relationship(
    driver, entity_name: str, entity_type: str, rel_type: str, max_hops: int = 2
) -> list[dict]:


    results = run_query(
        driver,
        f"""
        MATCH path = (start: `{entity_type}` {{name: $name}})-[:`{rel_type}`*1..{max_hops}]->(connected)
        RETURN
            [node in nodes(path) | node.name]        AS node_names,
            [rel  in relationships(path) | type(rel)] AS rel_types,
            connected.name                            AS endpoint_name,
            labels(connected)[0]                      AS endpoint_type,
            length(path)                              AS hops
        ORDER BY hops ASC
        LIMIT 50
        """,
        {"name": entity_name},
    )
    logger.debug("neigbhors found for the query entity: %s", results)
    return results


#
#
# ================================================
#
#





def graph_search(
    dense_embedder,
    driver: Driver,
    entity_collection_name,
    entity_mentions: list[str],
    max_hops: int = 1,
) -> tuple:
    """
    Relational navigation from named entities.
    Used when query is entirely about entity relationships/connections.
    """
    all_paths_in = []
    all_paths_out = []
    resolved_entry_points = []

    for mention in entity_mentions:
        print(f"entity collections: {entity_mentions}")
        entry_points = find_graph_entry_points(
            dense_embedder=dense_embedder,
            entity_collection_name=entity_collection_name,
            query_text=mention,
            top_k=2,
        )
        if not entry_points:
            continue

        resolved_entry_points.extend(entry_points)
        # Step 2: expand from each entry point
        for entry_point in entry_points:
            paths_out = traverse_from_entity(
                driver, entry_point.node_name, entry_point.node_type, max_hops=max_hops
            )
            paths_in = traverse_to_entity(
                driver, entry_point.node_name, entry_point.node_type, max_hops=max_hops
            )
            # for path in paths:
            #     path["source_entity"] = mention
            #     path["entry_point_confidence"] = entry_point.confidence
            all_paths_out.extend(paths_out)
            all_paths_in.extend(paths_in)

        logger.debug("entry points: %s", resolved_entry_points)
        logger.debug("all paths: %s", all_paths_out)
        logger.debug("all paths: %s", all_paths_in)

    graph_facts = []
    for path in all_paths_out:
        node_names = path["node_names"]
        node_types = path["node_types"]
        rel_types = path["rel_types"]

        fact = " → ".join(
            [
                f"{node_names[i]}({node_types[i]}) {rel_types[i] if i < len(rel_types) else ''}"
                for i in range(len(node_names))
            ]
        )
        graph_facts.append(fact)

    for path in all_paths_in:
        node_names = path["node_names"]
        node_types = path["node_types"]
        rel_types = path["rel_types"]

        fact = " ← ".join(
            [
                f"{node_names[i]}({node_types[i]}) {rel_types[i] if i < len(rel_types) else ''}"
                for i in range(len(node_names))
            ]
        )
        graph_facts.append(fact)

    logger.debug("graph facts: %s", graph_facts)

    return (
        graph_facts,
        resolved_entry_points,
    )


if __name__ == "__main__":
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "your_password_here")

    query = "ather?"

    driver = GraphDatabase.driver(URI, auth=AUTH)
    DENSE_MODEL = "BAAI/bge-small-en-v1.5"

    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)

    entity_rel = extract_graph("../../Downloads/text.txt")
    verified_entity_rel = verify_grounding(entity_rel, "../../Downloads/text.txt", 0.85)

    dupies = detect_dupies(
        verified_entity_rel.entities, verified_entity_rel.relations, 85, 0.85
    )
    dupies = get_relationship_overlap(driver, dupies, 0.85)

    stats = ingest_extraction_with_reporting(
        dense_embedder=dense_embedder,
        driver=driver,
        entity_collection_name="entity_collection",
        result=verified_entity_rel,
        doc_id="doc_1",
        batch_size=200,
    )
    print("======")
    print(stats)

    if dupies:
        for dupy in dupies:
            if len(dupy[0].name) >= len(dupy[1].name):
                merge_entities_in_graph(
                    driver,
                    canonical_name=dupy[0].name,
                    alias_name=dupy[1].name,
                    node_type=dupy[0].type.value,
                )
            merge_entities_in_graph(
                driver,
                canonical_name=dupy[0].name,
                alias_name=dupy[1].name,
                node_type=dupy[0].type.value,
            )

    results, _ = graph_search(
        dense_embedder=dense_embedder,
        driver=driver,
        entity_collection_name="entity_collection",
        entity_mentions=["ragger"],
        max_hops=1,
    )
    print(results)
