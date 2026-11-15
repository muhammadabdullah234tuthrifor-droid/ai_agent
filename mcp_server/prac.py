from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from google import genai


from data_prap import UltimateDataPrep
from gdb import (
    detect_dupies,
    extract_graph,
    find_graph_entry_points,
    get_relationship_overlap,
    graph_search,
    ingest_extraction_with_reporting,
    merge_entities_in_graph,
    verify_grounding,
)
from logger import get_logger

# from vec import add_chunks_to_chroma, vector_search
from vec1 import add_chunks, vector_search

logger = get_logger("prac")

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# processer = UltimateDataPrep("../../Downloads/text.txt", tokenizer)
# processer.yolo()
# docs = loader.load()
# =======================================================

# model = SentenceTransformer("all-MiniLM-L6-v2")
# reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")

# client = chromadb.PersistentClient(path="./chroma_db")

# collection = client.get_or_create_collection(
#     name="knowledge_base", metadata={"hnsw:space": "cosine"}
# )
# ==========================================================
# DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# SPARSE_MODEL = "Qdrant/bm25"
# RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# docs = loader.load()
# dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
# sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)
# reranker = TextCrossEncoder(model_name=RERANKER_MODEL)


# add_chunks_to_chroma(collection, model, docs)
#     add_chunks(
#     dense_embedder=dense_embedder, sparse_embedder=sparse_embedder, docs=docs
# )

# result = vector_search(
#     query="who works at Ather",
#     model=model,
#     reranker=reranker,
#     collection=collection,
#     i=20,
#     j=10,
#     k=5,
#     source_filter=None,
# )
# print(result)

#
# ============================================================
#


# URI = "neo4j://localhost"
# AUTH = ("neo4j", "your_password_here")

# query = "who works at ather?"

# driver = GraphDatabase.driver(URI, auth=AUTH)
# DENSE_MODEL = "BAAI/bge-small-en-v1.5"

# dense_embedder = TextEmbedding(model_name=DENSE_MODEL)

# entity_rel = extract_graph("../../Downloads/text.txt")
# verified_entity_rel = verify_grounding(entity_rel, "../../Downloads/text.txt", 0.85)

# dupies = detect_dupies(
#     verified_entity_rel.entities, verified_entity_rel.relations, 85, 0.85
# )
# dupies = get_relationship_overlap(driver, dupies, 0.85)

# stats = ingest_extraction_with_reporting(
#    dense_embedder= dense_embedder, driver= driver, entity_collection_name= "entity_collection",
#   result= verified_entity_rel, doc_id= "doc_1", batch_size= 200
# )
# print("======")
# print(stats)

# if dupies:
#     for dupy in dupies:
#         if len(dupy[0].name) >= len(dupy[1].name):
#             merge_entities_in_graph(
#                 driver,
#                 canonical_name=dupy[0].name,
#                 alias_name=dupy[1].name,
#                 node_type=dupy[0].type.value,
#             )
#         merge_entities_in_graph(
#             driver,
#             canonical_name=dupy[0].name,
#             alias_name=dupy[1].name,
#             node_type=dupy[0].type.value,
#         )

# results, _ = graph_search(dense_embedder= dense_embedder, driver= driver, entity_collection_name="entity_collection", entity_mentions= [""], max_hops= 1)

#
# ====================================================================
#
genai_client = genai.Client()


class RoutingDecision(BaseModel):
    pattern: Literal["vector_only", "graph_only", "fused"] = Field(
        description="Which retrieval pattern this query needs"
    )
    entity_mentions: list[str] = Field(
        default_factory=list,
        description="Named entities (people, companies, products) the query specifically refers to. Empty if the query is purely topical.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this pattern was chosen"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident are we in this routing decision (0.0-1.0). Below 0.5 suggests ambiguity worth noting.",
    )


ROUTER_SYSTEM_PROMPT = """
    You decide how to retrieve context for a user question, given two available retrieval systems:

    VECTOR SEARCH: Fast semantic matching over document text. Good for:
    - Purely topical/conceptual questions ("What are common AI safety approaches?")
    - Finding prose content about abstract concepts
    - When no specific named entity is central to the question

    GRAPH TRAVERSAL: Structured navigation of entities and relationships. Good for:
    - Questions about specific people, companies, products by name
    - Relational questions ("Who works where?", "What competes with what?")
    - Multi-hop questions requiring fact chains ("Companies X worked at, and their competitors")

    DECISION RULES:

    Choose "vector_only" if:
    - The query is purely topical with no specific named entity central to it
    - Example: "What are risks associated with foundation models?"

    Choose "graph_only" if:
    - The query names specific entities and asks only about their relationships/connections
    - The answer is a fact you can traverse from named entities
    - Example: "What companies has Elena Voss worked at?" (graph knows this directly)

    Choose "fused" if:
    - The query names specific entities AND asks about topical/semantic content related to them
    - You need relational structure (graph) to identify which entities, then topical content (vector) to contextualize
    - Example: "What are risks at companies Elena Voss has worked at?" (need graph for companies, vector for risks)

    Extraction: If the query mentions specific people, companies, or products by name, extract them as entity_mentions.
    If you're unsure what the query is asking, set confidence lower (but still make your best guess for pattern).
"""
ROUTER_TOOL = {
    "name": "route_query",
    "description": "Decide which retrieval strategy a user query needs.",
    "input_schema": RoutingDecision.model_json_schema(),
}


def route_query(query: str) -> RoutingDecision:

    response = genai_client.interactions.create(
        model="gemini-3.6-flash",
        system_instruction=ROUTER_SYSTEM_PROMPT,
        input=query,
        tools=[ROUTER_TOOL],
        generation_config={
            "max_output_tokens": 1024,
            "tool_choice": {"allowed_tools": {"mode": "any", "tools": ["route_query"]}},
        },
    )
    # response = RoutingDecision(
    #     pattern="fused",
    #     entity_mentions=["Elena Vance", "Aether Dynamics", "Apex Robotics"],
    #     reasoning="...",
    #     confidence=1.00,
    # )
    response = response.steps[-1].arguments # type: ignore

    response = RoutingDecision.model_validate(response)
    logger.debug("decision of router: %s", response)
    return response


def execute_fused(
    query: str,
    client,
    dense_embedder,
    sparse_embedder,
    driver,
    entity_collection_name,
    collection_name,
    entity_mentions: list[str],
    i: int = 20,
    j: int = 10,
    k: int = 5,
    max_hops: int = 1,
):

    vector_result = vector_search(
        client=client,
        query=query,
        collection_name=collection_name,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )
    logger.debug("vector result: %s", vector_result)

    graph_result, _ = graph_search(
        dense_embedder=dense_embedder,
        driver=driver,
        entity_collection_name=entity_collection_name,
        entity_mentions=entity_mentions,
        max_hops=1,
    )
    logger.debug("graph results: %s", graph_result)

    return str(graph_result) + str(vector_result)


def hybrid_retrieve(
    query: str,
    client,
    dense_embedder,
    sparse_embedder,
    driver,
    collection_name,
    entity_collection_name,
    filter: str | None = None,
):
    logger.debug("q: %s", query)
    routing_decision = route_query(query)

    if routing_decision.pattern == "vector_only":
        # result = vector_search(
        #     query=query,
        #     model=model,
        #     reranker=reranker,
        #     collection=collection,
        #     source_filter= filter
        # )
        result = vector_search(
            client=client,
            query=query,
            collection_name=collection_name,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        logger.debug("vector result: %s", result)
        return result

    elif routing_decision.pattern == "graph_only":
        result, _ = graph_search(
            dense_embedder=dense_embedder,
            driver=driver,
            entity_collection_name="entity_collection",
            entity_mentions=routing_decision.entity_mentions,
            max_hops=1,
        )
        logger.debug("graph results: %s", result)
        return result
    else:
        result = execute_fused(
            query=query,
            client=client,
            collection_name=collection_name,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            driver=driver,
            entity_collection_name=entity_collection_name,
            entity_mentions=routing_decision.entity_mentions,
        )
        logger.debug("fused results: %s", result)
        return result


if __name__ == "__main__":
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "your_password_here")

    DENSE_MODEL = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL = "Qdrant/bm25"
    RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)
    reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
    driver = GraphDatabase.driver(URI, auth=AUTH)
    client = QdrantClient(url="http://localhost:6333")
    print(
        hybrid_retrieve(
            query="who works at ather",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            driver=driver,
            client=client,
            collection_name="rag_collection",
            entity_collection_name="entity_collection",
        )
    )


# export BG_JOB_TIMEOUT_SECS=8800          # 30 minutes (default is often much lower)
# export BG_JOB_ISOLATED_LOOPS=true
