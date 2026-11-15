import ast
import json
import uuid
from pathlib import PurePath

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from loader import loader
from logger import get_logger

load_dotenv()

client = QdrantClient(url="http://localhost:6333")

logger = get_logger("vec_1")


def pt_cld_chks(docs):

    cld_chks = filter(
        lambda doc: doc.metadata["orig_elements"].split("_")[-2] == "child",
        docs[0],
    )
    logger.debug("type of doc: %s", type(docs[0]))

    logger.debug("Child Docs: %s", cld_chks)
    logger.debug("Type of hild Docs: %s", type(cld_chks))


    pt_chks = filter(
        lambda doc: doc.metadata["orig_elements"].split("_")[-2] == "parent",
        docs[0],
    )

    logger.debug("Parent Docs: %s", pt_chks)
    logger.debug("Type of parent Docs: %s", type(pt_chks))



    return (pt_chks, cld_chks)


def builder(cld_chks):

    text = []

    ids = []

    metadatas = []

    for record in cld_chks:
        if type(record).__name__ == "str":
            text = ast.literal_eval(record)
        logger.debug("Type of record: %s", type(record))
        logger.debug("Record Docs: %s", record)


        text.append(record.page_content)
        id = record.metadata["orig_elements"]

        ids.append(id)

        pg_nm: str

        if "page_number" in record.metadata:
            pg_nm = record.metadata["page_number"]

        else:
            pg_nm = "unknown"

        metadata = {
            "source": record.metadata["filename"],
            "chunk_index": pg_nm,
        }

        metadatas.append(metadata)

    logger.debug("Texts: %s", text)

    logger.debug("Ids: %s", ids)

    logger.debug("Metadatas: %s", metadatas)

    return (text, ids, metadatas)


def add_chunks(
    client,
    dense_embedder,
    sparse_embedder,
    docs,
    collection_name: str = "rag_collection",
):

    pt_chks, cld_chks = pt_cld_chks(docs)

    text, ids, metadatas = builder(cld_chks)

    pt_chks = [
        {"metadata": pt_chk.metadata, "page_content": pt_chk.page_content}
        for pt_chk in pt_chks
    ]

    with open(".data/vec_data.json", "w") as f:
        json.dump({"ids": ids, "text": text, "pt_chks": pt_chks}, f)

    dense_embedding = list(dense_embedder.embed(text))

    sparse_embedding = sparse_embedder.embed(text)
    # print(sparse_embedding)

    vector_size = len(dense_embedding[0]) if dense_embedding else 384

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_embedding": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse_embedding": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    collection = client.get_collection(collection_name)

    logger.debug(collection)

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=uuid.uuid4(),
                vector={
                    "dense_embedding": dense_vec,
                    "sparse_embedding": models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values
                    ),
                },
                payload={"id": id, "text": doc, "metadata": metadata},
            )
            for (id, doc, dense_vec, sparse_vec, metadata) in zip(
                ids, text, dense_embedding, sparse_embedding, metadatas
            )
        ],
    )


def query_with_filter(
    client,
    collection_name,
    dense_embedder,
    sparse_embedder,
    query: str,
    source_filter: str | None = None,
    j: int = 20,
    k: int = 10,
):

    dense_embedding = next(iter(dense_embedder.embed([query])))
    sparse_embedding = next(iter(sparse_embedder.embed(query)))
    logger.debug(sparse_embedding)

    if not client.collection_exists(collection_name):
        raise ValueError("Invalid age provided: age cannot be negative.")

    where_clause = {"chunk_index": source_filter} if source_filter else None

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_embedding,
                using="dense_embedding",
                limit=j,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_embedding.indices,
                    values=sparse_embedding.values,
                ),
                using="sparse_embedding",
                limit=j,
            ),
        ],
        query=models.RrfQuery(
            rrf=models.Rrf(k=60),
        ),
        limit=k,
    )

    logger.debug("Chroma Results: %s", results)

    docs = [point.payload.get("text") for point in results.points]

    ids = [point.payload.get("id") for point in results.points]

    metadatas = [point.payload["metadata"] for point in results.points]

    distances = [point.score for point in results.points]
    result = {
        "ids": ids,
        "documents": docs,
        "metadatas": metadatas,
        "distances": distances,
    }

    logger.debug(result)
    return result


def hybrid_search(
    query: str,
    client,
    collection_name: str,
    dense_embedder,
    sparse_embedder,
    i: int = 20,
    j: int = 10,
    k: int = 5,
    source_filter: str | None = None,
):

    results = query_with_filter(
        client=client,
        collection_name=collection_name,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        query=query,
        source_filter=source_filter,
        j=i,
        k=j,
    )
    logger.debug(results)

    return results


def reorder_by_position_bias(ranked_chunks: dict) -> dict:

    ids = ranked_chunks["ids"]

    docs = ranked_chunks["documents"]

    metadata = ranked_chunks["metadatas"]

    n = len(docs)

    if n <= 2:
        return ranked_chunks

    reordered_docs = [None] * n

    reordered_ids = [None] * n

    reordered_meta = [None] * n

    left = 0

    right = n - 1

    for i in range(n):
        if i % 2 == 0:
            reordered_docs[left] = docs[i]

            reordered_ids[left] = ids[i]

            reordered_meta[left] = metadata[i]

            left += 1

        else:
            reordered_docs[right] = docs[i]

            reordered_ids[right] = ids[i]

            reordered_meta[right] = metadata[i]

            right -= 1

    ranked_chunks["documents"] = reordered_docs

    ranked_chunks["ids"] = reordered_ids

    ranked_chunks["metadatas"] = reordered_meta

    return ranked_chunks


def yield_parents(cld_cks: dict, pt_cks: list) -> list:

    parent_lookup = {}

    for pk in pt_cks:
        p_id = pk["metadata"]["orig_elements"]

        if p_id:
            parent_lookup[p_id] = pk

    retrieved_parents = []

    seen_ids = set()

    for child_id in cld_cks.get("ids", []):
        logger.debug(cld_cks.get("ids"))
        if "_child_" in child_id:
            parent_id = child_id.rsplit("_child_", 1)[0]

        else:
            parent_id = child_id

        if parent_id in parent_lookup and parent_id not in seen_ids:
            retrieved_parents.append(parent_lookup[parent_id])

            seen_ids.add(parent_id)
    logger.debug("retrieved_parents: %s", retrieved_parents)
    return retrieved_parents


def format_parents_for_llm(parent_docs: list) -> str:

    formatted_pieces = []

    for doc in parent_docs:
        source = doc["metadata"].get("filename", "Unknown")

        page = doc["metadata"].get("page_number", "Unknown")

        piece = f"[Source: {source}, Page No: {page}]\n{doc['page_content']}"

        formatted_pieces.append(piece)

    return "\n\n---\n\n".join(formatted_pieces)


def vector_search(
    query: str,
    client,
    collection_name: str,
    dense_embedder,
    sparse_embedder,
    i: int = 20,
    j: int = 10,
    k: int = 5,
    source_filter: str | None = None,
) -> str:

    with open("vec_data.json", "r") as f:
        data = json.load(f)

    pt_chks = data["pt_chks"]

    result = hybrid_search(
        query=query,
        client=client,
        collection_name=collection_name,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        i=i,
        j=j,
        k=k,
        source_filter=None,
    )

    parent_chunks = yield_parents(result, pt_chks)

    final_prompt_context = format_parents_for_llm(parent_chunks)
    logger.debug("final prompt context: %s", final_prompt_context)

    return final_prompt_context


if __name__ == "__main__":
    DENSE_MODEL = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL = "Qdrant/bm25"
    RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
    paths = [".downloads/new.txt"]

    def transform_paths(paths):
        result = []
        for path in paths:
            filename = PurePath(path).name
            stem = PurePath(filename).stem
            suffix = ".jsonl"
            result.append(f".data/.Cleaned/{stem}_{suffix}")
        return result

    paths = transform_paths(paths)
    docs = loader(paths)
    print(f"docs type: {type(docs)}")

    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)
    reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
    add_chunks(
        client=client,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        docs=docs,
    )
    print(
        vector_search(
            query="maya",
            client=client,
            collection_name="rag_collection",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
    )
    response = client.get_collections()

    # Extract and display the collection names
    collection_names = [col.name for col in response.collections]
    print("+"*70)
    print("Collection Names====================================")
    for collection_name in collection_names:

        print(collection_name)
        print("/n")
    print("+"*70)
    print("Collection Metadata====================================")

    for collection_name in collection_names:
        response, _next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=10,  # Number of points to return
        with_payload=True,  # Set False to omit metadata payload
        with_vectors=False,  # Set True to include vector embeddings
    )

    # Print point details
        for point in response:
            print(f"ID: {point.id}, Payload: {point.payload}")
            print("------------------------------")
            print("\n\n")


    client.close()
