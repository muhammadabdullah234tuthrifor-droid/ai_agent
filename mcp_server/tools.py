import ast
import asyncio
import json

import os
from functools import cache
from pathlib import PurePath
import aiofiles
from dotenv import load_dotenv
import base64
from pathlib import Path
import httpx
from langchain_core.messages import BaseMessage
from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from transformers import AutoTokenizer
from mcp.server.transport_security import TransportSecuritySettings   # may or may not be needed
from data_prap import UltimateDataPrep
from experiences import recall_similar_executions
from gdb import (
    detect_dupies,
    extract_graph,
    get_relationship_overlap,
    ingest_extraction_with_reporting,
    merge_entities_in_graph,
    verify_grounding,
)
from loader import loader
from logger import get_logger
from prac import hybrid_retrieve
from vec1 import add_chunks
import os
load_dotenv()

@cache
def get_embedders():
    from fastembed import SparseTextEmbedding, TextEmbedding

    return (
        TextEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        SparseTextEmbedding(model_name="Qdrant/bm25"),
    )


@cache
def get_qdrant():
    from qdrant_client import QdrantClient
    url= os.getenv("QDRANT_URL")
    return QdrantClient(url=url)


@cache
def get_driver():
    from neo4j import GraphDatabase
    uri= os.getenv("URI")
    auth= os.getenv("AUTH")
    return GraphDatabase.driver(uri, auth=auth) # type: ignore


@cache
def get_tokenizer():  # also fixes loading gpt2 per clean_doc call
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


logger = get_logger("tools")
# tool_server = FastMCP("toolserver", host= "0.0.0.0", port= 8000,  transport_security=TransportSecuritySettings(
#        allowed_hosts=["*"],  # dev only; restrict in production
#    ),)
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"






os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        # Docker network
        "mcp-server",
        "mcp-server:8000",
        "mcp-server:*",

        # Local development
        "localhost",
        "localhost:8000",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:8000",
        "127.0.0.1:*",
    ],
)

tool_server = FastMCP(
    "toolserver",
    host="0.0.0.0",
    port=8000,
    transport_security=transport_security,
)

# IMPORTANT: Create the ASGI application AFTER
# configuring transport security.
app = tool_server.streamable_http_app()








# ===========================================
# playwright
# ==========================================


ALLOWED_IMPORTS = {
    "asyncio",
    "playwright.async_api",
}


ALLOWED_ROOT_NAMES = {
    "page",
    "context",
    "browser",
    "playwright",
    "asyncio",
}


BLOCKED_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}


BLOCKED_ATTRIBUTES = {
    "__class__",
    "__dict__",
    "__bases__",
    "__base__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__import__",
}


class PlaywrightCodeValidator(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []

    def fail(self, node, message: str):
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node):

        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                self.fail(
                    node,
                    f"Import not allowed: {alias.name}",
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node):

        module = node.module or ""

        if module not in ALLOWED_IMPORTS:
            self.fail(
                node,
                f"Import not allowed: {module}",
            )

        self.generic_visit(node)

    def visit_Name(self, node):

        if node.id in BLOCKED_NAMES:
            self.fail(
                node,
                f"Name not allowed: {node.id}",
            )

        self.generic_visit(node)

    def visit_Attribute(self, node):

        if node.attr.startswith("_"):
            self.fail(
                node,
                f"Private attribute not allowed: {node.attr}",
            )

        if node.attr in BLOCKED_ATTRIBUTES:
            self.fail(
                node,
                f"Attribute not allowed: {node.attr}",
            )

        self.generic_visit(node)

    def visit_Lambda(self, node):

        self.fail(
            node,
            "Lambda expressions are not allowed",
        )

    def visit_ClassDef(self, node):

        self.fail(
            node,
            "Class definitions are not allowed",
        )

    def visit_FunctionDef(self, node):

        self.fail(
            node,
            "Function definitions are not allowed",
        )

    def visit_AsyncFunctionDef(self, node):

        self.fail(
            node,
            "Function definitions are not allowed",
        )

    def visit_While(self, node):

        self.fail(
            node,
            "while loops are not allowed",
        )

    def visit_ImportStar(self, node):

        self.fail(
            node,
            "Wildcard imports are not allowed",
        )


def validate_code(code: str) -> None:

    if len(code) > 20_000:
        raise ValueError("Code exceeds maximum length of 20,000 characters.")

    try:
        tree = ast.parse(
            code,
            mode="exec",
        )
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    validator = PlaywrightCodeValidator()

    validator.visit(tree)

    if validator.errors:
        raise ValueError("Code rejected:\n" + "\n".join(validator.errors))


# ==================


async def execute_in_sandbox(
    code: str,
    timeout: int = 90,
) -> dict:

    validate_code(code)

    payload = serialize_request(code)

    process = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        # Keep stdin connected to the container.
        "-i",
        "-p",
        "6080:6080",
        # Security.
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        # Resource limits.
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--pids-limit",
        "128",
        # Chromium shared memory.
        "--shm-size",
        "256m",
        # Temporary filesystem.
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=256m",
        # Browser needs network access.
        "--network",
        "bridge",
        # Give Chromium access to the host X11 display
        "-e",
        f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
        "-v",
        "/tmp/.X11-unix:/tmp/.X11-unix:rw",
        "playwright-sandbox",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(payload),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
        process.kill()

        await process.wait()

        return {
            "ok": False,
            "error": "Timeout",
            "message": (f"Execution exceeded {timeout} seconds."),
        }

    stdout_text = stdout.decode(
        "utf-8",
        errors="replace",
    )

    stderr_text = stderr.decode(
        "utf-8",
        errors="replace",
    )

    if not stdout_text.strip():
        return {
            "ok": False,
            "error": "SandboxProducedNoOutput",
            "message": stderr_text[-10_000:],
        }

    try:
        result = json.loads(stdout_text)

    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "InvalidSandboxOutput",
            "stdout": stdout_text[-10_000:],
            "stderr": stderr_text[-10_000:],
        }

    if stderr_text.strip():
        result["stderr"] = stderr_text[-10_000:]

    return result


# ========================


def serialize_request(code: str) -> bytes:

    payload = {
        "code": code,
    }

    return json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")


@tool_server.tool()
async def run_playwright(
    code: str,
    timeout: int = 480,
) -> dict:
    """Execute Python Playwright code inside a sandboxed browser environment.

    Use this tool to automate browser interactions, inspect web pages, scrape content,
    take screenshots, or test web applications. The code runs asynchronously within an
    isolated, disposable container.

    Args:
        code: A string containing valid Python code using `async/await`. The code has
          access to pre-instantiated globals:
            - `page`: Playwright Page instance
            - `context`: Playwright BrowserContext instance
            - `browser`: Playwright Browser instance
            - `playwright`: Playwright instance
            - `asyncio`: Python `asyncio` module

          Constraints for `code`:
            - Must return a JSON-serializable value (dict, list, string, number, bool).
            - Do NOT use standard filesystem, `subprocess`, `os`, or shell APIs.
            - All web interactions must go through the provided `page` or `context`.

        timeout: Execution timeout in seconds (default: 30). Automatically clamped
          between 1 and 120 seconds.

    Returns:
        dict: A dictionary containing the result of the execution.
            - On success: contains the returned value from the evaluated code.
            - On failure/timeout: contains error details and message.
    """

    if timeout < 1:
        timeout = 1

    if timeout > 120:
        timeout = 120

    # result = await execute_in_sandbox(
    #     code=code,
    #     timeout=timeout,
    # )

    async with httpx.AsyncClient(timeout=timeout + 10) as client:
            resp = await client.post(
                "http://playwright-sandbox:8080/execute",   # internal name
                json={"code": code, "timeout": timeout}
            )
            return resp.json()



@tool_server.tool()
async def search(
    query: str,
    max_results: int = 3,
) -> str:
    """Search the web for the given query and return extracted text content from the top results. Use this tool when you need up-to-date or current information on any topic. It performs a search and then scrapes the actual page content (not just snippets) from the best matching results, then returns them nicely formatted so you can read and cite them easily."""

    endpoint = "http://localhost:3800/scrape"

    # Use AsyncClient instead of regular Client
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            # Notice the 'await' keyword!
            response = await client.post(
                endpoint,
                json={"keywords": query, "max_results": max_results},
            )
            response.raise_for_status()

        except httpx.RequestError as e:
            return f"Error: could not reach scraping server — {e}"
        except httpx.HTTPStatusError as e:
            return f"Error: scraping server returned {e.response.status_code}"

    # Everything after the network call stays almost identical
    data = response.json()

    successful = [
        item
        for item in data.get("results", [])
        if item.get("status") == "success" and item.get("text", "").strip()
    ]

    if not successful:
        return (
            f"No content could be scraped for query: '{query}'. "
            f"{data.get('failed', 0)} of {data.get('total_requested', 0)} URLs failed."
        )

    # Build structured text
    chunks = []
    for item in successful:
        title = item.get("title") or "Untitled"
        url = item["url"]
        text = item["text"][:3000]  # Truncate to prevent token explosion
        chunks.append(f"--- Source: {title}\nURL: {url}\n\n{text}")

    return "\n\n".join(chunks)


@tool_server.tool()
async def retrieve(
    query: str, filter: str | None = None, collection_name: str = "rag_collection"
) -> str:
    """Retrieve relevant document information using hybrid search.

    Use this tool to find information in the indexed documents. Provide a
    natural-language query and, optionally, a filter to narrow the search.

    Args:
        query: The question or search terms to look up.
        filter: Optional filter expression restricting which documents are searched. Don't provide any filter.
        collection_name: Name of the document collection to search. Don't provide the collection name and let the default name to be used automatically.

    Returns:
        Retrieved information as a string.
    """
    dense_embedder, sparse_embedder = get_embedders()
    client = get_qdrant()
    driver = get_driver()
    restlt = hybrid_retrieve(
        query=query,
        filter=filter,
        client=client,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        driver=driver,
        collection_name=collection_name,
        entity_collection_name="entity_collection",
    )
    return f"That's the retrieved result, you can give to the user: {restlt}"


logger.info("module tools.py called.")


@tool_server.tool()
async def clean_doc(path: str) -> str:
    """
    turns files into clean chunks, ready to be ingest into a rag pipeline.
    always call this function on document before ingesting it into a database.
    argument: path(str): the full path(not just the name. it already starts with .data/.downloads. like .data/.downloads/file.txt) of the file to be cleaned.
    """
    logger.debug("file path is: %s", path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    try:
        processer = UltimateDataPrep(path, tokenizer) # type: ignore
        processer.yolo()
    except Exception as e:
        logger.debug("error cleaning file: %s", e)
        return "error occured while cleaning the document, don't retry, instead refuse to the user"
    return "document cleaned"


@tool_server.tool()
async def recall(query: str) -> str:
    """Retrieve execution blueprints of past similar tasks from memory.

    Searches vector/graph history for execution traces, steps taken, and tool calls
    used in previous similar user tasks.

    Args:
        query: The task description or user goal to search for in past executions.

    Returns:
        List of top-matching task blueprints detailing agent actions and outcomes.
    """
    dense_embedder, _ = get_embedders()
    client = get_qdrant()
    driver = get_driver()
    blueprints = recall_similar_executions(
        driver, client, dense_embedder, query, top_k=2
    )
    logger.debug("recalled: %s", blueprints)
    return blueprints


@tool_server.tool()
async def ingest_or_list(
    collection_name: str = "rag_collection",
    paths: list[str] | None = None,
    get_info: bool = False,
    doc_id: str = "",
    vector_ingestion: bool = True,
    graph_ingestion: bool = True,
):
    """Manages document ingestion into vector and knowledge graph databases, or lists Vector DB collections.

    This function handles persistent collection setup in qdrant, loads documents, and routes data to:
    1. A qdrant vector database for embeddings-based ingestion.
    2. A Neo4j graph database for entity and relationship extraction, grounding, duplicate detection, and graph ingestion.
    3. path: a list of one or more paths of the files. if not path is provided, then all the files would be ingested.
    Args:
        collection_name (str): The target qdrant collection name to retrieve or create, if user do not provide any name for collection or do not ask to create a new collection then leave it as is. it would use default collection.
        get_info (bool, optional): If True, lists available qdrant collections and logs item counts. Defaults to False.
        doc_id: id of document, in case of vector ingestion
        vector_ingestion (bool, optional): If True, triggers embedding generation and ingestion of document chunks into qdrant. Defaults to False.
        graph_ingestion (bool, optional): If True, extracts entities/relations, resolves duplicate entities, and ingests knowledge into Neo4j. Defaults to False.

    Returns:
        str | None: A status string indicating the success or failure of vector/graph ingestion operations, or None if no ingestion flag is set.
    """

    def transform_paths(paths):
        result = []
        for path in paths:
            filename = PurePath(path).name
            stem = PurePath(filename).stem
            suffix = ".jsonl"
            result.append(f".data/.Cleaned/{stem}_{suffix}")
        return result

    dense_embedder, sparse_embedder = get_embedders()
    client = get_qdrant()
    driver = get_driver()
    paths = transform_paths(paths)
    logger.debug("pahts at inget: %s", paths)

    if get_info:
        response, _next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=10,  # Number of points to return
            with_payload=True,  # Set False to omit metadata payload
            with_vectors=False,  # Set True to include vector embeddings
        )

        # Print point details
        for point in response:
            return f"ID: {point.id}, Payload: {point.payload}"

    if vector_ingestion:
        data = loader(paths)

        try:
            add_chunks(
                client=client,
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
                docs=data,
                collection_name=collection_name,
            )
            return "data ingested in vector database"

        except Exception as e:
            logger.debug("could not be ingest in qdrant: %s", e)
            return f"data could not be ingest in vector database: {e}"

    if graph_ingestion:
        data = loader(paths, graph_ingestion)

        try:
            facts = extract_graph(data)
            verified_entity_rel = verify_grounding(facts, facts, 0.85)
            dupies = detect_dupies(
                verified_entity_rel.entities, verified_entity_rel.relations, 85, 0.85
            )
            dupies = get_relationship_overlap(driver, dupies, 0.85)
            stats = ingest_extraction_with_reporting(
                dense_embedder,
                driver,
                "entity_collection",
                verified_entity_rel,
                doc_id,
                200,
            )
            logger.debug(stats)

            if dupies:
                for dupy in dupies:
                    if len(dupy[0].name) >= len(dupy[1].name):
                        merge_entities_in_graph(
                            driver,
                            canonical_name=dupy[0].name,
                            alias_name=dupy[1].name,
                            node_type=dupy[0].type.value,
                        )

            return f"data ingested in graph database: {stats}"

        except Exception as e:
            logger.debug("could not be ingest in neo4j: %s", e)
            return f"data could not be ingest in graph database: {e}"

@tool_server.tool()
async def download_files(files) -> str:
    logger.debug("============================= \n\n tool called \n ================")
    if not files:
        return "no files"

    downloads_dir = Path(".data/.downloads")

    # Fully non-blocking directory creation (compatible with Python 3.9+)
    await asyncio.to_thread(lambda: downloads_dir.mkdir(parents=True, exist_ok=True))

    async def save_file(file_item):
        safe_name = Path(file_item.get("name", "unknown_file")).name
        b64_str = file_item.get("data", "")

        if not b64_str:
            print(f"⚠️ Skipping {safe_name}: No base64 data found.")
            return

        # Fully non-blocking CPU decode
        decoded_bytes = await asyncio.to_thread(base64.b64decode, b64_str)
        output_path = downloads_dir / safe_name

        # Fully non-blocking disk I/O
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(decoded_bytes)

        logger.debug(f"✅ File downloaded,,,,,,,,,,: {output_path}")

    dict_files=[]
    for file in files:
        if type(file).__name__=="str":
            file= ast.literal_eval(file)
        dict_files.append(file)

    await asyncio.gather(*(save_file(file_item) for file_item in dict_files))
    return "file downloaded"


if __name__ == "__main__":
    tool_server.run(transport="streamable-http")
