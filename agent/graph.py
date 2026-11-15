import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from node import (
    agent,
    final_gate,
    finalizer,
    memer,
    message_analyzer,
    ragger,
    web_scraper,
)
from psycopg_pool import AsyncConnectionPool
from state import AgentState

load_dotenv()
DB_URI = os.getenv("DB_URI")

# Keep these at module level so they survive for the whole process lifetime
_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def get_checkpointer() -> AsyncPostgresSaver:
    global _pool, _checkpointer

    if _checkpointer is None:
        _pool = AsyncConnectionPool(
            conninfo=DB_URI,  # type: ignore
            min_size=2,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await _pool.open()

        _checkpointer = AsyncPostgresSaver(_pool)  # type: ignore
        await _checkpointer.setup()

    return _checkpointer


async def close_resources():
    """Call this on application shutdown."""
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
        _pool = None
        _checkpointer = None


async def build_graph():
    memory = await get_checkpointer()

    build = StateGraph(AgentState)  # type: ignore

    build.add_node("agent", agent)
    build.add_node("memer", memer)
    build.add_node("web_scraper", web_scraper)
    build.add_node("ragger", ragger)
    build.add_node("finalizer", finalizer)
    build.add_node("final_gate", final_gate)

    build.set_entry_point("agent")

    build.add_conditional_edges(
        "agent",
        message_analyzer,
        {
            "continue": "agent",
            "finalizer": "finalizer",
            "end": END,
        },
    )

    build.add_edge("ragger", "agent")
    build.add_edge("web_scraper", "agent")
    build.add_edge("finalizer", "final_gate")
    build.add_edge("final_gate", END)

    return build.compile(checkpointer=memory)
