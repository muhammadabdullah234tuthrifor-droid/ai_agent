import json
from pathlib import Path

from agent_memory import conversation_history


async def save_thread(thread_id: str, path: Path = Path("threads.json")) -> None:
    """Asynchronously save a thread ID to the JSON file."""
    try:
        if path.exists():
            content = path.read_text()
            all_thread_ids = json.loads(content) if content.strip() else []
        else:
            all_thread_ids = []
    except (FileNotFoundError, json.JSONDecodeError):
        all_thread_ids = []

    if thread_id not in all_thread_ids:
        all_thread_ids.append(thread_id)

    path.write_text(json.dumps(all_thread_ids))


async def load_thread(path: Path = Path(".data/threads.json")) -> list:
    """Asynchronously load all previous threads and extend conversation history."""
    from graph import build_graph

    graphg = await build_graph()

    try:
        if path.exists():
            content = path.read_text()
            all_thread_ids = json.loads(content) or []
        else:
            all_thread_ids = []
    except (FileNotFoundError, json.JSONDecodeError):
        all_thread_ids = []

    for thread_id in all_thread_ids:  # type: ignore
        # Use aget_state instead of get_state for async checkpointers in the main thread
        state = await graphg.aget_state({"configurable": {"thread_id": thread_id}})
        previous_messages = state.values.get("messages", [])
        conversation_history.extend(previous_messages)

    return conversation_history  # type: ignore
