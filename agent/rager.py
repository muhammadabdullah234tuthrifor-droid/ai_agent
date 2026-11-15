import json
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

import aiofiles
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent_session import load_thread
from llm import mock_sub_llm, tool_map
from logger import get_logger

logger = get_logger(name="ragger")


async def write(data):
    data = json.dumps(data) + "\n"
    async with aiofiles.open(".data/agent_logs.jsonl", "a") as f:
        await f.write(data)


class State(TypedDict):
    msgs: Annotated[list[BaseMessage], operator.add]
    uploaded_files: Annotated[list[dict[str, Any]], operator.add]
    marked_files: list[str]
    rslt: str


state = State


async def mem() -> list[BaseMessage]:
    """Loads the previous conversation history from saved threads. Returns past messages so the agent can remember previous context and continue the conversation naturally. Load the full conversation history from previous sessions. This is a very important memory tool. Always use this tool when the user refers to anything from past conversations, personal preferences, previous requests, location, country, time zone, or any context that was mentioned before. Examples: user asks about 'my country', 'today's hijri date', 'continue what we were discussing', 'my previous question', or anything personal. Use this tool frequently to maintain long-term memory and provide personalized responses."""
    conversation_history = await load_thread()
    return conversation_history


async def tool_call(state: state) -> state:
    last_message = state["msgs"][-1]

    tool_results = []
    marked_files = []
    for tool_call in last_message.tool_calls:  # type: ignore
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        await write(
            {"tool_name": tool_name, "tool_args": tool_args, "caller_agent": "ragger"}
        )
        if tool_name == "mem":
            result = await mem()
        else:
            tool = tool_map.get(tool_name)
            if tool is not None:
                if tool_name == "clean_doc":
                    marked_file = Path(tool_args.get("path"))
                    marked_files.append(marked_file)

                payload = interrupt(
                    {"tool_name": tool_name, "tool_args": tool_args, "status": "N"}
                )
                if payload["status"].strip() == "Y":
                    result = await tool.ainvoke(tool_args)
                    logger.info("LLM tool calls: %s", result)

                elif payload["status"].strip() == "E":
                    new_args = payload["tool_args"]
                    result = await tool.ainvoke(new_args)
                else:
                    result = "tool call interrupted"
            else:
                result = ["unknown tool. double check before calling tool."]
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    logger.info("tool result from ragger: %s", tool_results)
    return {
        "msgs": tool_results,
        "marked_files": marked_files,  # type: ignore
    }  # type: ignore


async def ragger(state: state) -> state:
    writer = get_stream_writer()
    writer({"status": "Retrieving..."})
    files = state.get("uploaded_files", [])
    marked_files = state.get("marked_files", [])
    files_info = []
    for file in files:
        file_name = Path(str(file.get("name", ""))).name
        if file_name and file_name not in marked_files:
            file_name = Path(str(file.get("name", "")))
            file_path = Path(".data/.downloads") / file_name
            files_info.append({"filename": file_name, "file_path": file_path})
    if files:
        tool = tool_map.get("download_files")
        logger.info("type of files: %s", files)
        info = await tool.ainvoke({"files": files})  # type: ignore
        logger.info("file: %s", info)
    logger.info("files received at ragger: %s", files_info)
    msg = state["msgs"] + [HumanMessage(content=" ".join(str(i) for i in files_info))]
    logger.debug("msgs: %s", msg)
    response = await mock_sub_llm.ainvoke(msg)

    logger.info("Ragger Messages: %s", msg)
    logger.info("Ragger Response: %s", response)

    return {"msgs": [response], "rslt": str(response.content)}  # type: ignore


def edge(state: state) -> str:
    last_message = state["msgs"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"


graph = (
    StateGraph(State)  # type: ignore
    .add_node("ragger", ragger)
    .add_node("tool_call", tool_call)
    .set_entry_point("ragger")
    .add_conditional_edges("ragger", edge, {"use_tool": "tool_call", "end": END})
    .add_edge("tool_call", "ragger")
)
rag_graph = graph.compile()

sys_msg = SystemMessage(
    content="you're a rag agent with a retrieve tool for searching documents for answering latest info. format and describe the returned result"
)
usr_msg = HumanMessage(content="who is the ceo of company")


async def main():
    await rag_graph.ainvoke({"msgs": [sys_msg, usr_msg]})
