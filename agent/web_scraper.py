import json
import operator
from typing import Annotated, TypedDict

import aiofiles
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from llm import mock_sub_llm, tool_map
from logger import get_logger

logger = get_logger(name="web_scraper")


async def write(data):
    data = json.dumps(data) + "\n"
    async with aiofiles.open(".data/agent_logs.jsonl", "a") as f:
        await f.write(data)


class State(TypedDict):
    msgs: Annotated[list[BaseMessage], operator.add]
    rslt: str


state = State


async def tool_call(state: state) -> state:
    last_message = state["msgs"][-1]

    tool_results = []
    for tool_call in last_message.tool_calls:  # type: ignore
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        await write(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "caller_agent": "web_scraper",
            }
        )

        tool = tool_map.get(tool_name)
        result = []
        if tool is not None:
            payload = interrupt(
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "status": "N",
                }
            )
            if payload["status"].strip() == "Y":
                result = await tool.ainvoke(tool_args)

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
        logger.debug("tool result from web_scraper: %s", tool_results)
    return {"msgs": tool_results}  # type: ignore


async def web_scraper(state: state) -> state:
    logger.debug("messages at web scraper before calling llm: %s", state["msgs"])
    response = await mock_sub_llm.ainvoke(state["msgs"])
    logger.debug("Message in web scraper: %s", state["msgs"])
    logger.debug("Web Scraper Response: %s", response)

    return {"msgs": [response], "rslt": str(response.content)}  # type: ignore


def edge(state: state) -> str:
    last_message = state["msgs"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return "end"


graph = (
    StateGraph(State)  # type: ignore
    .add_node("web_scraper", web_scraper)
    .add_node("tool_call", tool_call)
    .set_entry_point("web_scraper")
    .add_conditional_edges("web_scraper", edge, {"use_tool": "tool_call", "end": END})
    .add_edge("tool_call", "web_scraper")
)
web_graph = graph.compile()

sys_msg = SystemMessage(
    content="you're a specialized web researcher with a search tool for searching web for answering latest info. format and describe the returned result."
)
usr_msg = HumanMessage(content="what's the weather in muree")


async def main():
    await web_graph.ainvoke({"msgs": [sys_msg, usr_msg]})
