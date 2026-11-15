import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
from logger import get_logger
from mockllm import FakeToolCallingChatModel
logger= get_logger("llm")

class Outputt(BaseModel):
    agent: str = Field(
        description="the name of agent for the specific task. available agents: 'memer(an agent that has tool for recalling previous memory)'| 'ragger(a agent for ingesting documents in database and retrieving information from documents, use this agent when the user asks any question about the uploaded document.' | 'web_scraper(an agent for interacting with browser and searching websites, use this agent when the user want any latest information from internet)' | 'finalizer(when the task is done, then send all the information to this agent to format for the user)'"
    )

    task: str = Field(description="the specific instruction for the chosen agent.")
    task_name: str = Field(description="A short name suited for the task")
    id: str = Field(description="a unique id for the task")


class Output(BaseModel):
    reasoning: str = Field(description="your brief reasoning about what to do next")
    task_description: str = Field(description="a brief summary of the task or question given by user")
    tasks: list[Outputt] = Field(description="List of tasks to execute in parallel.")


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

sup_model = "gemini-2.5-flash-lite"
sub_model = "gemini-2.5-flash-lite"

sup_llm = ChatGoogleGenerativeAI(
    model=sup_model,
    api_key=api_key,
).with_structured_output(Output)
sub_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=api_key)
mcp_url= os.getenv("MCP_URL")
async def load_tools(client):
    return await client.get_tools()

client = MultiServerMCPClient(
{
    "toolserver": {
        "url": mcp_url,
        "transport": "streamable-http",


    }
}) # type: ignore

tool_schema = asyncio.run(load_tools(client))






tool_map = {tool.name: tool for tool in tool_schema}


sub_llm_with_tools = sub_llm.bind_tools(tool_schema)


sub_scripted_responses = iter(
    [
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "run_playwright",
                    "arguments": '{"code":"" }',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "run_playwright",
                    "args": {"code":"""
await page.goto("https://www.youtube.com/results?search_query=harry+potter")
page.set_default_timeout(600000)
page.set_default_navigation_timeout(600000)
# Wait for video titles to render
await page.wait_for_selector("a#video-title")

links = await page.locator("a#video-title").all()
results = []

for link in links[:10]:
    text = await link.inner_text()
    href = await link.get_attribute("href")

    if href and href.startswith("/"):
        href = f"https://www.youtube.com{href}"

    results.append({
    "text": text.strip(),
    "href": href,
    })

# Click the first video to open and play it
if links:
    await links[0].click()

# Watch the video for 50 seconds
await asyncio.sleep(70)

return results
""" },
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "mem",
                    "arguments": '{}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "mem",
                    "args": {},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),

        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "search",
                    "arguments": '{"query": "todays weather"}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "todays weather"},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),

        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "clean_doc",
                    "arguments": '{"path": ".data/.downloads/newest.txt"}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "clean_doc",
                    "args": {"path": ".data/.downloads/newest.txt"},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "ingest_or_list",
                    "arguments": '{"paths": [".data/.downloads/para.txt"]}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "ingest_or_list",
                    "args": {"paths": [".data/.downloads/para.txt"]},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "retrieve",
                    "arguments": '{"query": "clinical research"}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "retrieve",
                    "args": {"query": "clinical research"},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),

        AIMessage(content="weather is kinda pleasant today!"),
        AIMessage(content="ceo is someone."),
        AIMessage(content="convo loaded."),

        AIMessage(content="finalized the response."),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "search",
                    "arguments": '{"query": "todays ..."}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "weather today"},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "search",
                    "arguments": '{"query": "weather today"}',
                }
            },
            response_metadata={
                "finish_reason": "STOP",
                "model_name": "gemini-2.5-flash-lite",
                "safety_ratings": [],
                "model_provider": "google_genai",
            },
            id="lc_run--019f700f-85ad-7de1-9637-1e795bc70228-0",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "2+3"},
                    "id": "f0d45375-ffbe-45c5-9ab4-5f9446bdc05f",
                    "type": "tool_call",
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                "input_tokens": 435,
                "output_tokens": 15,
                "total_tokens": 450,
                "input_token_details": {"cache_read": 0},
            },
        ),
    ]
)

sup_scripted_responses = iter(
    [
        # ── Response 1: call both tools simultaneously ──────────────────────────
        AIMessage(
            content=json.dumps(
                {
                    "reasoning": "need to call the agents",
                    "task_description": "user asked for...",
                    "tasks": [
                        {"agent": "web_scraper", "task": "web search", "id": ""},
                        {"agent": "ragger", "task": "retrieval task", "id": ""},
                        {"agent": "memer", "task": "mem task", "id": ""},
                    ],
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {
                    "reasoning": "need to call finalizer",
                    "task_description": "user asked for...",
                    "tasks": [
                        {"agent": "finalizer", "task": "synthasize results", "id": ""}
                    ],
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {
                    "reasoning": "need to call finalizer",
                    "next_agent": "finalizer",
                    "task": "weather is fine",
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {
                    "reasoning": "some reasoning",
                    "next_agent": "web_scraper",
                    "task": "some_task",
                }
            )
        ),
        AIMessage(
            content=json.dumps(
                {
                    "reasoning": "some reasoning",
                    "next_agent": "finish",
                    "task": "some_task",
                }
            )
        ),
    ]
)


mock_sup_llm = FakeToolCallingChatModel(responses=sup_scripted_responses)
mock_sub_llm = FakeToolCallingChatModel(responses=sub_scripted_responses)
