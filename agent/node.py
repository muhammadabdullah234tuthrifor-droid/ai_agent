import asyncio
import json
import random
import time
import uuid
from enum import Enum
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
import aiofiles

from google.genai import errors
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from agent_session import save_thread
from llm import mock_sub_llm, mock_sup_llm
from logger import get_logger
from memer import mem_graph
from rager import rag_graph
from state import AgentState
from web_scraper import web_graph


async def write(data):
    data = json.dumps(data) + "\n"
    async with aiofiles.open("agent_logs.jsonl", "a") as f:
        await f.write(data)


class ErrorType(Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


start = time.time()
state = AgentState
logger = get_logger(name="Nodes")


async def agent(state: state) -> state: # type: ignore
    # thread_id = state["thread_id"]  # type: ignore
    thread_id = state.get("thread_id") or str(uuid.uuid4())
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 0)
    sys_msg = [
        {
            "role": "system",
            "content": """\
            You are the SUPERVISOR of a multi-agent system. You never answer the user
            directly. Your only job is to analyze the conversation and dispatch tasks to
            the right specialist agents.

            ## YOUR AGENTS
            - memer: loads long-term memory (previous conversations). Use for ANY
              reference to past chats, e.g. "as I said before", "my country", "my name",
              "continue what we discussed", personal preferences, earlier results.
            - ragger: searches the user's UPLOADED DOCUMENTS. Use for summarize, extract,
              search inside files, compare file contents.
            - web_scraper: searches the live web. Use when the answer needs current or
              recent information: news, prices, weather, releases, anything after your
              training data.
            - finalizer: the LAST step, always. It formats the final answer for the user
              from the collected results.

            ## ROUTING RULES
            1. Decide what the user actually needs before dispatching anything.
            2. Dispatch independent memer / ragger / web_scraper tasks TOGETHER in one
               response so they run in parallel.
            3. If a request touches memory, documents AND the web, use all three.
            4. Send a task to finalizer ONLY after all other needed agents have returned
               their results. Exactly one finalizer task, never more.
            5. If nothing needs to be retrieved, send one finalizer task directly.
            6. If the user says goodbye, send one finalizer task to produce a farewell.

            ## HOW TO WRITE A TASK
            Workers are blind: they see ONLY the `task` text, never the conversation.
            Every task must be fully self-contained.
            - Repeat the user's exact question inside the task.
            - Include all names, dates, places, file names and constraints needed.
            - Bad:  "search for it"
            - Good: "Find today's weather in Murree, Pakistan. Return temperature,
              conditions and a clothing suggestion."
            For the finalizer task: paste a compact digest of EVERY collected result plus
            the original user question into the `task` field. Nothing may be lost here.

            ## OUTPUT CONTRACT
            Respond ONLY with the structured output:
            - reasoning: 1-2 sentences on what you decided and why.
            - task_description: one-line summary of what the user wants.
            - tasks: list of {agent, task, task_name, id} where:
              - agent: exactly one of "memer", "ragger", "web_scraper", "finalizer".
              - task: the self-contained instruction.
              - task_name: 2-4 word label, e.g. "weather_lookup".
              - id: short unique id, e.g. "t1", "t2".
            Never invent agent names. Never leave work undone: if results are already in
            the conversation, your next step is finalizer.
            """
,
        },
    ]
    sup_agent = state.get("messages", [])


    sup_agent = sys_msg + sup_agent
    print(f"\n Messages Received: {sup_agent} \n")

    carry_on = True
    base_delay: float = 1.0

    human_msg: HumanMessage = HumanMessage(content="")
    for msg in reversed(sup_agent):
        if isinstance(msg, HumanMessage):
            human_msg = msg
            logger.debug("Human Message: %s", human_msg)
            break
    if human_msg.content.strip().lower().startswith("bye"):  # type: ignore
        logger.info("Goodbye!")
        carry_on = False

    await save_thread(thread_id)
    logger.debug("Full conversation going to agent: %s", sup_agent)
    last_exception: Exception | None = None

    for attempt in range(max_iterations + 1):
        try:
            response = await mock_sup_llm.ainvoke(sup_agent)
            logger.debug("Agent Response: %s", response)
            # response = response.model_dump()  # type: ignore

            response = json.loads(response.content) # type: ignore
            await write(response)
        except errors.ClientError as e:
            last_exception = e
            error_type = ErrorType.TRANSIENT
            print(
                f"  [Retry] Rate limited (attempt {attempt + 1}/{max_iterations + 1})"
            )

        except errors.ServerError as e:
            last_exception = e
            error_type = ErrorType.TRANSIENT
            print(
                f"  [Retry] Server error (attempt {attempt + 1}/{max_iterations + 1})"
            )

        except errors.APIError as e:
            last_exception = e
            error_type = ErrorType.TRANSIENT
            logger.warning("API error (attempt %s/%s)", attempt + 1, max_iterations + 1)

        except Exception as e:
            last_exception = e
            error_type = ErrorType.UNKNOWN
            logger.warning("Unexpected error (attempt %s/%s): %s", attempt + 1, max_iterations + 1, e)
        else:
            responsee = AIMessage(content=str(response))
            return {
                "messages": [responsee],
                "tasks": response.get("tasks", []),
                "max_iterations": state.get("max_iterations", 5),
                "iteration_count": iteration_count + 1,
                "carry_on": carry_on,
            }  # type: ignore

        # Backoff before next attempt
        if attempt < max_iterations:
            delay = base_delay * (2**attempt)
            jitter = random.uniform(0, delay * 0.3)
            total_delay = delay + jitter
            logger.info("Waiting %.1fs before retry...", total_delay)
            await asyncio.sleep(total_delay)  # Non-blocking async sleep

    if last_exception:
        raise last_exception




async def web_scraper(state: state) -> state:
    sys_msg = SystemMessage(
        content= """\
       You are WEB_SCRAPER, the web research specialist of a multi-agent system.

       ## YOUR TOOL
       `search(query, max_results)` — searches the web, scrapes the top pages and
       returns their text with titles and URLs.

       ## PROCEDURE
       1. Turn the task into one or two precise search queries.
       2. Call `search`.
       3. Read the scraped text and extract only what answers the task.
       4. Reply with a report: the direct answer first, then the key supporting
          facts, then the list of sources (title + URL).

       ## RULES
       - Tool calls are reviewed by a human operator before they run. Write queries
         that a human can understand at a glance.
       - Prefer recent and authoritative sources; if results conflict, say so.
       - If the search fails or returns nothing relevant, report the failure. NEVER
         invent facts, numbers or URLs.
       - Do not answer from your own knowledge: your value is live web data.
       - Your final message is a report that will be forwarded to the supervisor
         as-is. No greetings, no filler.
       """
    )
    task = state.get("current_task", "")
    logger.debug("task received at web scrapper: %s", task)


    usr_msg = HumanMessage(content=task)
    logger.debug("task received at web scrapper: %s", usr_msg)
    response = await web_graph.ainvoke(
        {"msgs": [sys_msg, usr_msg]}  # type: ignore
    )
    logger.debug("response from web scraper: %s", response)
    result = f" result from web_scraper: {response['rslt']}"
    result = AIMessage(content=str(result))
    logger.debug("result: %s", result)

    return {
        "messages": [result],

    } # type: ignore


async def memer(state: state) -> state:
    sys_msg = SystemMessage(
        content= """\
       You are MEMER, the memory specialist of a multi-agent system.

       ## YOUR TOOL
       `mem` — loads the full saved conversation history from previous sessions:
       past messages, user preferences, personal details, earlier questions and
       answers.

       ## PROCEDURE
       1. Call `mem` once to load the history.
       2. Search the returned history for exactly what the task asks for.
       3. Reply with a compact, factual report of ONLY the relevant findings.

       ## RULES
       - Tool calls are reviewed by a human operator before they run. Make each call
         clean and self-explanatory.
       - If memory contains the answer, quote or closely paraphrase it and note that
         it comes from a previous conversation.
       - If memory is empty or does not contain the information, say exactly that.
         NEVER invent past conversations, names, dates or preferences.
       - Do not answer general-knowledge or web questions: that is not your job.
       - Your final message is a report that will be forwarded to the supervisor
         as-is. No greetings, no filler.
       """
    )
    task = state.get("current_task", "")
    logger.info("task received at memer: %s", task)

    usr_msg = HumanMessage(content=task)

    response = await mem_graph.ainvoke(
        {"msgs": [sys_msg, usr_msg]}  # type: ignore
    )
    logger.info("response from memer: %s", response)
    result = f" result from memer: {response['rslt']}"

    result = AIMessage(content=str(result))
    logger.info("result: %s", result)

    return {
        "messages": [result],
    } # type: ignore


async def ragger(state: state) -> state:

    sys_msg = SystemMessage(
        content="""\
        You are RAGGER, the document specialist of a multi-agent system.

        ## YOUR FILES
        You receive uploaded file info as a message: filename plus file_path inside
        the .data/.downloads/ directory. Use exactly those paths.

        ## YOUR TOOLS
        - clean_doc(path): converts a file into clean chunks. ALWAYS run this on a
          file before ingesting it.
        - ingest_or_list(...): ingests cleaned files into the vector and graph
          databases, or lists collections (get_info=True).
        - retrieve(query, ...): hybrid search over the ingested documents.

        ## PROCEDURE
        1. If the needed file is not ingested yet: clean_doc -> ingest_or_list (set
           vector_ingestion / graph_ingestion as required) -> retrieve.
        2. If it is already ingested, go straight to retrieve.
        3. Make each retrieve query precise and self-contained; one intent per query.
        4. Reply with a report: the findings, each tied to the file it came from.

        ## RULES
        - Tool calls are reviewed by a human operator before they run. Use exact file
          paths and correct arguments.
        - If clean_doc fails, do NOT retry and do not call it again: report that the
          document cannot be processed.
        - Never fabricate document content. If retrieval returns nothing relevant,
          say the documents do not contain the answer.
        - Do not answer from general knowledge, and do not search the web: that is
          not your job.
        - Your final message is a report that will be forwarded to the supervisor
          as-is. No greetings, no filler.
        """
    )
    task = state.get("current_task", "")
    logger.info("task received at ragger: %s", task)
    ragger_context = state.get("ragger_context", [])

    usr_msg = HumanMessage(content=task)
    logger.info("task received at ragger: %s", usr_msg)

    files = state.get("uploaded_files", [])
    
    # print(files)
    msgs= [sys_msg, usr_msg] + ragger_context
    response = await rag_graph.ainvoke(
        {"msgs":  msgs, "uploaded_files": files}  # type: ignore
    )
    logger.info("response from ragger: %s", response)
    result = f" result from ragger: {response['rslt']}"
    ragger_context= response["msgs"]

    result = AIMessage(content=str(result))
    logger.info("result: %s", result)

    return {
        "messages": [result],
        "ragger_context": ragger_context
    } # type: ignore



async def finalizer(state: state) -> state:
    sys_msg = SystemMessage(
        content="you have to prepare the final respone according to the given task and context."
    )

    current_task = state.get("current_task")
    logger.debug("Task Received at finalizer: %s", current_task)

    usr_msg = HumanMessage(content=current_task)

    logger.debug("Full conversation going to finalizer: %s \t %s", sys_msg, usr_msg)
    response = await mock_sub_llm.ainvoke([sys_msg, usr_msg])
    logger.debug("Finalizer Response: %s", response)
    logger.debug("Elapsed: %.2fs", time.time() - start)





    return {"final_response": str(response.content), "carry_on": False}  # type: ignore


async def final_gate(state) -> state:
    feedback_payload = interrupt({
        "type": "final_feedback",
        "question": "Was that what you want?",
    })
    status = (feedback_payload or {}).get("status", "N")
    if status == "Y":
        await write({"Result": "Success"})
        custom_event = {"Result": "Success"}

        get_stream_writer()(custom_event)
    else:
        note = (feedback_payload or {}).get("feedback", "")
        await write({"Result": "Fail", "Feedback": note})
        custom_event = {"Result": "Fail", "Feedback": note}
        get_stream_writer()(custom_event)


    return {"carry_on": False}   # type: ignore

def message_analyzer(state: state) -> list[Send] | str:
    carry_on = state.get("carry_on")
    max_iterations = state.get("max_iterations", 5)
    iteration_count = state.get("iteration_count", 0)
    current_tasks = state.get("tasks", [])
    files = state.get("uploaded_files", [])

    valid_routes = {"web_scraper", "ragger", "memer", "finalizer"}
    if iteration_count >= max_iterations:
        logger.warning("max iteration reached")
        logger.debug("returning end")
        return "end"

    elif not carry_on:
        logger.debug("returning end")
        return "end"

    else:
        results = [
            Send(
                task["agent"], # type: ignore
                {"current_task": task["task"], "uploaded_files": files}, # type: ignore
            )
            for task in current_tasks
        ]
        logger.debug("Sending tasks...")
        return results
