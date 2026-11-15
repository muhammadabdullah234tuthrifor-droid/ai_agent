from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator
from langgraph_sdk import get_client
from langchain_core.messages import BaseMessage
import asyncio
import json
import uuid
import base64
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangGraph API with Multi-HITL v5",
    description="Handles parallel multi-interrupt cycles correctly",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = get_client(url="http://127.0.0.1:2024")
thread_metadata: Dict[str, Dict] = {}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    max_iterations: int = 7

class SingleToolApproval(BaseModel):
    interrupt_id: str
    decision: str = Field(..., pattern="^(Y|N|E)$")
    tool_args: Optional[Dict[str, Any]] = None
    feedback: Optional[str] = None

class BatchApprovalRequest(BaseModel):
    thread_id: str
    approvals: List[SingleToolApproval]


# ═══════════════════════════════════════════════════════════════
# Safe Serialization
# ═══════════════════════════════════════════════════════════════

def _safe_json_serialize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_safe_json_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _safe_json_serialize(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return _safe_json_serialize(obj.model_dump())
    if hasattr(obj, "dict"):
        return _safe_json_serialize(obj.dict())
    if isinstance(obj, BaseMessage):
        return {
            "type": obj.type,
            "content": getattr(obj, "content", None),
            "tool_calls": getattr(obj, "tool_calls", None),
        }
    if hasattr(obj, "__dict__"):
        return _safe_json_serialize(obj.__dict__)
    try:
        return str(obj)
    except Exception:
        return f"<unserializable: {type(obj).__name__}>"


# ═══════════════════════════════════════════════════════════════
# Interrupt Extraction
# ═══════════════════════════════════════════════════════════════

def _extract_interrupts(data: Any) -> List[Dict]:
    """Extract __interrupt__ from stream data or state."""
    interrupts = []
    if not isinstance(data, dict):
        return interrupts
    raw = data.get("__interrupt__")
    if raw is None:
        return interrupts
    if isinstance(raw, list):
        for item in raw:
            if hasattr(item, "value"):
                interrupts.append({
                    "id": getattr(item, "id", str(uuid.uuid4())[:16]),
                    "value": _safe_json_serialize(item.value),
                    "ns": getattr(item, "ns", None),
                })
            elif isinstance(item, dict):
                interrupts.append(item)
    elif isinstance(raw, dict):
        interrupts.append(raw)
    return interrupts


def _extract_all_interrupts_from_state(state: Any) -> List[Dict]:
    """Get ALL interrupts from thread state (for when stream misses parallel ones)."""
    interrupts = []

    # 1. LangGraph's ThreadState object stores pending interrupts in the 'interrupts' attribute
    if hasattr(state, 'interrupts') and state.interrupts:
        raw = state.interrupts
        if isinstance(raw, list):
            for item in raw:
                # LangGraph wraps interrupt values in an object with a 'value' attribute
                if hasattr(item, 'value'):
                    interrupts.append({
                        "id": getattr(item, 'id', str(uuid.uuid4())[:16]),
                        "value": _safe_json_serialize(item.value),
                        "ns": getattr(item, 'ns', None),
                    })
                elif isinstance(item, dict):
                    interrupts.append(item)
        elif isinstance(raw, dict):
            interrupts.append(raw)

    # 2. Fallback: check if it's already a dict (e.g. from a serialized state snapshot)
    if not interrupts and isinstance(state, dict):
        for key in ("interrupts", "interrupt", "__interrupt__"):
            if state.get(key) is not None:
                raw = state[key]
                if isinstance(raw, list):
                    for item in raw:
                        if hasattr(item, 'value'):
                            interrupts.append({
                                "id": getattr(item, 'id', str(uuid.uuid4())[:16]),
                                "value": _safe_json_serialize(item.value),
                                "ns": getattr(item, 'ns', None),
                            })
                        elif isinstance(item, dict):
                            interrupts.append(item)
                elif isinstance(raw, dict):
                    interrupts.append(raw)

    return interrupts

# ═══════════════════════════════════════════════════════════════
# Interrupt classification
# ═══════════════════════════════════════════════════════════════

FINAL_FEEDBACK_TYPES = {"final_feedback", "finalizer_feedback"}


def _is_final_feedback_interrupt(interrupt: Dict[str, Any]) -> bool:
    """True if this interrupt came from the finalizer's Y/N prompt."""
    value = interrupt.get("value", interrupt)
    return isinstance(value, dict) and value.get("type") in FINAL_FEEDBACK_TYPES


def _classify_interrupts(interrupts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split interrupts into tool approvals vs finalizer feedback."""
    tools, feedback = [], []
    for itr in interrupts:
        (feedback if _is_final_feedback_interrupt(itr) else tools).append(itr)
    return {"tool_approvals": tools, "final_feedback": feedback}

# ═══════════════════════════════════════════════════════════════
# Health & Threads
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    try:
        await client.threads.search(limit=1)
        return {"status": "healthy", "langgraph_connected": True}
    except Exception as e:
        raise HTTPException(503, f"LangGraph unreachable: {str(e)}")


@app.post("/threads/create")
async def create_thread():
    try:
        thread = await client.threads.create()
        tid = thread["thread_id"]

        # ✅ Assign graph_id immediately so state can be updated before the first run
        await client.threads.update(tid, metadata={"graph_id": "agent"})

        thread_metadata[tid] = {
            "created_at": datetime.utcnow().isoformat(),
            "pending_interrupts": []
        }
        return {"thread_id": tid, "status": "created"}
    except Exception as e:
        raise HTTPException(500, f"Thread creation failed: {str(e)}")


@app.get("/threads/list")
async def list_threads(limit: int = 20, offset: int = 0):
    try:
        threads = await client.threads.search(limit=limit, offset=offset)
        enriched = []
        for t in threads:
            tid = t.get("thread_id")
            enriched.append({**t, "local_meta": thread_metadata.get(tid, {})})
        return {"threads": enriched, "count": len(enriched)}
    except Exception as e:
        raise HTTPException(500, f"Failed to list threads: {str(e)}")


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    try:
        await client.threads.delete(thread_id)
        thread_metadata.pop(thread_id, None)
        return {"thread_id": thread_id, "status": "deleted"}
    except Exception as e:
        raise HTTPException(500, f"Deletion failed: {str(e)}")


@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    try:
        state = await client.threads.get_state(thread_id)
        return {"thread_id": thread_id, "state": _safe_json_serialize(state)}
    except Exception as e:
        raise HTTPException(404, f"State fetch failed: {str(e)}")


@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str, limit: int = 50):
    try:
        history = []
        async for state in client.threads.get_history(thread_id, limit=limit):
            history.append({
                "checkpoint": getattr(state, 'checkpoint', None),
                "values": _safe_json_serialize(state.values if hasattr(state, 'values') else dict(state)),
                "next": getattr(state, 'next', None),
            })
        return {"thread_id": thread_id, "history": history}
    except Exception as e:
        raise HTTPException(500, f"History fetch failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# CRITICAL FIX: Get ALL interrupts from state (not just stream)
# ═══════════════════════════════════════════════════════════════

@app.get("/threads/{thread_id}/interrupts")
async def get_interrupts(thread_id: str):
    """
    Fetch ALL pending interrupts from thread state.
    This catches parallel interrupts that the stream might have missed.
    """
    try:
        state = await client.threads.get_state(thread_id)
        interrupts = _extract_all_interrupts_from_state(state)

        # Also merge with locally cached ones
        local_pending = thread_metadata.get(thread_id, {}).get("pending_interrupts", [])

        # Merge by ID to avoid duplicates
        all_ids = {i.get("id", str(i)) for i in interrupts}
        for local in local_pending:
            if local.get("id", str(local)) not in all_ids:
                interrupts.append(local)

        return {"thread_id": thread_id, "interrupts": interrupts, "count": len(interrupts)}
    except Exception as e:
        raise HTTPException(500, f"Failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# File Upload
# ═══════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_file(thread_id: str = Form(...), file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(400, "No filename provided")

        thread_dir = UPLOAD_DIR / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename).name  # Sanitize filename
        file_path = thread_dir / safe_name

        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Uploaded file is empty")

        # 1. Save physical file to disk
        with open(file_path, "wb") as f:
            f.write(contents)

        file_b64 = base64.b64encode(contents).decode()
        new_file_info = {
            "name": safe_name,
            "type": file.content_type,
            "size": len(contents),
            "path": str(file_path),
            "data": file_b64,
            "uploaded_at": datetime.utcnow().isoformat()
        }

        # Helper function to fetch current files and update state
        async def _update_thread_state():
            try:
                state = await client.threads.get_state(thread_id)
                # Handle both object and dict returns from get_state
                current_files = state.values.get("uploaded_files", []) if hasattr(state, 'values') else state.get("values", {}).get("uploaded_files", [])
            except Exception:
                current_files = []

            # ✅ Append new file instead of overwriting the whole list
            current_files.append(new_file_info)

            # ✅ CRITICAL FIX: Add as_node="agent" to resolve "Ambiguous update"
            await client.threads.update_state(
                thread_id,
                values={"uploaded_files": current_files},
                as_node="agent"  # <--- THIS FIXES THE ERROR
            )

        # 2. Try updating the LangGraph state
        try:
            await _update_thread_state()
        except Exception as state_err:
            err_msg = str(state_err)
            # Catch the specific "no assigned graph ID" error for older threads
            if "no assigned graph ID" in err_msg:
                logger.info(f"Thread {thread_id} missing graph_id. Assigning 'agent' and retrying...")
                try:
                    await client.threads.update(thread_id, metadata={"graph_id": "agent"})
                    await _update_thread_state()
                    logger.info(f"✅ State updated successfully after assigning graph_id.")
                except Exception as retry_err:
                    logger.warning(f"State update failed even after assigning graph_id: {retry_err}")
            else:
                logger.warning(f"LangGraph state update failed (file still saved to disk): {state_err}")

        # 3. Return success to frontend
        return {
            "filename": safe_name,
            "thread_id": thread_id,
            "status": "uploaded",
            "file_type": file.content_type or "unknown",
            "size": len(contents),
            "path": str(file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

# ═══════════════════════════════════════════════════════════════
# Streaming Chat — CORRECT multi-interrupt cycle handling
# ═══════════════════════════════════════════════════════════════

async def chat_stream_generator(tid: str, message: str, max_iter: int):
    """
    FIXED: Handles parallel multi-interrupt cycles.
    When parallel sub-agents all hit interrupts, we:
    1. Stream until first interrupt
    2. PAUSE and tell frontend to collect ALL approvals
    3. Frontend calls /approve/batch with complete map
    4. We resume and continue streaming results
    """
    try:
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': tid})}\n\n"

        config = {"configurable": {"thread_id": tid}}
        collected_interrupts = []

        stream = client.runs.stream(
            tid, "agent",
            input={
                "messages": [{"role": "user", "content": message}],
                "thread_id": tid,
                "max_iterations": max_iter,
                "iteration_count": 0,
                "carry_on": True,
                "current_task": "",
                "final_response": "",
                "tasks": [],
            },
            config=config,
            stream_mode=["updates", "custom"]
        )

        async for chunk in stream:
            safe_data = _safe_json_serialize(chunk.data)

            # INTERRUPT DETECTION
            if chunk.event == "updates" and isinstance(chunk.data, dict):
                interrupts = _extract_interrupts(chunk.data)
                if interrupts:
                    collected_interrupts.extend(interrupts)
                    thread_metadata.setdefault(tid, {})["pending_interrupts"] = collected_interrupts

                    # CRITICAL: Instead of yielding immediately and stopping,
                    # we continue briefly to see if more parallel interrupts arrive
                    # But the stream will actually stop here, so we yield what we have
                    yield f"data: {json.dumps({'type': 'interrupt', 'payload': interrupts, 'note': 'There may be more parallel interrupts. Check /interrupts endpoint.'})}\n\n"
                    continue

            # CUSTOM EVENTS
            if chunk.event == "custom":
                yield f"data: {json.dumps({'type': 'custom', 'data': safe_data})}\n\n"
                continue

            # REGULAR UPDATES
            yield f"data: {json.dumps({'type': 'chunk', 'event': chunk.event, 'data': safe_data})}\n\n"

        # After stream ends, check state for ANY remaining interrupts
        # This catches parallel interrupts that didn't come through the stream
        try:
            state = await client.threads.get_state(tid)
            remaining = _extract_all_interrupts_from_state(state)
            if remaining:
                # Merge with collected, avoiding duplicates
                existing_ids = {i.get("id", str(i)) for i in collected_interrupts}
                for r in remaining:
                    if r.get("id", str(r)) not in existing_ids:
                        collected_interrupts.append(r)

                thread_metadata.setdefault(tid, {})["pending_interrupts"] = collected_interrupts
                yield f"data: {json.dumps({'type': 'interrupt', 'payload': remaining, 'note': 'Additional parallel interrupts found in state'})}\n\n"
        except Exception as e:
            logger.warning(f"Post-stream state check failed: {e}")

        # End of stream
        if collected_interrupts:
            yield f"data: {json.dumps({'type': 'awaiting_approval', 'thread_id': tid, 'interrupts': collected_interrupts, 'count': len(collected_interrupts)})}\n\n"
        else:
            try:
                state = await client.threads.get_state(tid)
                yield f"data: {json.dumps({'type': 'final_state', 'state': _safe_json_serialize(state)})}\n\n"
            except Exception as e:
                logger.warning(f"Final state fetch failed: {e}")
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    tid = req.thread_id or (await client.threads.create())["thread_id"]
    return StreamingResponse(
        chat_stream_generator(tid, req.message, req.max_iterations),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Thread-Id": tid}
    )

@app.get("/threads/{thread_id}/interrupts")
async def get_interrupts(thread_id: str):
    try:
        state = await client.threads.get_state(thread_id)
        interrupts = _extract_all_interrupts_from_state(state)

        local_pending = thread_metadata.get(thread_id, {}).get("pending_interrupts", [])
        all_ids = {i.get("id", str(i)) for i in interrupts}
        for local in local_pending:
            if local.get("id", str(local)) not in all_ids:
                interrupts.append(local)

        return {
            "thread_id": thread_id,
            "interrupts": interrupts,
            "count": len(interrupts),
            "classified": _classify_interrupts(interrupts),   # ← new
        }
    except Exception as e:
        raise HTTPException(500, f"Failed: {str(e)}")
# ═══════════════════════════════════════════════════════════════
# Resume — Handles multi-interrupt cycles with state polling
# ═══════════════════════════════════════════════════════════════

def _build_resume_map(approvals: List[SingleToolApproval]) -> Dict[str, Any]:
    resume_map = {}
    for approval in approvals:
        if approval.decision == "Y":
            resume_map[approval.interrupt_id] = {"status": "Y"}
        elif approval.decision == "N":
            resume_map[approval.interrupt_id] = {
                "status": "N",
                "feedback": approval.feedback or "User rejected",
            }
        elif approval.decision == "E":
            resume_map[approval.interrupt_id] = {
                "status": "E",
                "tool_args": approval.tool_args or {},
            }
    return resume_map

async def resume_stream_generator(tid: str, approvals: List[SingleToolApproval]):
    try:
        resume_map = _build_resume_map(approvals)
        approved_ids = set(resume_map.keys())          # ← NEW
        config = {"configurable": {"thread_id": tid}}
        command = {"resume": resume_map}

        yield f"data: {json.dumps({'type': 'resume_started', 'resume_map': resume_map})}\n\n"

        stream = client.runs.stream(
            tid, "agent",
            command=command,
            config=config,
            stream_mode=["updates", "custom"],
        )

        async for chunk in stream:
            safe_data = _safe_json_serialize(chunk.data)

            if chunk.event == "updates" and isinstance(chunk.data, dict):
                interrupts = _extract_interrupts(chunk.data)
                if interrupts:
                    yield f"data: {json.dumps({'type': 'interrupt', 'payload': interrupts})}\n\n"
                    continue

            if chunk.event == "custom":
                yield f"data: {json.dumps({'type': 'custom', 'data': safe_data})}\n\n"
                continue

            yield f"data: {json.dumps({'type': 'chunk', 'event': chunk.event, 'data': safe_data})}\n\n"

        # Poll for genuinely NEW interrupts only
        for poll in range(5):
            await asyncio.sleep(0.5)
            try:
                state = await client.threads.get_state(tid)
                remaining_all = _extract_all_interrupts_from_state(state)
                remaining = [
                    r for r in remaining_all
                    if r.get("id") not in approved_ids          # ← NEW
                ]
                if not remaining:
                    break

                thread_metadata.setdefault(tid, {})["pending_interrupts"] = remaining
                yield f"data: {json.dumps({'type': 'interrupt', 'payload': remaining, 'cycle': poll + 1})}\n\n"
                yield f"data: {json.dumps({'type': 'awaiting_approval', 'thread_id': tid, 'interrupts': remaining, 'count': len(remaining), 'cycle': poll + 1})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"          # ← NEW
                return
            except Exception as e:
                logger.warning(f"Poll {poll} failed: {e}")
                break

        try:
            state = await client.threads.get_state(tid)
            yield f"data: {json.dumps({'type': 'final_state', 'state': _safe_json_serialize(state)})}\n\n"
        except Exception as e:
            logger.warning(f"Final state fetch failed: {e}")
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        logger.error(f"Resume error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/resume/stream")
async def resume_stream(req: BatchApprovalRequest):
    return StreamingResponse(
        resume_stream_generator(req.thread_id, req.approvals),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Thread-Id": req.thread_id}
    )


@app.post("/approve/batch")
async def approve_batch(req: BatchApprovalRequest):
    try:
        return StreamingResponse(
            resume_stream_generator(req.thread_id, req.approvals),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Thread-Id": req.thread_id}
        )
    except Exception as e:
        raise HTTPException(500, f"Batch approval failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload=True)
