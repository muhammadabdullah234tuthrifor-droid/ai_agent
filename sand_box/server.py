# sand_box/server.py
import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from playwright.async_api import async_playwright

MAX_OUTPUT_SIZE = 100_000
SLOW_MO  = int(os.getenv("SLOW_MO", "400"))
VIEWPORT = {"width": 1920, "height": 1080}

_state = {"pw": None, "browser": None, "lock": asyncio.Lock()}


# ---------------------------------------------------------------- browser
async def _ensure_browser():
    b = _state["browser"]
    if b is not None and b.is_connected():
        return b
    if _state["pw"] is None:
        _state["pw"] = await async_playwright().start()
    _state["browser"] = await _state["pw"].chromium.launch(
        headless=False,                 # needs DISPLAY=:99 (set by start.sh)
        slow_mo=SLOW_MO,
        args=[
            "--disable-dev-shm-usage",
            "--no-first-run",
            "--no-default-browser-check",
        ],
    )
    return _state["browser"]


async def _reset_browser():
    """Kill Chromium entirely — used after timeouts."""
    b = _state["browser"]
    _state["browser"] = None
    if b is not None:
        try:
            await b.close()
        except Exception:
            pass


# ---------------------------------------------------------------- exec
async def _run_user_code(code: str):
    browser = await _ensure_browser()
    context = await browser.new_context(viewport=VIEWPORT)
    page    = await context.new_page()

    namespace = {
        "page": page,
        "context": context,
        "browser": browser,
        "playwright": _state["pw"],
        "asyncio": asyncio,
    }

    function_source = (
        "async def __user_main__():\n"
        + "\n".join("    " + line for line in code.splitlines())
    )
    exec(function_source, namespace)

    try:
        return await namespace["__user_main__"]()
    finally:
        # close every context — including ones the user spawned and forgot
        for ctx in list(browser.contexts):
            try:
                await ctx.close()
            except Exception:
                pass


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app):
    yield
    await _reset_browser()
    if _state["pw"]:
        await _state["pw"].stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/execute")
async def execute(req: Request):
    try:
        payload = await req.json()
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": "JSONDecodeError", "message": str(exc)},
            status_code=400,
        )

    code    = payload.get("code")
    timeout = payload.get("timeout", 120)

    if not isinstance(code, str):
        return JSONResponse(
            {"ok": False, "error": "InvalidCode",
             "message": "Payload must contain string field 'code'."},
            status_code=400,
        )

    # one shared Chromium → serialize callers
    async with _state["lock"]:
        try:
            result = await asyncio.wait_for(_run_user_code(code), timeout=timeout)
            out = {"ok": True, "result": result}
        except asyncio.TimeoutError:
            await _reset_browser()
            out = {"ok": False, "error": "Timeout",
                   "message": f"Exceeded {timeout}s"}
        except Exception as exc:
            out = {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

    serialized = json.dumps(out, default=str)
    if len(serialized) > MAX_OUTPUT_SIZE:
        serialized = json.dumps({
            "ok": False,
            "error": "OutputTooLarge",
            "message": f"Output exceeded {MAX_OUTPUT_SIZE} bytes.",
        })

    return Response(content=serialized, media_type="application/json")
