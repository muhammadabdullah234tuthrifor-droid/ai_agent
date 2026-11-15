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
SLOW_MO         = int(os.getenv("SLOW_MO", "400"))
VIEWPORT        = {"width": 1920, "height": 1080}

_state: dict = {"pw": None, "browser": None, "lock": asyncio.Lock()}


# ---------------------------------------------------------------- lifecycle
async def _launch_browser():
    if _state["browser"] is None:
        if _state["pw"] is None:
            _state["pw"] = await async_playwright().start()
        _state["browser"] = await _state["pw"].chromium.launch(
            headless=False,                      # needs DISPLAY -> Xvfb :99
            slow_mo=SLOW_MO,
            args=[
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                # add "--no-sandbox" only if you run the container as root
            ],
        )
    return _state["browser"]


async def _hard_reset_browser():
    """Kill Chromium completely — used after timeouts / crashes."""
    b = _state.get("browser")
    _state["browser"] = None
    if b is not None:
        try:
            await b.close()
        except Exception:
            pass


async def _close_extra_contexts(browser):
    """Per-request isolation: close anything the user spawned."""
    for ctx in list(browser.contexts):
        try:
            await ctx.close()
        except Exception:
            pass


# ---------------------------------------------------------------- exec core
async def _run_user_code(code: str):
    browser = await _launch_browser()
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
        result = await namespace["__user_main__"]()
    finally:
        try:
            await context.close()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app):
    yield
    await _hard_reset_browser()
    if _state["pw"]:
        await _state["pw"].stop()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/execute")
async def execute(request: Request):
    try:
        payload = await request.json()
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

    # One browser instance → serialize requests.
    async with _state["lock"]:
        try:
            result = await asyncio.wait_for(_run_user_code(code), timeout=timeout)
            output = {"ok": True, "result": result}

        except asyncio.TimeoutError:
            await _hard_reset_browser()
            output = {"ok": False, "error": "Timeout",
                      "message": f"Exceeded {timeout}s"}

        except Exception as exc:
            output = {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            b = _state.get("browser")
            if b is not None:
                await _close_extra_contexts(b)

    serialized = json.dumps(output, default=str)
    if len(serialized) > MAX_OUTPUT_SIZE:
        serialized = json.dumps({
            "ok": False,
            "error": "OutputTooLarge",
            "message": f"Output exceeded {MAX_OUTPUT_SIZE} bytes.",
        })

    return Response(content=serialized, media_type="application/json")
