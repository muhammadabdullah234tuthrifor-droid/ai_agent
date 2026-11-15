"""
Playwright sandbox client + AST-based code validator.

This module provides a secure way to execute user-supplied Playwright code
inside an isolated Docker container. The AST validator blocks dangerous
constructs before the code ever reaches the sandbox.
"""

import ast
import asyncio
import json
import os


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
                self.fail(node, f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        if module not in ALLOWED_IMPORTS:
            self.fail(node, f"Import not allowed: {module}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in BLOCKED_NAMES:
            self.fail(node, f"Name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            self.fail(node, f"Private attribute not allowed: {node.attr}")
        if node.attr in BLOCKED_ATTRIBUTES:
            self.fail(node, f"Attribute not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.fail(node, "Lambda expressions are not allowed")

    def visit_ClassDef(self, node):
        self.fail(node, "Class definitions are not allowed")

    def visit_FunctionDef(self, node):
        self.fail(node, "Function definitions are not allowed")

    def visit_AsyncFunctionDef(self, node):
        self.fail(node, "Function definitions are not allowed")

    def visit_While(self, node):
        self.fail(node, "while loops are not allowed")

    def visit_ImportStar(self, node):
        self.fail(node, "Wildcard imports are not allowed")


def validate_code(code: str) -> None:
    if len(code) > 20_000:
        raise ValueError("Code exceeds maximum length of 20,000 characters.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    validator = PlaywrightCodeValidator()
    validator.visit(tree)

    if validator.errors:
        raise ValueError("Code rejected:\n" + "\n".join(validator.errors))


def serialize_request(code: str) -> bytes:
    payload = {"code": code}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def execute_in_sandbox(code: str, timeout: int = 90) -> dict:
    """Run validated Playwright code inside a disposable Docker container."""
    validate_code(code)
    payload = serialize_request(code)

    process = await asyncio.create_subprocess_exec(
        "docker",
        "run",
        "--rm",
        "-i",
        "-p", "6080:6080",
        "--security-opt", "no-new-privileges:true",
        "--cap-drop", "ALL",
        "--memory", "1g",
        "--cpus", "2",
        "--pids-limit", "128",
        "--shm-size", "256m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        "--network", "bridge",
        "-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
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
            "message": f"Execution exceeded {timeout} seconds.",
        }

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

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


async def run_playwright(code: str, timeout: int = 120) -> dict:
    """
    Execute Python Playwright code inside a sandboxed browser environment.

    The code has access to pre-instantiated globals:
      - page, context, browser, playwright, asyncio

    Constraints:
      - Must return a JSON-serializable value
      - No filesystem, subprocess, or shell access
      - All web interactions go through the provided page/context
    """
    timeout = max(1, min(timeout, 120))
    return await execute_in_sandbox(code=code, timeout=timeout)
