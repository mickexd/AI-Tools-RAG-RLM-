"""
RLM REPL Server — MCP tools for the Recursive Language Model paradigm.

Architecture: Claude (root LLM) orchestrates via MCP tool calls.
Sub-LLM calls go to OpenRouter (Gemini 3 Flash) with LM Studio fallback (Qwen3 14B).

Based on "Recursive Language Models" (Zhang, Kraska, Khattab — arXiv:2512.24601).

The key insight: Claude never sees the raw context. It lives as a REPL variable.
Claude writes Python to inspect/chunk/process it, delegating heavy lifting to sub-LLMs.
Each REPL result returns only metadata (variable names, truncated stdout) to keep
Claude's context window clean — enabling dozens of iterations instead of 1-2.

Mac-compatible version with proper path resolution for macOS.
"""

import io
import json
import os
import re
import sys
import shutil
import tempfile
import threading
import time
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from mcp.server.fastmcp import FastMCP


# ─── Configuration ────────────────────────────────────────────────────────────

class Config:
    # Primary sub-LLM: OpenRouter → Gemini 2.5 Flash
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR API HERE")
    OPENROUTER_MODEL = os.environ.get("RLM_SUB_MODEL", "google/gemini-3-flash-preview")

    # Fallback sub-LLM: LM Studio → Qwen3 14B (local)
    LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-14b")

    # Sub-LLM parameters
    SUB_LLM_TEMPERATURE = 0.3
    SUB_LLM_MAX_TOKENS = 16384
    SUB_LLM_TIMEOUT = 120

    # REPL output limits — Algorithm 1 requires bounded turn size
    # "if we trim each turn to c tokens, we will have at most K/c root iterations"
    STDOUT_PREVIEW_CHARS = 500
    STDERR_PREVIEW_CHARS = 300
    MAX_BATCH_WORKERS = 8

    # File reading
    DEFAULT_ENCODING = "utf-8"
    FALLBACK_ENCODING = "latin-1"


# ─── Sub-LLM Client ──────────────────────────────────────────────────────────
# Two-tier: OpenRouter (primary) → LM Studio (fallback)
# No caching — each sub-call processes unique context (paper design).

class SubLLMClient:
    """
    Manages sub-LLM calls with automatic failover.
    Primary: OpenRouter API (Gemini 2.5 Flash — 1M context, fast, cheap).
    Fallback: LM Studio local server (Qwen3 14B — no API cost, offline capable).
    """

    def __init__(self):
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=2
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._openrouter_ok: Optional[bool] = None
        self._lmstudio_ok: Optional[bool] = None
        self._lock = threading.Lock()
        self._call_count = 0
        self._total_input_chars = 0
        self._total_output_chars = 0

    def _check_openrouter(self) -> bool:
        if not Config.OPENROUTER_API_KEY:
            return False
        try:
            resp = self._session.get(
                f"{Config.OPENROUTER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {Config.OPENROUTER_API_KEY}"},
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _check_lmstudio(self) -> bool:
        try:
            resp = self._session.get(
                f"{Config.LMSTUDIO_BASE_URL}/models", timeout=3
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def health_check(self) -> dict:
        """Check availability of both backends."""
        self._openrouter_ok = self._check_openrouter()
        self._lmstudio_ok = self._check_lmstudio()
        return {
            "openrouter": {
                "available": self._openrouter_ok,
                "model": Config.OPENROUTER_MODEL,
            },
            "lmstudio": {
                "available": self._lmstudio_ok,
                "model": Config.LMSTUDIO_MODEL,
                "url": Config.LMSTUDIO_BASE_URL,
            },
            "active_backend": self._resolve_backend_name(),
        }

    def _resolve_backend_name(self) -> str:
        if self._openrouter_ok is None:
            self._openrouter_ok = self._check_openrouter()
        if self._openrouter_ok:
            return "openrouter"
        if self._lmstudio_ok is None:
            self._lmstudio_ok = self._check_lmstudio()
        if self._lmstudio_ok:
            return "lmstudio"
        return "none"

    def _call_openrouter(self, prompt: str) -> str:
        resp = self._session.post(
            f"{Config.OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": Config.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": Config.SUB_LLM_TEMPERATURE,
                "max_tokens": Config.SUB_LLM_MAX_TOKENS,
            },
            timeout=Config.SUB_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_lmstudio(self, prompt: str) -> str:
        resp = self._session.post(
            f"{Config.LMSTUDIO_BASE_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": Config.LMSTUDIO_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": Config.SUB_LLM_TEMPERATURE,
                "max_tokens": Config.SUB_LLM_MAX_TOKENS,
            },
            timeout=Config.SUB_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def query(self, prompt: str) -> str:
        """
        Single sub-LLM call with automatic failover.
        Returns the response text or an error string.
        """
        # Try OpenRouter first
        if self._openrouter_ok is None:
            self._openrouter_ok = self._check_openrouter()

        if self._openrouter_ok:
            try:
                result = self._call_openrouter(prompt)
                self._track(prompt, result)
                return result
            except Exception as e:
                self._openrouter_ok = False  # Mark down, try fallback

        # Fallback to LM Studio
        if self._lmstudio_ok is None:
            self._lmstudio_ok = self._check_lmstudio()

        if self._lmstudio_ok:
            try:
                result = self._call_lmstudio(prompt)
                self._track(prompt, result)
                return result
            except Exception as e:
                self._lmstudio_ok = False
                return f"[SUB-LLM ERROR] Both backends failed. LM Studio: {e}"

        return "[SUB-LLM ERROR] No backends available. Set OPENROUTER_API_KEY or start LM Studio."

    def query_batch(self, prompts: list[str]) -> list[str]:
        """
        Parallel sub-LLM calls. Respects MAX_BATCH_WORKERS concurrency limit.
        """
        if not prompts:
            return []

        workers = min(len(prompts), Config.MAX_BATCH_WORKERS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.query, p) for p in prompts]
            return [f.result() for f in futures]

    def _track(self, prompt: str, result: str):
        with self._lock:
            self._call_count += 1
            self._total_input_chars += len(prompt)
            self._total_output_chars += len(result)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": self._call_count,
                "input_chars": self._total_input_chars,
                "output_chars": self._total_output_chars,
            }

    def reset_stats(self):
        with self._lock:
            self._call_count = 0
            self._total_input_chars = 0
            self._total_output_chars = 0

    def close(self):
        self._session.close()


# ─── REPL Environment ─────────────────────────────────────────────────────────
# Sandboxed Python execution with proper namespace handling.
# Follows reference implementation: imports in globals, auto-print last expression.

_SAFE_BUILTINS = {
    # Core types
    "int": int, "float": float, "str": str, "bool": bool, "bytes": bytes,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "complex": complex,
    "bytearray": bytearray, "frozenset": frozenset, "type": type, "object": object,
    # Iteration & functional
    "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "iter": iter, "next": next,
    "any": any, "all": all, "slice": slice,
    # Math
    "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "pow": pow, "divmod": divmod,
    # String & repr
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "format": format, "repr": repr, "ascii": ascii, "hash": hash,
    # Attribute access
    "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "dir": dir, "vars": vars, "callable": callable, "id": id,
    # I/O
    "print": print, "open": open, "input": None,
    # Imports allowed
    "__import__": __import__,
    # Common exceptions
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ImportError": ImportError,
    "OSError": OSError, "IOError": IOError, "NameError": NameError,
    "NotImplementedError": NotImplementedError, "ZeroDivisionError": ZeroDivisionError,
    # Block dangerous builtins
    "eval": None, "exec": None, "compile": None, "globals": None, "locals": None,
}


class REPLEnvironment:
    """
    Sandboxed Python REPL that persists across tool calls within a session.

    Key design choices from the paper:
    - Imports execute in globals (available to all subsequent code)
    - Auto-prints the last expression (like Jupyter/IPython)
    - Returns metadata only: variable names + truncated stdout
    """

    def __init__(self, sub_llm: SubLLMClient):
        self._sub_llm = sub_llm
        self._lock = threading.Lock()
        self._temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")

        # Separate namespaces — avoids the globals/locals collision bug
        self._globals = {"__builtins__": _SAFE_BUILTINS.copy(), "__name__": "__repl__"}
        self._locals = {}

        # Inject RLM functions
        self._globals["llm_query"] = self._llm_query
        self._globals["llm_query_batched"] = self._llm_query_batched

    # ── Context loading ──

    def load_string(self, content: str, var_name: str = "context"):
        """Load a string into the REPL as a variable."""
        if len(content) > 100_000:
            # Large content: write to temp file, read back in REPL
            path = os.path.join(self._temp_dir, f"{var_name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.execute(
                f"with open(r'{path}', 'r', encoding='utf-8') as _f:\n"
                f"    {var_name} = _f.read()"
            )
        else:
            self._locals[var_name] = content

    def load_json(self, data: Any, var_name: str = "context"):
        """Load JSON-serializable data into the REPL."""
        serialized = json.dumps(data)
        if len(serialized) > 100_000:
            path = os.path.join(self._temp_dir, f"{var_name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            self.execute(
                f"import json\n"
                f"with open(r'{path}', 'r') as _f:\n"
                f"    {var_name} = json.load(_f)"
            )
        else:
            self._locals[var_name] = data

    # ── Code execution ──

    def execute(self, code: str) -> dict:
        """
        Execute Python code and return metadata-only result.

        Returns dict with: stdout_preview, stderr, variables, execution_time.
        Per Algorithm 1: "this forces M to rely on variables and sub-calls to manage
        long strings instead of polluting its window."
        """
        start = time.perf_counter()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with self._lock:
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf

                lines = code.split("\n")
                import_lines = []
                other_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")) and not stripped.startswith("#"):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)

                # Imports go to globals (available in all future executions)
                if import_lines:
                    exec("\n".join(import_lines), self._globals)

                if other_lines:
                    other_code = "\n".join(other_lines)
                    combined = {**self._globals, **self._locals}

                    # Auto-print last expression (like Jupyter)
                    non_empty = [l for l in other_lines if l.strip() and not l.strip().startswith("#")]
                    did_auto_print = False

                    if non_empty:
                        last = non_empty[-1].strip()
                        is_expr = (
                            not last.startswith(("import ", "from ", "def ", "class ", "if ",
                                                 "for ", "while ", "try:", "with ", "return ",
                                                 "yield ", "break", "continue", "pass", "raise ",
                                                 "del ", "assert ", "elif ", "else:", "except",
                                                 "finally:", "print("))
                            and "=" not in last.split("#")[0]
                            and not last.endswith(":")
                        )
                        if is_expr:
                            try:
                                # Execute everything except last line as statements
                                last_idx = None
                                for i in range(len(other_lines) - 1, -1, -1):
                                    if other_lines[i].strip() == last:
                                        last_idx = i
                                        break
                                if last_idx and last_idx > 0:
                                    exec("\n".join(other_lines[:last_idx]), combined, combined)
                                val = eval(last, combined, combined)
                                if val is not None:
                                    print(repr(val))
                                did_auto_print = True
                            except Exception:
                                did_auto_print = False

                    if not did_auto_print:
                        exec(other_code, combined, combined)

                    # Update locals with new/changed variables
                    for k, v in combined.items():
                        if k not in self._globals and not k.startswith("_"):
                            self._locals[k] = v

            except Exception as e:
                print(f"{type(e).__name__}: {e}", file=sys.stderr)
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        elapsed = time.perf_counter() - start
        raw_stdout = stdout_buf.getvalue()
        raw_stderr = stderr_buf.getvalue()

        # Build metadata-only response (Algorithm 1)
        var_info = {}
        for k, v in self._locals.items():
            if k.startswith("_"):
                continue
            try:
                vtype = type(v).__name__
                if isinstance(v, str):
                    vlen = f"{len(v):,} chars"
                elif isinstance(v, (list, dict, tuple, set)):
                    vlen = f"{len(v):,} items"
                else:
                    vlen = None
                var_info[k] = {"type": vtype, "size": vlen} if vlen else {"type": vtype}
            except Exception:
                var_info[k] = {"type": "unknown"}

        return {
            "stdout": _truncate(raw_stdout, Config.STDOUT_PREVIEW_CHARS),
            "stdout_full_length": len(raw_stdout),
            "stderr": _truncate(raw_stderr, Config.STDERR_PREVIEW_CHARS) if raw_stderr else None,
            "variables": var_info,
            "execution_time_ms": round(elapsed * 1000, 1),
        }

    # ── Sub-LLM bridge ──

    def _llm_query(self, prompt: str) -> str:
        """Available inside REPL as llm_query(prompt)."""
        return self._sub_llm.query(prompt)

    def _llm_query_batched(self, prompts: list) -> list:
        """Available inside REPL as llm_query_batched([prompts])."""
        return self._sub_llm.query_batch(prompts)

    # ── Variable access ──

    def get_var(self, name: str) -> Any:
        name = name.strip().strip("'\"")
        if name in self._locals:
            return self._locals[name]
        raise KeyError(f"Variable '{name}' not found. Available: {self.var_names}")

    @property
    def var_names(self) -> list[str]:
        return [k for k in self._locals if not k.startswith("_")]

    # ── Cleanup ──

    def destroy(self):
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass


# ─── Session Manager ──────────────────────────────────────────────────────────
# Single active session. Resets when a new context is loaded.

class Session:
    """
    Holds the active REPL environment and sub-LLM client.
    One session at a time — loading new context resets it.
    """

    def __init__(self):
        self.sub_llm = SubLLMClient()
        self.repl: Optional[REPLEnvironment] = None
        self.context_info: Optional[dict] = None
        self._created_at: Optional[float] = None

    def init_repl(self, context: Any, context_type: str = "text") -> dict:
        """Initialize or reset the REPL with new context."""
        if self.repl:
            self.repl.destroy()
        self.sub_llm.reset_stats()

        self.repl = REPLEnvironment(self.sub_llm)
        self._created_at = time.time()

        if isinstance(context, str):
            self.repl.load_string(context)
            size_desc = f"{len(context):,} chars"
        elif isinstance(context, (dict, list)):
            self.repl.load_json(context)
            size_desc = f"{len(json.dumps(context)):,} chars (serialized)"
        else:
            self.repl.load_string(str(context))
            size_desc = f"{len(str(context)):,} chars"

        self.context_info = {
            "context_type": context_type,
            "size": size_desc,
            "variables": self.repl.var_names,
        }
        return self.context_info

    @property
    def is_active(self) -> bool:
        return self.repl is not None

    @property
    def status(self) -> dict:
        if not self.is_active:
            return {"active": False}
        return {
            "active": True,
            "context": self.context_info,
            "variables": self.repl.var_names if self.repl else [],
            "sub_llm_stats": self.sub_llm.stats,
            "uptime_seconds": round(time.time() - self._created_at, 1) if self._created_at else 0,
        }


# ─── Utilities ────────────────────────────────────────────────────────────────

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit:,} more chars — use print(var) or slicing to inspect]"


def _read_file(path: str) -> tuple[str, str]:
    """Read file content with encoding fallback. Returns (content, detected_encoding)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")

    for enc in (Config.DEFAULT_ENCODING, Config.FALLBACK_ENCODING):
        try:
            return p.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue
    raise IOError(f"Cannot decode file: {path}")


def _resolve_path(filename: str) -> Path:
    """Resolve file path with cross-platform search (macOS-optimized)."""
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p

    # macOS-specific search directories
    search_dirs = [
        Path.cwd(),
        Path.home(),
        Path("/tmp"),
        Path.home() / "Documents",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
    ]

    for d in search_dirs:
        candidate = d / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"'{filename}' not found. Searched: {[str(d) for d in search_dirs if d.exists()]}"
    )


# ─── Singleton ────────────────────────────────────────────────────────────────

_session = Session()


# ─── MCP Server & Tools ──────────────────────────────────────────────────────

mcp = FastMCP("rlm-repl-server")


@mcp.tool()
def load_context(
    source: str,
    source_type: str = "file",
    context_label: str = "text",
) -> dict:
    """
    Initialize the RLM REPL session with context data.

    This loads context into the REPL as a `context` variable that you (Claude) can
    inspect and process via repl_execute(). The raw content is NEVER returned to you —
    you must write Python code to interact with it.

    Args:
        source: File path (if source_type="file") or raw text content (if source_type="inline").
        source_type: "file" to read from disk, "inline" to use the source string directly.
        context_label: Description of what this context is (e.g. "research paper", "chat logs").

    Returns:
        Session info with context metadata (type, size, available variables).
    """
    if source_type == "file":
        path = _resolve_path(source)
        content, encoding = _read_file(str(path))
        info = _session.init_repl(content, context_label)
        info["file"] = str(path)
        info["encoding"] = encoding
        return info
    elif source_type == "inline":
        return _session.init_repl(source, context_label)
    else:
        return {"error": f"Unknown source_type: {source_type}. Use 'file' or 'inline'."}


@mcp.tool()
def repl_execute(code: str) -> dict:
    """
    Execute Python code in the RLM REPL environment.

    The REPL has a persistent `context` variable (loaded via load_context) and any
    variables you create persist across calls. You also have access to:
      - llm_query(prompt) → str: Call the sub-LLM with a prompt.
      - llm_query_batched([prompts]) → [str]: Parallel sub-LLM calls.

    Returns metadata only (variable names, truncated stdout) to keep your context
    window clean. Write `print(var[:500])` to inspect specific values.

    Strategy guidance (from the paper):
      1. First inspect: print(len(context)), print(context[:500])
      2. Decide chunking: by size, by delimiter, by structure
      3. Process chunks via llm_query or llm_query_batched
      4. Aggregate results in REPL variables
      5. You (Claude) provide the final answer directly in conversation — no FINAL() needed.

    Args:
        code: Python code to execute. Wrap in a single string, not in code fences.

    Returns:
        Dict with stdout (truncated), stderr, variables (names + types + sizes),
        and execution time. Full stdout length is included so you know if truncation occurred.
    """
    if not _session.is_active:
        return {"error": "No active session. Call load_context() first."}
    return _session.repl.execute(code)


@mcp.tool()
def sub_llm_query(prompt: str) -> str:
    """
    Call the sub-LLM directly (outside the REPL).

    Use this when you want a quick sub-LLM response without writing REPL code.
    For processing context chunks, prefer using llm_query() inside repl_execute()
    so results stay in REPL variables.

    The sub-LLM is Gemini 2.5 Flash (1M token context) with Qwen3 14B fallback.

    Args:
        prompt: The prompt to send to the sub-LLM.

    Returns:
        The sub-LLM's response text.
    """
    return _session.sub_llm.query(prompt)


@mcp.tool()
def sub_llm_batch(prompts: list[str]) -> list[str]:
    """
    Parallel sub-LLM calls (outside the REPL).

    Processes up to 8 prompts concurrently. For batch processing inside the REPL,
    use llm_query_batched([prompts]) instead.

    Args:
        prompts: List of prompts to send in parallel.

    Returns:
        List of response strings in the same order.
    """
    return _session.sub_llm.query_batch(prompts)


@mcp.tool()
def get_variable(name: str, max_chars: int = 5000) -> str:
    """
    Retrieve the string value of a REPL variable.

    Use this to pull a computed result out of the REPL when you need to see it
    in full (up to max_chars). For very large variables, use repl_execute with
    slicing instead: print(var[:1000])

    Args:
        name: Variable name in the REPL.
        max_chars: Maximum characters to return (default 5000).

    Returns:
        String representation of the variable, truncated if necessary.
    """
    if not _session.is_active:
        return "Error: No active session."
    try:
        val = str(_session.repl.get_var(name))
        if len(val) > max_chars:
            return val[:max_chars] + f"\n... [{len(val) - max_chars:,} more chars]"
        return val
    except KeyError as e:
        return str(e)


@mcp.tool()
def session_status() -> dict:
    """
    Check the current session state, sub-LLM availability, and usage stats.

    Returns:
        Session info including active state, context metadata, variables,
        sub-LLM backend health, and call statistics.
    """
    status = _session.status
    status["sub_llm_health"] = _session.sub_llm.health_check()
    return status


@mcp.tool()
def reset_session() -> dict:
    """
    Destroy the current REPL session and release resources.

    Returns:
        Confirmation with final session stats.
    """
    stats = _session.status
    if _session.repl:
        _session.repl.destroy()
        _session.repl = None
        _session.context_info = None
    return {"reset": True, "final_stats": stats}


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
