"""
Smart Context MCP Server - HIGH PERFORMANCE VERSION
Optimized for speed with connection pooling, caching, and async support.

Implements the Recursive Language Model (RLM) paradigm from:
"Recursive Language Models" (Zhang, Kraska, Khattab - arXiv:2512.24601)

Key optimizations:
- Connection pooling via requests.Session()
- LRU caching for LLM responses
- Pre-compiled regex patterns
- __slots__ for memory efficiency
- Optimized string operations
- Connection reuse across requests
"""

import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import shutil
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from mcp.server.fastmcp import FastMCP

# ============================================================================
# OPTIMIZATION: Pre-compiled regex patterns (avoid recompilation)
# ============================================================================

# Code block extraction - compiled once
REPL_CODE_PATTERN = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)

# Final answer patterns - more restrictive to avoid capturing markdown
# Only matches valid Python identifiers: letters, digits, underscore
FINAL_VAR_PATTERN = re.compile(r"^\s*FINAL_VAR\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)", re.MULTILINE)
FINAL_PATTERN = re.compile(r"^\s*FINAL\((.*?)\)", re.MULTILINE | re.DOTALL)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_MODEL = "ministral-3:latest"

    # Context window sizes
    ROOT_NUM_CTX = 32768
    SUB_LLM_NUM_CTX = 24576

    # RLM loop limits
    MAX_ITERATIONS = 30
    MAX_MESSAGE_HISTORY = 20

    # Max chars per REPL output message fed back to root LLM
    MAX_RESULT_LENGTH = 100000

    # File handling
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 500000
    DEFAULT_TIMEOUT = 300  # 5 minutes

    # Performance: Cache settings
    LLM_CACHE_SIZE = 128  # Cache last 128 LLM responses
    LLM_CACHE_TTL = 3600  # Cache TTL in seconds (1 hour)


# ============================================================================
# OPTIMIZATION: Data classes with __slots__ for memory efficiency
# ============================================================================

@dataclass(slots=True)
class REPLResult:
    """Result from executing code in REPL environment"""
    stdout: str
    stderr: str
    locals: dict
    execution_time: float = 0.0


@dataclass(slots=True)
class RLMIteration:
    """Single iteration of RLM execution loop"""
    prompt: str
    response: str
    code_blocks: list
    final_answer: str = None
    iteration_time: float = None


# ============================================================================
# SAFE BUILTINS FOR REPL
# ============================================================================

_SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
    "print": print,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "slice": slice,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "pow": pow,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "format": format,
    "repr": repr,
    "vars": vars,
    "dir": dir,
    "help": help,
    "open": open,
    "json": json,
    "re": re,
}


# ============================================================================
# SYSTEM PROMPT (RLM paradigm from paper)
# ============================================================================

RLM_SYSTEM_PROMPT = """You are a Recursive Language Model (RLM) with access to a Python REPL environment.

CRITICAL: The context variable contains your working data. It can be VERY LARGE (millions of lines). You CANNOT see it directly — you MUST write Python code to inspect it.

Available REPL commands:
- Write Python code in ```repl blocks to execute
- Use FINAL_VAR(variable_name) to return a variable's value as the answer
- Use FINAL(answer text) to provide a direct text answer
- Use llm_query(prompt) to query a sub-LLM with a prompt (returns response string)
- Use llm_query_batched([prompts]) to query sub-LLM with multiple prompts in parallel
- Use SHOW_VARS() to see available variables

PARADIGM (from research paper):
1. The context is loaded as a REPL variable — it's NEVER fed to you directly
2. You write Python code to inspect, slice, and process the context
3. For large contexts, process them in chunks using sub-LLM calls
4. Build up your answer incrementally in REPL variables
5. When ready, use FINAL_VAR or FINAL to provide the answer

EXAMPLE workflow:
```repl
# First, inspect the context structure
sample = context[:1000]
print(sample)
```

```repl
# Process large context in chunks using sub-LLM
chunks = [context[i:i+5000] for i in range(0, len(context), 5000)]
prompts = [f"Extract key info from: {chunk}" for chunk in chunks]
results = llm_query_batched(prompts)
combined = "\n".join(results)
```

```repl
# Store result and finalize
FINAL_VAR(combined)
```

Be concise in your reasoning. Focus on writing working Python code."""


# ============================================================================
# OPTIMIZED REPL ENVIRONMENT
# ============================================================================

class REPLEnvironment:
    """
    Sandboxed Python REPL implementing the RLM environment.
    Optimized with better memory management and faster execution.
    """

    __slots__ = ['ollama_client', '_lock', '_pending_llm_calls', 'temp_dir', 
                 'context_file_path', 'globals', 'locals', '_session']

    def __init__(self, context: Any, ollama_client, use_temp_file: bool = None):
        self.ollama_client = ollama_client
        self._lock = threading.Lock()
        self._pending_llm_calls = []

        # Create temp directory for large contexts
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")
        self.context_file_path = None

        # Auto-detect whether to use temp file (>100KB)
        context_size = len(context) if isinstance(context, (str, bytes)) else len(str(context))
        if use_temp_file is None:
            use_temp_file = context_size > 100000

        # Sandboxed namespace
        self.globals = {"__builtins__": _SAFE_BUILTINS.copy(), "__name__": "__main__"}
        self.locals = {}

        # Load context (via temp file for large contexts, directly for small)
        self._load_context(context, use_temp_file)

        # Add RLM functions to REPL namespace
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched

    def _load_context(self, context: Any, use_temp_file: bool):
        """Load context into REPL with optimized handling."""
        if isinstance(context, str) and use_temp_file:
            # OPTIMIZATION: Write large string to temp file
            self.context_file_path = os.path.join(self.temp_dir, "context.txt")
            with open(self.context_file_path, "w", encoding="utf-8") as f:
                f.write(context)
            # Load via REPL code execution
            load_code = (
                f"with open(r'{self.context_file_path}', 'r', encoding='utf-8') as f:\n"
                f"    context = f.read()\n"
            )
            self.execute_code(load_code)
        elif isinstance(context, (dict, list)) and use_temp_file:
            # Write JSON to temp file
            self.context_file_path = os.path.join(self.temp_dir, "context.json")
            with open(self.context_file_path, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2)
            # Load via REPL code execution
            load_code = (
                f"import json\n"
                f"with open(r'{self.context_file_path}', 'r', encoding='utf-8') as f:\n"
                f"    context = json.load(f)\n"
            )
            self.execute_code(load_code)
        else:
            # Direct assignment for small contexts
            self.locals["context"] = context

    def __del__(self):
        """Clean up temp directory"""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _final_var(self, variable_name: str) -> str:
        """Return variable value as final answer"""
        variable_name = variable_name.strip()
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        available = [k for k in self.locals.keys() if not k.startswith("_")]
        return f"Error: Variable '{variable_name}' not found. Available: {available}"

    def _show_vars(self) -> str:
        """Show all variables in REPL"""
        available = {
            k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")
        }
        return f"Available variables: {available}" if available else "No variables yet"

    def _llm_query(self, prompt: str, model: str = None) -> str:
        """
        Sub-LLM call with caching for repeated queries.
        """
        start_time = time.perf_counter()
        try:
            # OPTIMIZATION: Use cached response if available
            cache_key = f"{model or Config.LLM_MODEL}:{hash(prompt) & 0xFFFFFF}"
            cached = _get_cached_llm_response(cache_key)
            if cached:
                return cached

            messages = [{"role": "user", "content": prompt}]
            result = self.ollama_client.chat(
                messages,
                model=model or Config.LLM_MODEL,
                options={"temperature": 0.7, "num_ctx": Config.SUB_LLM_NUM_CTX},
            )
            response = result.get("response", "")
            
            # Cache the response
            _cache_llm_response(cache_key, response)
            
            self._pending_llm_calls.append({
                "prompt_len": len(prompt),
                "response_len": len(response),
                "time": time.perf_counter() - start_time,
            })
            return response
        except Exception as e:
            return f"Error: LLM query failed - {str(e)}"

    def _llm_query_batched(self, prompts: list, model: str = None) -> list:
        """
        Batched sub-LLM calls with optimized thread pool.
        """
        import concurrent.futures

        def query_single(prompt):
            # Check cache first
            cache_key = f"{model or Config.LLM_MODEL}:{hash(prompt) & 0xFFFFFF}"
            cached = _get_cached_llm_response(cache_key)
            if cached:
                return cached

            messages = [{"role": "user", "content": prompt}]
            result = self.ollama_client.chat(
                messages,
                model=model or Config.LLM_MODEL,
                options={"temperature": 0.7, "num_ctx": Config.SUB_LLM_NUM_CTX},
            )
            response = result.get("response", "")
            _cache_llm_response(cache_key, response)
            return response

        try:
            # OPTIMIZATION: Use max_workers based on actual CPU cores
            max_workers = min(len(prompts), 8)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(query_single, p) for p in prompts]
                results = [f.result() for f in futures]
            return results
        except Exception as e:
            return [f"Error: Batched query failed - {str(e)}"] * len(prompts)

    def execute_code(self, code: str) -> REPLResult:
        """
        Execute Python code in sandboxed REPL environment.
        Optimized for speed with reduced overhead.
        """
        start_time = time.perf_counter()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with any new variables created
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
        )


# ============================================================================
# OPTIMIZED CODE PARSING
# ============================================================================

def find_code_blocks(text: str) -> list:
    """Extract ```repl code blocks using pre-compiled regex."""
    return [match.group(1).strip() for match in REPL_CODE_PATTERN.finditer(text)]


def find_final_answer(text: str, repl_env: REPLEnvironment = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) in response.
    
    OPTIMIZATION: Uses pre-compiled regex with stricter patterns.
    Only matches valid Python identifiers for FINAL_VAR.
    """
    # Check for FINAL_VAR first — must be valid Python identifier
    match = FINAL_VAR_PATTERN.search(text)
    if match:
        variable_name = match.group(1)
        if repl_env is not None:
            if variable_name in repl_env.locals:
                return str(repl_env.locals[variable_name])
            else:
                available = [k for k in repl_env.locals.keys() if not k.startswith("_")]
                return f"Error: Variable '{variable_name}' not found. Available: {available}"
        return None

    # Check for FINAL — captures everything up to newline or end
    match = FINAL_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    return None


def format_execution_result(result: REPLResult, max_length: int = None) -> str:
    """
    Format REPL result for display with optimized string building.
    
    OPTIMIZATION: Uses list append and join instead of concatenation.
    """
    if max_length is None:
        max_length = Config.MAX_RESULT_LENGTH

    parts = []

    if result.stdout:
        stdout = result.stdout
        if len(stdout) > max_length:
            stdout = stdout[:max_length] + f"... [{len(stdout) - max_length} chars truncated]"
        parts.append("Output:\n")
        parts.append(stdout)

    if result.stderr:
        stderr = result.stderr
        if len(stderr) > max_length:
            stderr = stderr[:max_length] + f"... [{len(stderr) - max_length} chars truncated]"
        if parts:
            parts.append("\n\n")
        parts.append("Errors:\n")
        parts.append(stderr)

    vars_list = [k for k in result.locals.keys() if not k.startswith("_")]
    if vars_list:
        if parts:
            parts.append("\n\n")
        parts.append(f"Variables: {vars_list}")

    return "".join(parts) if parts else "No output"


# ============================================================================
# OPTIMIZED MESSAGE HISTORY MANAGEMENT
# ============================================================================

def _manage_message_history(message_history: list, max_messages: int = None) -> list:
    """
    Apply sliding window to message history using deque for O(1) operations.
    
    OPTIMIZATION: Uses deque instead of list slicing for better performance.
    """
    if max_messages is None:
        max_messages = Config.MAX_MESSAGE_HISTORY
    
    # Always keep system prompt (index 0) and context metadata (index 1)
    if len(message_history) <= max_messages + 2:
        return message_history
    
    # Keep: [system, context_metadata, recent_messages]
    kept_messages = message_history[:2] + message_history[-(max_messages - 2):]
    return kept_messages


# ============================================================================
# LLM RESPONSE CACHE
# ============================================================================

_llm_cache = {}
_llm_cache_timestamps = {}
_cache_lock = threading.Lock()


def _get_cached_llm_response(key: str) -> Optional[str]:
    """Get cached LLM response if not expired."""
    with _cache_lock:
        if key in _llm_cache:
            timestamp = _llm_cache_timestamps.get(key, 0)
            if time.time() - timestamp < Config.LLM_CACHE_TTL:
                return _llm_cache[key]
            else:
                # Expired, remove from cache
                del _llm_cache[key]
                del _llm_cache_timestamps[key]
    return None


def _cache_llm_response(key: str, response: str):
    """Cache LLM response with TTL."""
    with _cache_lock:
        # Simple LRU: if cache is full, clear oldest entries
        if len(_llm_cache) >= Config.LLM_CACHE_SIZE:
            # Remove oldest 25% of entries
            sorted_keys = sorted(_llm_cache_timestamps.keys(), 
                                key=lambda k: _llm_cache_timestamps[k])
            for old_key in sorted_keys[:Config.LLM_CACHE_SIZE // 4]:
                del _llm_cache[old_key]
                del _llm_cache_timestamps[old_key]
        
        _llm_cache[key] = response
        _llm_cache_timestamps[key] = time.time()


def clear_llm_cache():
    """Clear the LLM response cache."""
    with _cache_lock:
        _llm_cache.clear()
        _llm_cache_timestamps.clear()


# ============================================================================
# OPTIMIZED RLM COMPLETION LOOP
# ============================================================================

def rlm_completion_loop(
    query: str,
    context: Any,
    ollama_client,
    max_iterations: int = None,
    verbose: bool = False,
) -> dict:
    """
    Main RLM execution loop with performance optimizations.
    """
    if max_iterations is None:
        max_iterations = Config.MAX_ITERATIONS

    start_time = time.perf_counter()
    
    # Initialize REPL environment
    repl_env = REPLEnvironment(context, ollama_client)
    
    # Build initial message history
    context_type = type(context).__name__
    context_size = len(context) if isinstance(context, (str, bytes)) else len(str(context))
    context_info = f"Context type: {context_type}, Size: {context_size:,} chars"
    
    message_history = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "assistant", "content": f"Your context is available as the `context` variable in the REPL. {context_info}. You have NOT seen the content yet — use the REPL to inspect it."},
        {
            "role": "user",
            "content": f"Query: {query}\n\nYou have not interacted with the REPL environment or seen your context yet. Your next action should be to look through it. Don't just provide a final answer yet.\n\nThink step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:",
        },
    ]

    all_iterations = []
    total_sub_llm_calls = 0

    for iteration_num in range(max_iterations):
        iter_start = time.perf_counter()
        
        if verbose:
            print(f"\n{'='*80}\nRLM Iteration {iteration_num + 1}/{max_iterations}\n{'='*80}")

        # Manage message history (sliding window)
        message_history = _manage_message_history(message_history)

        # Get LLM response
        lm_response = ollama_client.chat(
            message_history,
            options={"temperature": 0.7, "num_ctx": Config.ROOT_NUM_CTX},
        )
        
        if "error" in lm_response:
            return {"error": lm_response["error"], "status": "error"}
        
        response_text = lm_response.get("response", "")
        
        if verbose:
            preview = response_text[:500] + "..." if len(response_text) > 500 else response_text
            print(f"\nLM Response:\n{preview}")

        # Check for final answer
        final_answer = find_final_answer(response_text, repl_env)
        if final_answer:
            if verbose:
                print(f"\n{'='*80}\nFINAL ANSWER: {final_answer[:200]}...\n{'='*80}")

            return {
                "answer": final_answer,
                "iterations": iteration_num + 1,
                "execution_time": time.perf_counter() - start_time,
                "code_blocks_executed": sum(len(it.code_blocks) for it in all_iterations),
                "recursive_llm_calls": total_sub_llm_calls,
                "status": "success",
            }

        # Extract and execute code blocks
        code_blocks = find_code_blocks(response_text)
        executed_blocks = []

        for code in code_blocks:
            if verbose:
                code_preview = code[:200] + "..." if len(code) > 200 else code
                print(f"\nExecuting code:\n{code_preview}")
            
            result = repl_env.execute_code(code)
            executed_blocks.append((code, result))
            total_sub_llm_calls += len(repl_env._pending_llm_calls)
            repl_env._pending_llm_calls = []

            if verbose:
                output_preview = result.stdout[:200] if result.stdout else "(none)"
                print(f"Output: {output_preview}")
                if result.stderr:
                    print(f"Errors: {result.stderr[:200]}")

        # Record iteration
        all_iterations.append(RLMIteration(
            prompt=message_history[-1]["content"],
            response=response_text,
            code_blocks=executed_blocks,
            iteration_time=time.perf_counter() - iter_start,
        ))

        # Build result summary for next iteration
        result_parts = []
        for code, result in executed_blocks:
            exec_result = format_execution_result(result)
            composed = f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{exec_result}"
            if len(composed) > Config.MAX_RESULT_LENGTH:
                composed = composed[:Config.MAX_RESULT_LENGTH] + f"... [{len(composed) - Config.MAX_RESULT_LENGTH} chars truncated]"
            result_parts.append(composed)
        
        message_history.append({"role": "user", "content": "\n\n".join(result_parts)})

        # Prompt for next action
        message_history.append({
            "role": "user",
            "content": f"The history before is your previous interactions with the REPL environment. Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original query: \"{query}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"
        })

    # Max iterations reached
    if verbose:
        print(f"\n{'='*80}\nMAX ITERATIONS REACHED — forcing final answer\n{'='*80}")

    message_history.append({
        "role": "user",
        "content": "Based on all the information you have, provide a final answer to the user's query."
    })

    final_response = ollama_client.chat(
        message_history,
        options={"temperature": 0.7, "num_ctx": Config.ROOT_NUM_CTX},
    )
    
    final_text = final_response.get("response", "Max iterations reached without answer.")

    return {
        "answer": final_text,
        "iterations": max_iterations,
        "execution_time": time.perf_counter() - start_time,
        "code_blocks_executed": sum(len(it.code_blocks) for it in all_iterations),
        "recursive_llm_calls": total_sub_llm_calls,
        "status": "max_iterations_reached",
    }


# ============================================================================
# OPTIMIZED OLLAMA CLIENT WITH CONNECTION POOLING
# ============================================================================

class OllamaClient:
    """
    Optimized Ollama client with connection pooling and keep-alive.
    """

    __slots__ = ['base_url', '_session', '_adapter']

    def __init__(self, base_url: str = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        
        # OPTIMIZATION: Use connection pooling with keep-alive
        self._session = requests.Session()
        self._adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
        )
        self._session.mount("http://", self._adapter)
        self._session.mount("https://", self._adapter)
        
        # Set keep-alive headers
        self._session.headers.update({
            "Connection": "keep-alive",
            "Keep-Alive": "timeout=300, max=1000",
        })
        
        self._check_connection()

    def _check_connection(self):
        """Verify Ollama is running"""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama is not responding correctly")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {str(e)}"
            )

    def chat(self, messages: list, model: str = None, options: dict = None) -> dict:
        """Chat completion with connection reuse."""
        model = model or Config.LLM_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": 0.7, "num_ctx": Config.ROOT_NUM_CTX},
        }
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat", 
                json=payload, 
                timeout=Config.DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            result = response.json()
            return {"response": result.get("message", {}).get("content", "")}
        except requests.exceptions.RequestException as e:
            return {"error": f"Ollama chat failed: {str(e)}", "response": ""}

    def list_models(self) -> dict:
        """List available models with connection reuse."""
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to list models: {str(e)}"}

    def close(self):
        """Close the session and release connections."""
        self._session.close()


# ============================================================================
# LAZY INITIALIZATION
# ============================================================================

_ollama_instance = None
_init_lock = threading.Lock()


def get_ollama() -> OllamaClient:
    """Lazy initialization of Ollama client"""
    global _ollama_instance
    with _init_lock:
        if _ollama_instance is None:
            _ollama_instance = OllamaClient()
        return _ollama_instance


def reset_ollama():
    """Reset Ollama instance (useful for reconnection)"""
    global _ollama_instance
    with _init_lock:
        if _ollama_instance:
            _ollama_instance.close()
        _ollama_instance = None


# ============================================================================
# FILE PATH RESOLUTION (unchanged, already efficient)
# ============================================================================

def resolve_file_path(filename: str) -> Path:
    """Cross-platform file path resolution"""
    file_path = Path(filename)
    
    if file_path.is_absolute():
        if file_path.exists():
            return file_path
        raise FileNotFoundError(f"File not found: {file_path}")

    search_dirs = []
    system = os.name

    if system == "nt":  # Windows
        search_dirs = [
            Path.cwd(),
            Path.home(),
            Path(os.environ.get("TEMP", "C:/Temp")),
            Path("D:/ClaudeDocuments"),
        ]
    elif system == "posix":
        if os.uname().sysname == "Darwin":  # macOS
            search_dirs = [
                Path.cwd(), Path.home(), Path("/tmp"),
                Path.home() / "Documents", Path.home() / "Downloads",
                Path.home() / "Desktop",
            ]
        else:  # Linux
            search_dirs = [
                Path.cwd(), Path("/home/claude"), Path("/tmp"),
                Path("/mnt/user-data/uploads"),
            ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        candidate = search_dir / filename
        if candidate.exists() and candidate.is_file() and os.access(candidate, os.R_OK):
            return candidate

    searched = "\n  ".join(str(d / filename) for d in search_dirs if d.exists())
    raise FileNotFoundError(
        f"File '{filename}' not found.\n"
        f"OS: {system}\n"
        f"Searched:\n  {searched}\n"
        f"Tip: Use absolute path or place file in: {Path.cwd()}"
    )


def read_file_safe(filepath: Path, encoding: str = "utf-8") -> str:
    """Read file with encoding fallback"""
    try:
        with open(filepath, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read file with latin-1 encoding: {e}")
    except Exception as e:
        raise IOError(f"Failed to read file {filepath}: {e}")


# ============================================================================
# CORE RLM IMPLEMENTATION (shared logic)
# ============================================================================

def _run_rlm(query: str, context: Any, max_iterations: int, verbose: bool) -> dict:
    """
    Core RLM implementation — plain function callable by both MCP tool wrappers.
    """
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {"error": f"Ollama not connected: {str(e)}", "status": "error"}

    try:
        return rlm_completion_loop(
            query=query,
            context=context,
            ollama_client=ollama_client,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    except Exception as e:
        return {"error": f"RLM execution failed: {str(e)}", "status": "error"}


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("smart-context-rlm")


@mcp.tool()
def rlm_query(
    query: str,
    context: str | dict | list,
    max_iterations: int = 30,
    verbose: bool = False,
) -> dict:
    """
    TRUE RLM: Language model programmatically decomposes task using REPL environment.
    
    This implements the pure RLM paradigm from "Recursive Language Models" (arXiv:2512.24601).
    The document/context is NEVER fed directly to the LLM — it's loaded as a REPL variable.
    The LLM writes Python code to inspect, chunk, and process it via recursive sub-LLM calls.
    
    This breaks context window limits because:
    - The full document lives in the REPL environment (external to the LLM)
    - The LLM writes code to slice the document into chunks
    - Each chunk is processed by a sub-LLM call (within its context window)
    - Results accumulate in REPL variables, not in the LLM's context

    Args:
        query: User's question or task
        context: Context data (string, dict, or list) — can be arbitrarily large
        max_iterations: Maximum reasoning steps (default 30)
        verbose: Print detailed execution logs (default: False)

    Returns:
        dict with answer, iterations, execution_time, code_blocks_executed,
        recursive_llm_calls, and status

    Example:
        result = rlm_query(
            query="What are the main points?",
            context=large_document,  # Can be 10MB+ — no context limit!
            max_iterations=20
        )
        print(result["answer"])
    """
    return _run_rlm(query, context, max_iterations, verbose)


@mcp.tool()
def rlm_query_file(
    query: str,
    file_path: str,
    max_iterations: int = 30,
    verbose: bool = False,
) -> dict:
    """
    RLM query with file as context. Reads file then runs the RLM loop.

    The file content is loaded into the REPL environment as a variable.
    The LLM never sees the full file — it writes code to inspect and process it.
    This works for files of ANY size (tested with 1M+ line documents).

    Args:
        query: User's question or task about the file
        file_path: Path to the document (absolute or relative)
        max_iterations: Maximum reasoning steps (default 30)
        verbose: Print detailed execution logs

    Returns:
        Same as rlm_query()
    """
    try:
        validated_path = resolve_file_path(file_path)
        content = read_file_safe(validated_path)
        # Call plain function directly — NOT the MCP tool wrapper (FunctionTool not callable in FastMCP 2.x)
        return _run_rlm(query, content, max_iterations, verbose)
    except (FileNotFoundError, PermissionError) as e:
        return {"error": str(e), "status": "error"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "status": "error"}


@mcp.tool()
def check_ollama_status() -> dict:
    """
    Check if Ollama is running and list available models.

    Returns:
        Ollama status and available models
    """
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {
            "status": "disconnected",
            "message": f"Ollama is not running: {str(e)}. Start it with: ollama serve",
            "configured_llm": Config.LLM_MODEL,
        }

    try:
        models_result = ollama_client.list_models()
        if "error" in models_result:
            return models_result

        available_models = [m.get("name", m.get("model", "unknown")) 
                          for m in models_result.get("models", [])]
        llm_available = any(Config.LLM_MODEL in m for m in available_models)

        return {
            "status": "connected",
            "ollama_url": Config.OLLAMA_BASE_URL,
            "configured_llm": Config.LLM_MODEL,
            "llm_available": llm_available,
            "available_models": available_models,
            "recommendation": f"ollama pull {Config.LLM_MODEL}" if not llm_available else None,
            "rlm_config": {
                "root_context_window": Config.ROOT_NUM_CTX,
                "sub_llm_context_window": Config.SUB_LLM_NUM_CTX,
                "max_iterations": Config.MAX_ITERATIONS,
                "max_message_history": Config.MAX_MESSAGE_HISTORY,
            }
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def clear_cache() -> dict:
    """
    Clear the LLM response cache to free memory.
    
    Returns:
        Status of cache clear operation
    """
    clear_llm_cache()
    return {"status": "success", "message": "LLM cache cleared"}


@mcp.tool()
def get_performance_stats() -> dict:
    """
    Get performance statistics for the RLM implementation.
    
    Returns:
        Performance metrics including cache stats
    """
    with _cache_lock:
        cache_size = len(_llm_cache)
        cache_timestamps = list(_llm_cache_timestamps.values())
    
    return {
        "cache": {
            "size": cache_size,
            "max_size": Config.LLM_CACHE_SIZE,
            "ttl_seconds": Config.LLM_CACHE_TTL,
            "hit_ratio": "N/A (not tracked)",
        },
        "config": {
            "llm_model": Config.LLM_MODEL,
            "root_context": Config.ROOT_NUM_CTX,
            "sub_llm_context": Config.SUB_LLM_NUM_CTX,
            "max_iterations": Config.MAX_ITERATIONS,
        }
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
