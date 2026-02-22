# Smart Context Manager — RLM REPL Server

> **Based on:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) (arXiv:2512.24601)
> **Architecture:** Claude (root LLM) orchestrates via MCP → Sub-LLMs (OpenRouter/LM Studio) handle processing
> **Works with:** Claude Desktop, Claude Code, RooCode, OpenCode, and any MCP-compatible client

---

## What This Does (In One Sentence)

This tool gives your AI assistant an **effectively infinite context window** — it can read and reason over documents of any size without hitting context limits or suffering from context rot.

---

## Infinite Context Window — How It Actually Works

**Yes** — by offloading document processing to this MCP server, you bypass the context window of your main AI client entirely.

Here's the key insight from the research paper:

> _"Context rot"_ — the phenomenon where AI quality degrades steeply as prompts get longer — happens because the full document is stuffed into the AI's context window. RLMs solve this by **never putting the document in the context window at all.**

```
❌ Without RLM (context rot):
   Your chat: [system prompt] + [your question] + [10MB document] = OVERFLOW or degraded quality

✅ With RLM (infinite context):
   Your chat: [system prompt] + [your question] + [tool call]
                                                        ↓
                                              MCP Server (REPL)
                                              document lives HERE
                                              Claude writes Python to slice it
                                              Sub-LLM calls process chunks
                                              final answer returned
   Your chat receives: [clean final answer]  ← context window barely touched
```

**What this means in practice:**

- Your main AI (Claude) only sees **metadata** — variable names, types, sizes — not the document
- Your conversation context stays clean and fresh regardless of document size
- You can analyze a 100MB log file and your chat context barely grows
- No context rot — Claude never has to "remember" a huge document
- Works on files with **millions of lines**

The paper calls this _"unbounded input tokens"_ and _"unbounded semantic horizon"_ — the ability to do arbitrarily complex work over arbitrarily large inputs.

---

## Architecture: Claude Orchestrates, Sub-LLMs Process

This implementation follows the RLM paradigm where **Claude is the root LLM** that orchestrates everything via MCP tool calls:

```
Claude (Root LLM)
    ↓ calls MCP tools
┌─────────────────────────────────────────────────────────────┐
│  smart_context_mcp.py (MCP Server)                          │
│                                                             │
│  load_context() → Loads document into REPL variable         │
│  repl_execute() → Claude writes Python to inspect/process   │
│       ↓                                                     │
│  REPLEnvironment                                            │
│    context = [your document]  ← lives here, not in Claude  │
│    llm_query()                ← sub-LLM calls               │
│    llm_query_batched()        ← parallel sub-LLM calls      │
│       ↓                                                     │
│  Sub-LLM Client (automatic failover)                        │
│    Primary: OpenRouter → Gemini 3 Flash (1M context)        │
│    Fallback: LM Studio → Qwen3 14B (local)                  │
└─────────────────────────────────────────────────────────────┘
    ↓
Claude receives: metadata-only results (variable names, truncated stdout)
    ↓
Claude provides final answer in conversation
```

**Key insight:** Claude never sees the raw document. It writes Python code to inspect and process it. Results stay in REPL variables. Only metadata returns to Claude's context.

---

## Quick Start (5 Minutes)

### Step 1: Get a Sub-LLM Backend

**Option A: OpenRouter (Recommended — Cloud)**

1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Get an API key
3. Set environment variable: `OPENROUTER_API_KEY=sk-or-v1-...`

Uses Gemini 3 Flash (1M token context, fast, cheap) by default.

**Option B: LM Studio (Local — Free)**

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install a model (e.g., Qwen3 14B)
3. Start the local server on port 1234

No API key needed — runs entirely offline.

### Step 2: Install Python Dependencies

```bash
pip install fastmcp requests
```

That's it. No embeddings, no vector databases, no sentence-transformers needed.

### Step 3: Configure Your AI Client

**Claude Desktop (Windows)**  
Edit: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rlm-repl": {
      "command": "python",
      "args": [
        "C:/Users/YourName/Path/To/Claude Desktop Windows/RLM + Nomic Text Emebeding/smart_context_mcp.py"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
      }
    }
  }
}
```

**Claude Desktop (macOS)**  
Edit: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rlm-repl": {
      "command": "python3",
      "args": [
        "/Users/YourName/Path/To/Claude Desktop Mac/RLM + Nomic Text Emebeding/smart_context_mcp.py"
      ],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
      }
    }
  }
}
```

### Step 4: Restart Your AI Client

After saving the config, restart Claude Desktop. The tools will appear automatically.

---

## The Seven Tools

The server exposes **7 tools** for the RLM paradigm:

### `load_context` — Load document into REPL

```
Use load_context with:
- source: "C:/Users/YourName/Documents/report.txt"
- source_type: "file"
- context_label: "research paper"
```

| Parameter       | Type   | Default   | Description                                    |
| --------------- | ------ | --------- | ---------------------------------------------- |
| `source`        | string | required  | File path or inline content                    |
| `source_type`   | string | "file"    | "file" or "inline"                             |
| `context_label` | string | "text"    | Description of what this context is            |

**Returns:** Session info with context metadata (type, size, available variables).

---

### `repl_execute` — Run Python in the REPL

```
Use repl_execute with:
- code: "print(len(context))\nprint(context[:500])"
```

| Parameter | Type   | Description                                   |
| --------- | ------ | --------------------------------------------- |
| `code`    | string | Python code to execute in the REPL environment |

**Returns:** Dict with stdout (truncated), stderr, variables (names + types + sizes), execution time.

**Available in REPL:**
- `context` — The loaded document
- `llm_query(prompt)` — Call sub-LLM with a prompt
- `llm_query_batched([prompts])` — Parallel sub-LLM calls

---

### `sub_llm_query` — Direct sub-LLM call

```
Use sub_llm_query with:
- prompt: "Summarize the key points of this text..."
```

| Parameter | Type   | Description              |
| --------- | ------ | ------------------------ |
| `prompt`  | string | The prompt to send       |

**Returns:** The sub-LLM's response text.

---

### `sub_llm_batch` — Parallel sub-LLM calls

```
Use sub_llm_batch with:
- prompts: ["Analyze chunk 1...", "Analyze chunk 2...", "Analyze chunk 3..."]
```

| Parameter | Type         | Description                        |
| --------- | ------------ | ---------------------------------- |
| `prompts` | list[string] | List of prompts (max 8 concurrent) |

**Returns:** List of response strings in the same order.

---

### `get_variable` — Retrieve REPL variable

```
Use get_variable with:
- name: "analysis_result"
- max_chars: 5000
```

| Parameter  | Type   | Default | Description                        |
| ---------- | ------ | ------- | ---------------------------------- |
| `name`     | string | required| Variable name in the REPL          |
| `max_chars`| int    | 5000    | Maximum characters to return       |

**Returns:** String representation of the variable.

---

### `session_status` — Check session state

```
Use session_status
```

**Returns:** Session info including active state, context metadata, variables, sub-LLM backend health, and call statistics.

---

### `reset_session` — Clear the REPL

```
Use reset_session
```

**Returns:** Confirmation with final session stats.

---

## Usage Examples

### Analyze a Large Document

```
Use load_context with:
- source: "C:/Users/YourName/Documents/annual_report.txt"
- source_type: "file"
- context_label: "annual report"
```

Then Claude will:
1. Inspect the document via `repl_execute`: `print(len(context))`, `print(context[:500])`
2. Write code to chunk and process it
3. Use `llm_query_batched()` to process chunks in parallel
4. Aggregate results in REPL variables
5. Provide the final answer directly in conversation

### Search a Codebase

```
Use load_context with:
- source: "C:/Projects/myapp/main.py"
- source_type: "file"
- context_label: "source code"
```

Then ask Claude: "Find all functions that handle authentication and identify potential security issues"

### Analyze Log Files

```
Use load_context with:
- source: "C:/Logs/app.log"
- source_type: "file"
- context_label: "application logs"
```

Then ask Claude: "Find all ERROR and EXCEPTION entries, group them by type, and identify the most frequent issues"

### Needle in a Haystack

```
Use load_context with:
- source: "C:/Data/million_lines.txt"
- source_type: "file"
```

Then ask Claude: "Find the magic number hidden in this file"

Claude will write code to scan the file in chunks, find the relevant line, and return the answer — even if the file has 1 million lines.

---

## What Happens Inside the RLM Loop

When you load context and ask a question, this is what happens:

```
1. Your document is loaded into a REPL variable called `context`
   (for files >100KB, written to a temp file and loaded via file I/O)

2. Claude sees only metadata:
   "Context loaded: 5,242,880 chars, available as 'context' variable"

3. Claude writes Python code via repl_execute():
   
   # Inspect the document
   print(len(context))
   print(context[:500])  # Preview first 500 chars

4. Claude sees the output and writes more code:

   # Chunk and process in parallel
   chunk_size = 15000
   chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
   
   # Screen chunks for relevance (runs all concurrently)
   results = llm_query_batched([
       f"Does this contain information about X? Reply YES/NO: {chunk}"
       for chunk in chunks
   ])
   relevant = [chunks[i] for i, r in enumerate(results) if "YES" in r]

5. Sub-LLM calls (llm_query / llm_query_batched) process each chunk
   within their own context window (1M tokens for Gemini)

6. Results accumulate in REPL variables

7. Claude provides the final answer directly in conversation
```

**Key difference from the old approach:** Claude provides the answer directly — no `FINAL()` or `FINAL_VAR()` needed. The REPL is just for processing.

---

## Does This Work With Other AI Clients?

**Yes.** This is a standard MCP server. Any software that supports the Model Context Protocol can use it.

| Client | Support | Notes |
|--------|---------|-------|
| **Claude Desktop** | ✅ Full | Primary target. Configure via `claude_desktop_config.json` |
| **Claude Code** | ✅ Full | Add to MCP config in Claude Code settings |
| **RooCode** | ✅ Full | Add to `.roo/mcp.json` or global `~/.roo/mcp.json` |
| **OpenCode** | ✅ Full | Add to OpenCode's MCP server configuration |
| **Any MCP client** | ✅ Full | Any client implementing the MCP standard works |

---

## Configuration

Edit the `Config` class at the top of `smart_context_mcp.py`:

```python
class Config:
    # Primary sub-LLM: OpenRouter → Gemini 3 Flash
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-key-here")
    OPENROUTER_MODEL = os.environ.get("RLM_SUB_MODEL", "google/gemini-3-flash-preview")

    # Fallback sub-LLM: LM Studio → Qwen3 14B (local)
    LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-14b")

    # Sub-LLM parameters
    SUB_LLM_TEMPERATURE = 0.3
    SUB_LLM_MAX_TOKENS = 16384
    SUB_LLM_TIMEOUT = 120

    # REPL output limits
    STDOUT_PREVIEW_CHARS = 500
    STDERR_PREVIEW_CHARS = 300
    MAX_BATCH_WORKERS = 8
```

**Recommended OpenRouter models:**

| Model                       | Context  | Best For                          |
| --------------------------- | -------- | --------------------------------- |
| `google/gemini-3-flash-preview` | 1M tokens | Default — fast, huge context      |
| `anthropic/claude-3-haiku`  | 200K     | Fast, high quality                |
| `openai/gpt-4o-mini`        | 128K     | Good balance                      |
| `meta-llama/llama-3.1-70b-instruct` | 128K | Open source option            |

**LM Studio models (local):**

| Model                | Size  | Best For                          |
| -------------------- | ----- | --------------------------------- |
| `qwen3-14b`          | ~9GB  | Default — good reasoning          |
| `llama-3.2-3b`       | ~2GB  | Lighter, faster                   |
| `deepseek-r1-8b`     | ~5GB  | Strong code analysis              |

---

## System Requirements

| Component               | Minimum       | Recommended |
| ----------------------- | ------------- | ----------- |
| **Python**              | 3.10+         | 3.12+       |
| **RAM**                 | 4 GB          | 8 GB+       |
| **Internet**            | Required for OpenRouter | Optional (LM Studio works offline) |

> **Note:** This version uses cloud-based OpenRouter by default, so no local GPU needed. For offline use, configure LM Studio with a local model.

---

## Troubleshooting

### "No backends available"

```bash
# Option 1: Set OpenRouter API key
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Option 2: Start LM Studio
# Open LM Studio, load a model, start the server on port 1234
```

Then verify: `Use session_status`

---

### "File not found"

Use the **full absolute path**:

- Windows: `C:\Users\YourName\Documents\report.txt`
- macOS: `/Users/YourName/Documents/report.txt`
- Linux: `/home/yourname/documents/report.txt`

---

### Sub-LLM calls are slow

For OpenRouter: This depends on the model and current load. Gemini Flash is typically fast.

For LM Studio: Performance depends on your hardware.

To speed up:
1. Use a faster model (Gemini Flash, GPT-4o-mini)
2. Increase `SUB_LLM_MAX_TOKENS` to process larger chunks (fewer total calls)
3. Use GPU for LM Studio — CPU inference is 10–50x slower

---

### "No module named 'fastmcp'" or "No module named 'requests'"

```bash
pip install fastmcp requests
```

---

## Also Included: Document RAG + Memory Servers

The repository also includes two additional MCP servers in the `RAG (LanceDB)` folder. These are **separate tools** with different capabilities:

| Server                   | What It Does                                        | Technology                      |
| ------------------------ | --------------------------------------------------- | ------------------------------- |
| `document_rag_server.py` | Upload PDFs/DOCX/XLSX, hybrid search with citations | LanceDB + sentence-transformers |
| `lancedb_memory.py`      | Persistent conversation memory across sessions      | LanceDB + sentence-transformers |

These servers use **sentence-transformers** (not sub-LLMs) and run entirely on CPU. They use traditional RAG (Retrieval-Augmented Generation) with vector embeddings.

**To use them, install additional dependencies:**

```bash
pip install lancedb sentence-transformers torch pyarrow pandas

# For document parsing
pip install PyPDF2 python-docx openpyxl python-pptx rank-bm25 nltk langchain-text-splitters
```

---

## FAQ

**Q: Does my data leave my machine?**  
A: With OpenRouter, yes — prompts are sent to the cloud API. With LM Studio, no — everything runs locally.

**Q: How large a file can it handle?**  
A: There is no hard limit. The document lives in the REPL environment, not in Claude's context window. Files with millions of lines work fine.

**Q: Do I need to know Python?**  
A: No. Claude writes the Python code internally. You just ask questions in plain language.

**Q: Can I use a different sub-LLM model?**  
A: Yes. Set `RLM_SUB_MODEL` environment variable for OpenRouter, or `LMSTUDIO_MODEL` for LM Studio.

**Q: What's the difference between this and the old Ollama-based version?**  
A: This version uses Claude as the root LLM (via MCP) and sub-LLMs for processing. The old version ran everything in Ollama. This approach is more flexible — you can use any MCP-compatible client as the root LLM.

**Q: Is the REPL sandboxed?**  
A: Yes. The REPL blocks `eval`, `exec`, `compile`, `input`, `globals`, and `locals`. Standard library imports and file access are allowed.

---

## Reference

This implementation is based on:

> **Recursive Language Models**  
> Alex L. Zhang, Tim Kraska, Omar Khattab  
> arXiv:2512.24601v2 [cs.AI] — January 28, 2026  
> https://arxiv.org/abs/2512.24601

Reference implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

---

_Runs with cloud APIs (OpenRouter) or entirely locally (LM Studio). Choose based on your privacy and offline needs._
