# Smart Context Manager — Pure RLM Edition

> **Based on:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) (arXiv:2512.24601)
> **Powered by:** Ollama (local AI) + FastMCP
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
                                              MCP Server (Ollama)
                                              document lives HERE
                                              AI writes code to slice it
                                              sub-AI calls process chunks
                                              final answer returned
   Your chat receives: [clean final answer]  ← context window barely touched
```

**What this means in practice:**

- Your main AI (Claude, etc.) only sees the **final answer** — not the document
- Your conversation context stays clean and fresh regardless of document size
- You can analyze a 100MB log file and your chat context barely grows
- No context rot — the main AI never has to "remember" a huge document
- Works on files with **millions of lines** (tested in the reference implementation)

The paper calls this _"unbounded input tokens"_ and _"unbounded semantic horizon"_ — the ability to do arbitrarily complex work over arbitrarily large inputs.

---

## How It Breaks Context Window Limits

Most AI tools fail on large documents because they try to feed the entire file into the AI's context window. This tool does something fundamentally different:

```
Traditional approach (FAILS on large files):
  Document (10MB) → AI context window (32K tokens) → OVERFLOW ❌

RLM approach (WORKS on any size):
  Document (10MB) → REPL variable (external to AI)
                         ↓
                    AI sees only: "context is a string, 10MB long"
                         ↓
                    AI writes Python code to slice it:
                    chunks = [context[i:i+15000] for i in range(0, len(context), 15000)]
                         ↓
                    Sub-AI calls process each chunk (within window)
                         ↓
                    Results accumulate in REPL variables
                         ↓
                    Final answer synthesized ✅
```

The document **never enters the AI's context window directly**. The AI writes code to inspect and process it programmatically — exactly like a programmer would.

---

## Quick Start (5 Minutes)

### Step 1: Install Ollama

Download from [ollama.com](https://ollama.com/) and install it.

Then pull the model:

```bash
ollama pull ministral-3:latest
```

> You can use any Ollama model. Change `LLM_MODEL` in the `Config` class inside `smart_context_mcp.py` to switch models.

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
    "smart-context": {
      "command": "python",
      "args": [
        "C:/Users/YourName/Path/To/Claude Desktop Windows/RLM + Nomic Text Emebeding/smart_context_mcp.py"
      ],
      "env": {
        "OLLAMA_NUM_PARALLEL": "1"
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
    "smart-context": {
      "command": "python3",
      "args": [
        "/Users/YourName/Path/To/Claude Desktop Mac/RLM + Nomic Text Emebeding/smart_context_mcp.py"
      ],
      "env": {
        "OLLAMA_NUM_PARALLEL": "1"
      }
    }
  }
}
```

### Step 4: Restart Your AI Client

After saving the config, restart Claude Desktop (or your AI client). The tool will appear automatically.

---

## The Three Tools

The server exposes exactly **3 tools** — nothing more, nothing less.

### `rlm_query` — Ask anything about any data

```
Use rlm_query with:
- query: "What are the main security vulnerabilities?"
- context: [paste your code, document, or data here]
```

| Parameter        | Type                 | Default  | Description                              |
| ---------------- | -------------------- | -------- | ---------------------------------------- |
| `query`          | string               | required | Your question or task                    |
| `context`        | string / dict / list | required | The data to analyze — can be any size    |
| `max_iterations` | integer              | 30       | How many reasoning steps before stopping |
| `verbose`        | boolean              | false    | Print detailed logs                      |

**Returns:** `answer`, `iterations`, `execution_time`, `code_blocks_executed`, `recursive_llm_calls`, `status`

---

### `rlm_query_file` — Ask anything about any file

```
Use rlm_query_file with:
- query: "Summarize the main findings"
- file_path: "C:/Users/YourName/Documents/report.txt"
```

| Parameter        | Type    | Default  | Description                 |
| ---------------- | ------- | -------- | --------------------------- |
| `query`          | string  | required | Your question or task       |
| `file_path`      | string  | required | Path to the file (any size) |
| `max_iterations` | integer | 30       | Reasoning steps             |
| `verbose`        | boolean | false    | Detailed logs               |

**File path tips:**

- Use full absolute paths for reliability
- If you give just a filename, it searches: current directory, home folder, Downloads, Documents, Desktop
- Works on Windows, macOS, and Linux

---

### `check_ollama_status` — Verify your setup

```
Use check_ollama_status
```

Returns: connection status, available models, context window config, and install commands if anything is missing.

---

## Usage Examples

### Analyze a Large Document

```
Use rlm_query_file with:
- query: "What are the key findings and recommendations?"
- file_path: "C:/Users/YourName/Documents/annual_report.txt"
```

The AI will:

1. Load the file into the REPL environment (not into its context)
2. Write code to inspect the document structure
3. Chunk it intelligently and process each chunk via sub-AI calls
4. Synthesize a final answer from all the results

### Search a Codebase

```
Use rlm_query_file with:
- query: "Find all functions that handle authentication and identify potential security issues"
- file_path: "C:/Projects/myapp/main.py"
```

### Analyze Log Files

```
Use rlm_query_file with:
- query: "Find all ERROR and EXCEPTION entries, group them by type, and identify the most frequent issues"
- file_path: "C:/Logs/app.log"
```

### Process Structured Data

```
Use rlm_query with:
- query: "Which customers have the highest lifetime value and what do they have in common?"
- context: [paste CSV data or JSON here]
```

### Needle in a Haystack

```
Use rlm_query_file with:
- query: "Find the magic number hidden in this file"
- file_path: "C:/Data/million_lines.txt"
- max_iterations: 15
```

The AI will write code to scan the file in chunks, find the relevant line, and return the answer — even if the file has 1 million lines.

---

## What Happens Inside the RLM Loop

When you call `rlm_query` or `rlm_query_file`, this is what happens:

````
1. Your document is loaded into a REPL variable called `context`
   (for files >100KB, written to a temp file and loaded via file I/O)

2. The root AI sees only metadata:
   "Context type: str, length: 5242880 chars (~1310720 tokens)"

3. The AI writes Python code in ```repl blocks:
   ```repl
   # Inspect the document
   print(len(context))
   print(context[:500])  # Preview first 500 chars
````

4. The code runs in a sandboxed Python environment

5. The AI sees the output and writes more code:

   ```repl
   # Chunk and process in parallel
   chunk_size = 15000
   chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

   # Screen chunks for relevance (runs all concurrently)
   results = llm_query_batched([
       f"Does this contain information about X? Reply YES/NO: {chunk}"
       for chunk in chunks
   ])
   relevant = [chunks[i] for i, r in enumerate(results) if "YES" in r]
   ```

6. Sub-AI calls (llm_query / llm_query_batched) process each chunk
   within their own context window

7. Results accumulate in REPL variables

8. When done, the AI signals completion:
   FINAL(Here is my answer based on the analysis...)
   or
   FINAL_VAR(final_answer) ← returns a variable from the REPL

````

The message history uses a **sliding window** (last 20 messages) to prevent the root AI's context from overflowing during long analysis sessions.

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

The RLM reasoning happens **inside the MCP server** (in Ollama), not inside the AI client. The client just calls the tool and gets back the final answer. This means:

- The client's own context window is **not consumed** by the document analysis
- The tool works the same regardless of which client calls it
- You can use a small-context client (like a mobile app) to analyze million-line files

---

## Configuration

Edit the `Config` class at the top of `smart_context_mcp.py`:

```python
class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_MODEL = "ministral-3:latest"    # Change to any Ollama model

    # Context window sizes
    ROOT_NUM_CTX = 32768    # Root AI: needs room for system prompt + history
    SUB_LLM_NUM_CTX = 24576 # Sub-AI: processes individual document chunks

    # RLM loop limits
    MAX_ITERATIONS = 30          # Max reasoning steps per query
    MAX_MESSAGE_HISTORY = 20     # Sliding window size (prevents root overflow)
    MAX_RESULT_LENGTH = 100000   # Max chars per REPL output fed back to root AI
````

**Recommended models (Ollama):**

| Model                | Size | Best For                          |
| -------------------- | ---- | --------------------------------- |
| `ministral-3:latest` | ~3GB | Default — fast, good reasoning    |
| `llama3.2:latest`    | ~2GB | Lighter, faster                   |
| `qwen2.5:14b`        | ~9GB | Better reasoning, needs more VRAM |
| `deepseek-r1:8b`     | ~5GB | Strong code analysis              |

---

## System Requirements

| Component               | Minimum       | Recommended |
| ----------------------- | ------------- | ----------- |
| **Python**              | 3.10+         | 3.12+       |
| **RAM**                 | 8 GB          | 16 GB+      |
| **VRAM (GPU)**          | 4 GB          | 8 GB+       |
| **Mac (Apple Silicon)** | 16 GB unified | 24 GB+      |
| **Storage**             | 5 GB          | 10 GB+      |
| **Ollama**              | Any version   | Latest      |

> **CPU-only:** Ollama works on CPU but is much slower. For large documents with many sub-AI calls, expect 5–30 minutes per query on CPU.

---

## Troubleshooting

### "Ollama not connected"

```bash
# Start Ollama
ollama serve

# Check it's running
ollama list

# Install the model if missing
ollama pull ministral-3:latest
```

Then verify: `Use check_ollama_status`

---

### "File not found"

Use the **full absolute path**:

- Windows: `C:\Users\YourName\Documents\report.txt`
- macOS: `/Users/YourName/Documents/report.txt`
- Linux: `/home/yourname/documents/report.txt`

---

### AI keeps hitting max iterations without finishing

The document may be too large for the default settings. Try:

```
Use rlm_query_file with:
- query: "Summarize this document"
- file_path: "your_file.txt"
- max_iterations: 50    ← increase this
```

Or ask a more specific question to reduce the scope of analysis.

---

### Sub-AI calls are slow

This is normal for large documents — each `llm_query_batched` call runs multiple Ollama requests concurrently. Performance depends on your hardware.

To speed up:

1. Use a smaller/faster model (e.g., `llama3.2:latest`)
2. Increase `SUB_LLM_NUM_CTX` to process larger chunks (fewer total calls)
3. Use a GPU — CPU inference is 10–50x slower

---

### "No module named 'fastmcp'" or "No module named 'requests'"

```bash
pip install fastmcp requests
```

---

## Architecture

```
Your Question
     ↓
AI Client (Claude Desktop / RooCode / OpenCode / etc.)
     ↓ calls MCP tool
┌─────────────────────────────────────────────────────────┐
│  smart_context_mcp.py (MCP Server)                      │
│                                                         │
│  rlm_query / rlm_query_file                             │
│       ↓                                                 │
│  REPLEnvironment                                        │
│    context = [your document]  ← lives here, not in AI  │
│    llm_query()                ← sub-AI calls            │
│    llm_query_batched()        ← parallel sub-AI calls   │
│       ↓                                                 │
│  rlm_completion_loop (Algorithm 1 from the paper)       │
│    Root AI ← sees only metadata about context           │
│    Root AI → writes Python code                         │
│    REPL executes code                                   │
│    Sub-AI calls process chunks                          │
│    Results accumulate in REPL variables                 │
│    Repeat until FINAL() or FINAL_VAR()                  │
│       ↓                                                 │
│  Ollama (local AI, no internet)                         │
│    ministral-3:latest (or any model you choose)         │
└─────────────────────────────────────────────────────────┘
     ↓
Final Answer returned to your AI client
```

---

## Also Included: Document RAG + Memory Servers

The repository also includes two additional MCP servers in the `RAG (LanceDB)` folder. These are **separate tools** with different capabilities:

| Server                   | What It Does                                        | Technology                      |
| ------------------------ | --------------------------------------------------- | ------------------------------- |
| `document_rag_server.py` | Upload PDFs/DOCX/XLSX, hybrid search with citations | LanceDB + sentence-transformers |
| `lancedb_memory.py`      | Persistent conversation memory across sessions      | LanceDB + sentence-transformers |

These servers use **sentence-transformers** (not Ollama) and run entirely on CPU. They do not use the RLM approach — they use traditional RAG (Retrieval-Augmented Generation) with vector embeddings.

**To use them, install additional dependencies:**

```bash
pip install lancedb sentence-transformers torch pyarrow pandas

# For document parsing
pip install PyPDF2 python-docx openpyxl python-pptx rank-bm25 nltk langchain-text-splitters
```

**Add to your config:**

```json
{
  "mcpServers": {
    "smart-context": {
      "command": "python",
      "args": ["...path.../smart_context_mcp.py"],
      "env": { "OLLAMA_NUM_PARALLEL": "1" }
    },
    "memory": {
      "command": "python",
      "args": ["...path.../RAG (LanceDB)/lancedb_memory.py"]
    },
    "documents": {
      "command": "python",
      "args": ["...path.../RAG (LanceDB)/document_rag_server.py"]
    }
  }
}
```

---

## FAQ

**Q: Does my data leave my machine?**  
A: No. Ollama runs entirely locally. No data is sent to any external service.

**Q: How large a file can it handle?**  
A: There is no hard limit. The reference implementation (rlm-minimal) was tested with 1 million line files. The document lives in the REPL environment, not in the AI's context window.

**Q: Do I need to know Python?**  
A: No. The AI writes the Python code internally. You just ask questions in plain language.

**Q: Can I use a different model than `ministral-3:latest`?**  
A: Yes. Edit `LLM_MODEL` in the `Config` class. Any model available in Ollama works.

**Q: Why only 3 tools? The old version had 16+.**  
A: The extra tools (semantic search, embeddings, vector store, etc.) were not part of the RLM paradigm from the research paper. They added complexity and dependencies without improving the core capability. The pure RLM approach — just `rlm_query`, `rlm_query_file`, and `check_ollama_status` — is simpler, more reliable, and handles larger documents.

**Q: Is the REPL sandboxed?**  
A: Yes. The REPL blocks `eval`, `exec`, `compile`, `input`, `globals`, and `locals`. Standard library imports and file access are allowed (needed for the AI to read document chunks).

**Q: What if the AI doesn't finish in `max_iterations`?**  
A: The loop forces a final answer from the AI after max iterations are reached. You can increase `max_iterations` for complex tasks.

---

## Reference

This implementation is based on:

> **Recursive Language Models**  
> Alex L. Zhang, Tim Kraska, Omar Khattab  
> arXiv:2512.24601v2 [cs.AI] — January 28, 2026  
> https://arxiv.org/abs/2512.24601

Reference implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

---

_Runs entirely on your local machine. No cloud, no subscriptions, no data leaving your device._
