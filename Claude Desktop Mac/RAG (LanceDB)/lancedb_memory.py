#!/usr/bin/env python3
"""
LanceDB Memory MCP Server for Claude
Provides persistent conversational memory across sessions
CPU-OPTIMIZED: Tuned for maximum CPU performance and stability
CROSS-PLATFORM: Works on Windows, macOS, and Linux
"""

import asyncio
import json
import sys
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

# Redirect stdout to stderr for debugging (MCP uses stdout for JSON-RPC)
import builtins

original_print = builtins.print


def debug_print(*args, **kwargs):
    kwargs["file"] = sys.stderr
    original_print(*args, **kwargs)


builtins.print = debug_print

# CPU optimization environment variables
# Set before importing torch to maximize CPU utilization
cpu_cores = os.cpu_count() or 4
os.environ["TORCH_NUM_THREADS"] = str(cpu_cores)
os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_cores)

try:
    # MCP imports
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

    # LanceDB and embeddings
    import lancedb
    from sentence_transformers import SentenceTransformer
    import torch
    import pyarrow as pa
    import pandas as pd

    print("✓ Imports successful", flush=True)
except Exception as e:
    print(f"✗ Import error: {e}", flush=True)
    sys.exit(1)


# Cross-platform configuration
def get_memory_dir():
    """Get cross-platform memory directory"""
    system = platform.system()
    if system == "Windows":
        return Path.home() / ".claude_memory"
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "ClaudeMemory"
    else:  # Linux
        return Path.home() / ".local" / "share" / "claude_memory"


# Configuration
MEMORY_DIR = get_memory_dir()
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(MEMORY_DIR / "lancedb")

# Initialize server
server = Server("lancedb_memory")

# Global state
db = None
table = None
embed_model = None


def init_database():
    """Initialize LanceDB and embedding model - CPU optimized"""
    global db, table, embed_model

    try:
        # Force CPU for stability and compatibility
        device = "cpu"
        cpu_count = os.cpu_count() or 4

        print(f"═" * 60, flush=True)
        print(f"MEMORY SERVER - {platform.system()}", flush=True)
        print(f"═" * 60, flush=True)
        print(f"Device: CPU (optimized for performance)", flush=True)
        print(f"Available CPU cores: {cpu_count}", flush=True)
        print(f"Storage: {MEMORY_DIR}", flush=True)
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"PyTorch threads configured: {torch.get_num_threads()}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

        # Load embedding model on CPU with optimizations
        print(f"\nInitializing embedding model...", flush=True)
        embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device,
            model_kwargs={
                "torch_dtype": torch.float32
            },  # Use float32 for CPU stability
        )

        # Ensure model is in eval mode and on CPU
        embed_model = embed_model.to(device)
        embed_model.eval()  # Disable dropout and batch norm for inference

        print(f"✓ Embedding model loaded on CPU", flush=True)
        print(f"  Model type: {type(embed_model).__name__}", flush=True)
        print(f"  Embedding dimension: 384", flush=True)
        print(f"  Max sequence length: {embed_model.max_seq_length}", flush=True)

        # Connect to LanceDB
        db = lancedb.connect(DB_PATH)
        print(f"✓ Connected to LanceDB at {DB_PATH}", flush=True)

        # Create or open table
        try:
            table = db.open_table("memories")
            row_count = len(table)
            print(
                f"✓ Opened existing memories table ({row_count} memories)", flush=True
            )
        except Exception:
            # Create new table with schema
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), 384)),
                    pa.field("timestamp", pa.string()),
                    pa.field("role", pa.string()),
                    pa.field("conversation_id", pa.string()),
                    pa.field("metadata", pa.string()),
                ]
            )
            table = db.create_table("memories", schema=schema)
            print(f"✓ Created new memories table", flush=True)

        print(f"═" * 60, flush=True)

    except Exception as e:
        print(f"✗ Database initialization error: {e}", flush=True)
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise


def encode_batch(texts, batch_size=32):
    """
    Encode multiple texts in batches for efficiency

    Args:
        texts: List of strings to encode
        batch_size: Number of texts to process at once

    Returns:
        List of embeddings
    """
    with torch.no_grad():
        return embed_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
        )


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools"""
    return [
        Tool(
            name="store_memory",
            description="Store a conversation turn in long-term memory with semantic embedding",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to store"},
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Who said this",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Optional conversation ID",
                        "default": "default",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata",
                        "default": {},
                    },
                },
                "required": ["text", "role"],
            },
        ),
        Tool(
            name="search_memory",
            description="Search long-term memory using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "role_filter": {
                        "type": "string",
                        "enum": ["user", "assistant", "both"],
                        "description": "Filter by role",
                        "default": "both",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score",
                        "default": 0.3,
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_recent_memories",
            description="Get the most recent conversation turns",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent memories",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "Optional conversation filter",
                    },
                },
            },
        ),
        Tool(
            name="delete_memory",
            description="Delete a specific memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to delete",
                    }
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="update_memory",
            description="Update an existing memory's text and/or metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to update",
                    },
                    "text": {
                        "type": "string",
                        "description": "New text content (optional)",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "New metadata (optional)",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="clear_all_memories",
            description="Clear all stored memories (WARNING: irreversible)",
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm",
                    }
                },
                "required": ["confirm"],
            },
        ),
        Tool(
            name="get_memory_stats",
            description="Get statistics about stored memories",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    global table, db

    try:
        if name == "store_memory":
            text = arguments["text"]
            role = arguments["role"]
            conversation_id = arguments.get("conversation_id", "default")
            metadata = arguments.get("metadata", {})

            # Generate embedding with CPU optimization
            # torch.no_grad() reduces memory and speeds up inference on CPU
            with torch.no_grad():
                embedding = embed_model.encode(
                    [text], convert_to_tensor=False, show_progress_bar=False
                )[0].tolist()

            # Create record
            record = {
                "id": f"{datetime.now().timestamp()}_{role}",
                "text": text,
                "vector": embedding,
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "conversation_id": conversation_id,
                "metadata": json.dumps(metadata),
            }

            # Store in LanceDB
            table.add([record])

            return [
                TextContent(
                    type="text", text=f"✓ Stored {role} memory ({len(text)} chars)"
                )
            ]

        elif name == "search_memory":
            query = arguments["query"]
            top_k = arguments.get("top_k", 5)
            role_filter = arguments.get("role_filter", "both")
            min_score = arguments.get("min_score", 0.3)

            # Generate query embedding with CPU optimization
            with torch.no_grad():
                query_embedding = embed_model.encode(
                    [query], convert_to_tensor=False, show_progress_bar=False
                )[0].tolist()

            # Search in LanceDB with oversampling for filtering
            results = table.search(query_embedding).limit(top_k * 2).to_list()

            # Filter and format results
            filtered = []
            for r in results:
                # LanceDB returns _distance (Euclidean by default)
                # Convert to similarity score (0-1 range)
                # Using inverse distance formula: closer = higher score
                score = 1 / (1 + r.get("_distance", 1))

                if score < min_score:
                    continue
                if role_filter != "both" and r["role"] != role_filter:
                    continue

                filtered.append(
                    {
                        "id": r.get("id", "unknown"),
                        "text": r["text"],
                        "role": r["role"],
                        "timestamp": r["timestamp"],
                        "score": round(score, 3),
                        "conversation_id": r.get("conversation_id", "unknown"),
                    }
                )

                if len(filtered) >= top_k:
                    break

            if not filtered:
                return [TextContent(type="text", text="No relevant memories found.")]

            result_text = f"Found {len(filtered)} relevant memories:\n\n"
            for i, mem in enumerate(filtered, 1):
                preview = (
                    mem["text"][:150] + "..." if len(mem["text"]) > 150 else mem["text"]
                )
                result_text += f"{i}. [{mem['role']}] (score: {mem['score']})\n"
                result_text += f"   Conversation: {mem['conversation_id']}\n"
                result_text += f"   {preview}\n\n"

            return [TextContent(type="text", text=result_text)]

        elif name == "get_recent_memories":
            limit = arguments.get("limit", 10)
            conversation_id = arguments.get("conversation_id")

            # Convert table to pandas for easier manipulation
            df = table.to_pandas()

            # Sort by timestamp (most recent first)
            df = df.sort_values("timestamp", ascending=False)

            # Filter by conversation_id if provided
            if conversation_id:
                df = df[df["conversation_id"] == conversation_id]

            # Get top N records
            recent = df.head(limit)

            if len(recent) == 0:
                return [TextContent(type="text", text="No memories stored yet.")]

            result_text = f"Most recent {len(recent)} memories:\n\n"
            for idx, row in recent.iterrows():
                preview = (
                    row["text"][:150] + "..." if len(row["text"]) > 150 else row["text"]
                )
                result_text += f"ID: {row['id']}\n"
                result_text += f"[{row['role']}] {row['timestamp']}\n"
                result_text += f"Conversation: {row['conversation_id']}\n"
                result_text += f"{preview}\n\n"

            return [TextContent(type="text", text=result_text)]

        elif name == "delete_memory":
            memory_id = arguments["memory_id"]

            # Convert table to pandas for filtering
            df = table.to_pandas()

            # Check if memory exists
            if memory_id not in df["id"].values:
                return [
                    TextContent(
                        type="text", text=f"❌ Memory with ID '{memory_id}' not found."
                    )
                ]

            # Filter out the memory to delete
            df_filtered = df[df["id"] != memory_id]

            # Drop and recreate table with filtered data
            db.drop_table("memories")

            # Recreate schema
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), 384)),
                    pa.field("timestamp", pa.string()),
                    pa.field("role", pa.string()),
                    pa.field("conversation_id", pa.string()),
                    pa.field("metadata", pa.string()),
                ]
            )

            table = db.create_table("memories", schema=schema)

            # Re-add all memories except the deleted one
            if len(df_filtered) > 0:
                records = df_filtered.to_dict("records")
                table.add(records)

            return [TextContent(type="text", text=f"✓ Deleted memory '{memory_id}'")]

        elif name == "update_memory":
            memory_id = arguments["memory_id"]
            new_text = arguments.get("text")
            new_metadata = arguments.get("metadata")

            if not new_text and not new_metadata:
                return [
                    TextContent(
                        type="text",
                        text="❌ Must provide either 'text' or 'metadata' to update.",
                    )
                ]

            # Convert table to pandas for filtering
            df = table.to_pandas()

            # Check if memory exists
            if memory_id not in df["id"].values:
                return [
                    TextContent(
                        type="text", text=f"❌ Memory with ID '{memory_id}' not found."
                    )
                ]

            # Get the existing memory
            existing = df[df["id"] == memory_id].iloc[0]

            # Update fields
            updated_text = new_text if new_text else existing["text"]
            updated_metadata = (
                json.dumps(new_metadata) if new_metadata else existing["metadata"]
            )

            # Regenerate embedding if text changed
            if new_text:
                with torch.no_grad():
                    updated_vector = embed_model.encode(
                        [updated_text], convert_to_tensor=False, show_progress_bar=False
                    )[0].tolist()
            else:
                updated_vector = existing["vector"]

            # Create updated record
            updated_record = {
                "id": memory_id,
                "text": updated_text,
                "vector": updated_vector,
                "timestamp": existing["timestamp"],  # Keep original timestamp
                "role": existing["role"],
                "conversation_id": existing["conversation_id"],
                "metadata": updated_metadata,
            }

            # Filter out the old memory
            df_filtered = df[df["id"] != memory_id]

            # Drop and recreate table
            db.drop_table("memories")

            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), 384)),
                    pa.field("timestamp", pa.string()),
                    pa.field("role", pa.string()),
                    pa.field("conversation_id", pa.string()),
                    pa.field("metadata", pa.string()),
                ]
            )

            table = db.create_table("memories", schema=schema)

            # Re-add all memories
            if len(df_filtered) > 0:
                records = df_filtered.to_dict("records")
                table.add(records)

            # Add updated memory
            table.add([updated_record])

            changes = []
            if new_text:
                changes.append("text")
            if new_metadata:
                changes.append("metadata")

            return [
                TextContent(
                    type="text",
                    text=f"✓ Updated memory '{memory_id}' ({', '.join(changes)} changed)",
                )
            ]

        elif name == "clear_all_memories":
            if not arguments.get("confirm"):
                return [
                    TextContent(
                        type="text",
                        text="⚠️ Confirmation required. Set 'confirm: true' to delete all memories.",
                    )
                ]

            # Drop and recreate table
            db.drop_table("memories")
            init_database()

            return [TextContent(type="text", text="✓ All memories cleared.")]

        elif name == "get_memory_stats":
            df = table.to_pandas()

            if len(df) == 0:
                return [TextContent(type="text", text="No memories stored yet.")]

            # Calculate statistics
            stats = {
                "Total memories": len(df),
                "User messages": len(df[df["role"] == "user"]),
                "Assistant messages": len(df[df["role"] == "assistant"]),
                "Unique conversations": df["conversation_id"].nunique(),
                "Oldest memory": df["timestamp"].min(),
                "Newest memory": df["timestamp"].max(),
                "Average text length": int(df["text"].str.len().mean()),
                "Total storage (chars)": df["text"].str.len().sum(),
            }

            result_text = "📊 Memory Statistics:\n\n"
            for key, value in stats.items():
                result_text += f"{key}: {value}\n"

            # Add conversation breakdown
            result_text += f"\n📝 Conversations:\n"
            conv_counts = df["conversation_id"].value_counts()
            for conv_id, count in conv_counts.head(5).items():
                result_text += f"  {conv_id}: {count} memories\n"

            return [TextContent(type="text", text=result_text)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        print(f"✗ Tool execution error: {e}", flush=True)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point"""
    print("Starting LanceDB Memory MCP Server...", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Platform: {platform.system()} {platform.release()}", flush=True)

    try:
        # Initialize database
        init_database()
        print("✓ Initialization complete", flush=True)

        # Run server
        async with stdio_server() as (read_stream, write_stream):
            print("✓ Server running and ready for connections", flush=True)
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    except Exception as e:
        print(f"✗ Fatal error: {e}", flush=True)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user", flush=True)
    except Exception as e:
        print(f"✗ Startup error: {e}", flush=True)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
