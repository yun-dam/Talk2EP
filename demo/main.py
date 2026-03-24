#!/usr/bin/env python
"""
Talk2EP Demo — Interactive CLI.

Usage:
    conda activate talk2ep
    cd demo
    python main.py                                              # interactive REPL
    python main.py --query "list the sample IDF files"          # one-shot
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Ensure demo/ is on sys.path so `import config` works
sys.path.insert(0, str(Path(__file__).resolve().parent))

console = Console(force_terminal=True)


# ── Pretty-print helpers ──────────────────────────────────────────────────

def print_banner():
    console.print(
        Panel(
            "[bold cyan]Talk2EP[/bold cyan]  —  AI Co-Pilot for EnergyPlus\n"
            "[dim]Natural-language interface powered by SLIDERS RAG + EnergyPlus-MCP[/dim]",
            border_style="cyan",
        )
    )
    console.print("[dim]Type your request, or 'quit' / Ctrl-C to exit.[/dim]\n")


def print_tool_call(name: str, args: dict):
    args_short = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(args_short) > 120:
        args_short = args_short[:117] + "..."
    console.print(f"  [yellow]tool:[/yellow] {name}({args_short})")


def print_agent_message(text: str):
    console.print()
    console.print(Markdown(text))
    console.print()


# ── Agent invocation ──────────────────────────────────────────────────────

async def run_agent_query(agent, query: str, config: dict):
    """Stream the agent's response, printing tool calls and final answer."""
    console.print(f"\n[bold green]You:[/bold green] {query}")
    console.print("[dim]--- thinking ---[/dim]")

    inputs = {"messages": [("user", query)]}
    final_text = ""

    try:
        async for event in agent.astream_events(inputs, version="v2", config=config):
            kind = event.get("event", "")
            if kind == "on_tool_start":
                print_tool_call(event["name"], event.get("data", {}).get("input", {}))
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    final_text += chunk.content
    except Exception as e:
        console.print(f"[red]Streaming error: {e}[/red]")
        # Fallback: non-streaming
        try:
            result = await agent.ainvoke(inputs, config=config)
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content and msg.type == "ai":
                    final_text = msg.content
                    break
        except Exception as e2:
            console.print(f"[red]Agent error: {e2}[/red]")
            return

    if final_text:
        print_agent_message(final_text)


# ── Conversation loop ─────────────────────────────────────────────────────

async def repl(agent, config: dict):
    """Read-Eval-Print loop with conversation history (managed by checkpointer)."""
    print_banner()

    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # Only send the new user message — the checkpointer retains history
        inputs = {"messages": [("user", query)]}

        console.print("[dim]--- thinking ---[/dim]")
        final_text = ""

        try:
            async for event in agent.astream_events(inputs, version="v2", config=config):
                kind = event.get("event", "")
                if kind == "on_tool_start":
                    print_tool_call(
                        event["name"],
                        event.get("data", {}).get("input", {}),
                    )
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        final_text += chunk.content
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            try:
                result = await agent.ainvoke(inputs, config=config)
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "content") and msg.content and msg.type == "ai":
                        final_text = msg.content
                        break
            except Exception as e2:
                console.print(f"[red]Agent error: {e2}[/red]")
                continue

        if final_text:
            print_agent_message(final_text)


# ── Entry point ───────────────────────────────────────────────────────────

async def async_main(query: str | None = None):
    from mcp_client import mcp_client
    from agent import build_agent

    # 1. Start the MCP server and connect
    console.print("[dim]Connecting to EnergyPlus-MCP server...[/dim]")
    try:
        await mcp_client.connect()
    except Exception as e:
        console.print(f"[red]Failed to connect to MCP server: {e}[/red]")
        console.print("[dim]Make sure EPLUS_IDD_PATH is set and uv dependencies are installed in EnergyPlus-MCP.[/dim]")
        return

    tools = mcp_client.list_tools()
    console.print(f"[green]MCP server connected — {len(tools)} tools available.[/green]")

    # 2. Build the LangGraph agent
    console.print("[dim]Initialising agent...[/dim]")
    agent = build_agent()
    console.print("[green]Agent ready.[/green]\n")

    # Thread config — keeps conversation state across turns via checkpointer
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}

    # 3. Run
    try:
        if query:
            await run_agent_query(agent, query, config)
        else:
            await repl(agent, config)
    finally:
        # 4. Clean shutdown
        console.print("[dim]Disconnecting MCP server...[/dim]")
        await mcp_client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Talk2EP — AI Co-Pilot for EnergyPlus")
    parser.add_argument("--query", "-q", type=str, help="One-shot query (skip REPL)")
    args = parser.parse_args()
    asyncio.run(async_main(args.query))


if __name__ == "__main__":
    main()
