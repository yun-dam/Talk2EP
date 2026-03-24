"""
Talk2EP Demo — MCP client connection manager.

Spawns the EnergyPlus-MCP server as a subprocess via stdio transport,
establishes a ClientSession, and provides call_tool() for the agent tools.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_SERVER_DIR, EPLUS_IDD_PATH, DEMO_DIR


def _server_params() -> StdioServerParameters:
    """Build the parameters to spawn the EnergyPlus-MCP server."""
    env = {**os.environ}
    if EPLUS_IDD_PATH:
        env["EPLUS_IDD_PATH"] = EPLUS_IDD_PATH
    # Point workspace_root at the actual MCP server directory (not Docker default)
    env["EPLUS_WORKSPACE_ROOT"] = str(MCP_SERVER_DIR)

    # Use the wrapper script that suppresses import-time stdout noise
    # (eppy/graphviz print warnings that corrupt MCP's JSON-RPC protocol)
    wrapper = str(DEMO_DIR / "start_mcp_server.py")

    return StdioServerParameters(
        command="uv",
        args=["run", "python", wrapper],
        cwd=str(MCP_SERVER_DIR),
        env=env,
    )


class MCPClient:
    """Manages the lifecycle of an MCP server subprocess + client session."""

    def __init__(self):
        self._session: ClientSession | None = None
        self._cm_stdio = None
        self._cm_session = None
        self._tools: dict[str, dict] = {}  # name → schema

    @property
    def connected(self) -> bool:
        return self._session is not None

    async def connect(self) -> None:
        """Start the MCP server and initialise the session."""
        if self._session is not None:
            return

        params = _server_params()

        # Open the stdio transport (starts the server subprocess)
        self._cm_stdio = stdio_client(params)
        read, write = await self._cm_stdio.__aenter__()

        # Open the client session
        self._cm_session = ClientSession(read, write)
        self._session = await self._cm_session.__aenter__()

        # MCP handshake
        await self._session.initialize()

        # Cache the tool list
        result = await self._session.list_tools()
        self._tools = {t.name: t.inputSchema for t in result.tools}

    async def disconnect(self) -> None:
        """Shut down the session and server subprocess."""
        if self._cm_session is not None:
            await self._cm_session.__aexit__(None, None, None)
        if self._cm_stdio is not None:
            await self._cm_stdio.__aexit__(None, None, None)
        self._session = None
        self._cm_session = None
        self._cm_stdio = None
        self._tools = {}

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call an MCP tool and return the text result."""
        if self._session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        result = await self._session.call_tool(name, arguments or {})

        # Collect all text content blocks into a single string
        parts: list[str] = []
        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)

        text = "\n".join(parts)

        if result.isError:
            raise RuntimeError(f"MCP tool '{name}' failed: {text}")

        return text


# Module-level singleton so tools.py can reference it.
mcp_client = MCPClient()
