"""Thin wrapper to start the EnergyPlus-MCP server with stdout noise suppressed.

Some dependencies (eppy/graphviz) print warnings to stdout on import,
which corrupts the MCP JSON-RPC protocol.  This wrapper suppresses stdout
during module loading, then hands control to the real server.
"""

import sys
import os
import io

# Temporarily redirect stdout to suppress import-time print noise
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    # This triggers all the noisy imports (eppy, graphviz, etc.)
    from energyplus_mcp_server.server import mcp
finally:
    # Restore stdout so MCP can communicate
    sys.stdout = _real_stdout

if __name__ == "__main__":
    mcp.run(transport="stdio")
