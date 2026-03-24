"""
Talk2EP Demo — Configuration and path setup.

Loads environment variables, configures LLM providers, and sets up
sys.path so we can import from the SLIDERS submodule.
EnergyPlus-MCP is accessed via the MCP protocol (not direct import).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEMO_DIR = Path(__file__).resolve().parent
ROOT_DIR = DEMO_DIR.parent
MCP_SERVER_DIR = ROOT_DIR / "EnergyPlus-MCP" / "energyplus-mcp-server"
SLIDERS_DIR = ROOT_DIR / "sliders"
SAMPLE_FILES_DIR = MCP_SERVER_DIR / "sample_files"

# ---------------------------------------------------------------------------
# Add SLIDERS to sys.path (so we can `import sliders`).
# EnergyPlus-MCP is NOT imported — it runs as a subprocess via MCP.
# ---------------------------------------------------------------------------
_sliders_path = str(SLIDERS_DIR)
if _sliders_path not in sys.path:
    sys.path.insert(0, _sliders_path)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# Load .env from demo/ first, fall back to sliders/.env for Azure creds
load_dotenv(DEMO_DIR / ".env")
load_dotenv(SLIDERS_DIR / ".env", override=False)

# LLM provider for the orchestrator agent
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "vertex_ai")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-pro")

# EnergyPlus
EPLUS_IDD_PATH: str = os.getenv("EPLUS_IDD_PATH", "")

# SLIDERS knowledge base document path
EP_DOCS_PATH: str = os.getenv("EP_DOCS_PATH", "")


def get_orchestrator_llm():
    """Return a LangChain chat model for the orchestrator based on LLM_PROVIDER."""
    if LLM_PROVIDER == "vertex_ai":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model_name=LLM_MODEL,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            temperature=0.0,
            max_output_tokens=4096,
        )
    elif LLM_PROVIDER == "azure_openai":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZARE_URL_ENDPOINT"),
            api_version="2024-12-01-preview",
            temperature=0.0,
            max_tokens=4096,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=4096,
        )
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=LLM_MODEL,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
            max_tokens=4096,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
