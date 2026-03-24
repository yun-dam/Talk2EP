"""
Talk2EP Demo — SLIDERS-based knowledge layer for EnergyPlus domain knowledge.

Wraps the SLIDERS RAG pipeline so the orchestrator agent can query the
EnergyPlus Engineering Reference for building-physics parameters, ASHRAE
defaults, and simulation best-practices.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from config import EP_DOCS_PATH, SLIDERS_DIR

# ── Model name mapping per provider ──────────────────────────────────────
_PROVIDER = os.getenv("SLIDERS_LLM_PROVIDER", "vertex_ai")

_MODEL_MAP = {
    "vertex_ai": {
        "strong":  "gemini-2.5-pro",
        "light":   "gemini-2.5-flash",
    },
    "azure_openai": {
        "strong":  "gpt-4.1",
        "light":   "gpt-4.1-mini",
    },
}

def _m(tier: str) -> str:
    """Return the model name for the current provider and tier."""
    return _MODEL_MAP.get(_PROVIDER, _MODEL_MAP["vertex_ai"])[tier]


class EnergyPlusKnowledge:
    """Lazy-initialised wrapper around SlidersAgent for EP domain queries."""

    @staticmethod
    def _build_config() -> dict:
        """Build SLIDERS config with model names for the active LLM provider."""
        strong, light = _m("strong"), _m("light")
        return {
            "generate_task_guidelines": False,
            "generate_schema": {"add_extra_information_class": False},
            "extract_schema": {
                "decompose_fields": False,
                "dedupe_merged_rows": False,
                "num_samples_per_chunk": 1,
            },
            "merge_tables": {"merge_strategy": "seq_agent"},
            "models": {
                "answer":                   {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "answer_no_table":          {"model": light,  "max_tokens": 8192, "temperature": 0.0},
                "answer_tool_output":       {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "extract_schema":           {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "derive_atomic_fields":     {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "generate_schema":          {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "merge_tables":             {"model": strong, "max_tokens": 8192, "temperature": 0.2},
                "task_guidelines":          {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "check_objective_necessity":{"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "direct_answer":            {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "force_answer":             {"model": strong, "max_tokens": 8192, "temperature": 0.0},
                "is_relevant_chunk":        {"model": light,  "max_tokens": 8192, "temperature": 0.0},
                "check_if_merge_needed":    {"model": strong, "max_tokens": 8192, "temperature": 0.0},
            },
        }

    def __init__(self, docs_path: Optional[str] = None):
        self._docs_path = docs_path or EP_DOCS_PATH
        self._system = None
        self._documents: list = []
        self._initialized = False
        self._available = bool(self._docs_path and Path(self._docs_path).exists())

    @property
    def available(self) -> bool:
        return self._available

    async def initialize(self) -> None:
        """Load documents and create the SLIDERS agent (called once)."""
        if self._initialized:
            return

        if not self._available:
            self._initialized = True
            return

        # Import SLIDERS (globals.py runs init_llm on import)
        from sliders.globals import SlidersGlobal  # noqa: F401 — triggers prompt dir init
        from sliders.system import SlidersAgent
        from sliders.document import Document

        self._system = SlidersAgent(config=self._build_config())

        docs_path = Path(self._docs_path)
        if docs_path.is_file():
            doc = await Document.from_file_path(
                str(docs_path),
                description="EnergyPlus Engineering Reference — building energy simulation documentation",
                chunk_size=16000,
                overlap_size=0,
            )
            self._documents = [doc]
        elif docs_path.is_dir():
            # Load all .md / .txt files in the directory
            files = sorted(docs_path.glob("*.md")) + sorted(docs_path.glob("*.txt"))
            for f in files:
                doc = await Document.from_file_path(
                    str(f),
                    description=f"EnergyPlus reference document: {f.stem}",
                    chunk_size=16000,
                    overlap_size=0,
                )
                self._documents.append(doc)

        self._initialized = True

    async def query(self, question: str) -> str:
        """Ask a building-physics / EnergyPlus question via SLIDERS RAG."""
        if not self._initialized:
            await self.initialize()

        if not self._system or not self._documents:
            return (
                "[Knowledge base unavailable] "
                "Set EP_DOCS_PATH to the EnergyPlus Engineering Reference "
                "(markdown or directory of .md/.txt files) to enable RAG queries."
            )

        answer, _metadata = await self._system.run(
            question=question,
            documents=self._documents,
            question_id=f"talk2ep_{hash(question) & 0xFFFFFFFF:08x}",
        )
        return answer
