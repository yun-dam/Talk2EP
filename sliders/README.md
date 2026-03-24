# Long Document Sets QA (SLIDERS)

<p align="center">
    <h1 align="center">
        <img src="assets/sliders.png" width=100px>
        <br>
        <b>SLIDERS</b>
    </h1>
</p>
<p align="center">
    Scalable question answering on long documents using a divide-and-conquer pipeline that turns documents into structured tables, performs data reconciliation, and answers via direct reasoning or SQL over DuckDB.
</p>


### Highlights
- Modular systems: `sliders` agent, direct LLM baselines, and sequential chunking
- Pluggable components: schema generation, extraction, merging, and answering
- Reconciliation is required: independent chunk extractions are normalized and consolidated via an LLM-based agent that emits declarative SQL over entity tables; this step is foundational for downstream reasoning and transparent queries
- Config-driven experiments (FinanceBench, Loong, BabiLong)
- Built-in logging, caching, and result summaries

## 1) Installation

```bash
# Requires Python >= 3.11 and uv (https://docs.astral.sh/uv/)
uv venv            # create a virtual environment
uv sync            # install dependencies from pyproject.toml
```

Optional system deps:
- Redis (for zero-temp LLM response caching): `sudo apt-get install redis-server` then `sudo service redis-server start`

## 2) Environment

Set Azure OpenAI credentials (or compatible) in `.env` at the repo root:

```bash
AZURE_OPENAI_API_KEY=...        # required
AZARE_URL_ENDPOINT=...        # e.g., https://<your-endpoint>.openai.azure.com/
```

The library auto-loads `.env` via `sliders/globals.py` and initializes prompt templates.

## 3) Datasets and Paths

The experiment configs reference local dataset paths. Update them to match your environment:
- FinanceBench markdowns and JSONL
- Loong benchmark JSONL and docs directory
- BabiLong generated JSON

See: `configs/*.yaml` and the docs below for details.

## 4) Quick Start

```bash
# Run with a config (recommended)
uv run sliders/runner.py --config configs/finance_bench_sliders_agent.yaml

# Run in parallel (async per-question)
uv run sliders/runner.py --config configs/loong_sliders.yaml --parallel
```

Outputs are written to `SLIDERS_RESULTS` directory. Set it before running, for example:

```bash
export SLIDERS_RESULTS="$(pwd)/results" && mkdir -p "$SLIDERS_RESULTS"
```

## 5) Configuration Overview

Experiments are driven by YAML configs with three main sections:
- `experiment`: which benchmark to run, e.g., `finance_bench | loong | babilong`
- `system`: which system to use, e.g., `sliders | direct_tool_use | direct_no_tool_use | sequential | rlm`
- `system_config`, `experiment_config`, `output_file`: component-level knobs and I/O

Example: `configs/finance_bench_sliders_agent.yaml`

```yaml
experiment: finance_bench
system: sliders
system_config:
  generate_task_guidelines: false
  generate_schema:
    add_extra_information_class: false
  extract_schema:
    decompose_fields: false
    dedupe_merged_rows: false
    num_samples_per_chunk: 1
  merge_tables:
    merge_strategy: seq_agent   # or objectives_based
  models:
    answer: { model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
    # ... other model roles ...
experiment_config:
  benchmark_path: /path/to/financebench.jsonl
  files_dir: /path/to/financebench/markdown/pdfs/
  soft_evaluator_model: gpt-4.1
  hard_evaluator_model: gpt-4.1
  num_questions: null
  random_state: 42
  document_config: { chunk_size: 16000, overlap_size: 0 }
output_file: finance_bench_sliders.json
```

See the full configuration reference in the docs.

## 6) Systems and Components

- `sliders` agent: full pipeline
  - Schema generation (`sliders/modules/generate_schema.py`)
  - Schema-based extraction (`sliders/modules/extract_schema.py`)
  - Table merging (`sliders/modules/merge_schema.py`, strategies in `modules/merge_techniques/`)
  - Answering: direct or SQL over DuckDB

Note: Reconciliation/merging is a crucial, non-optional stage when running the `sliders` system. It consolidates partial, fragmented, or conflicting values into a consistent database-style representation that downstream reasoning depends on.
- Baselines (`sliders/baselines.py`)
  - `direct_no_tool_use`: prompt-only
  - `direct_tool_use`: ReAct-style with tools (e.g., Python)
  - `sequential`: chunk-by-chunk stopping when answer found

## 7) Logging, Caching, Results

- Logging: prompts, tool calls, and stages are logged via `sliders/callbacks/logging.py`.
- Caching: if temperature is `0.0` and Redis is available, identical calls are cached (`sliders/llm/llm.py`).
- Results: a JSON is emitted to `SLIDERS_RESULTS/<output_file>_<timestamp>.json` with per-question metadata and accuracy summary.

## 8) Documentation

- Getting started and configuration reference: see `docs/CONFIG.md`
- Architecture and pipeline details: see `docs/ARCHITECTURE.md`
- Benchmark-specific guidance: see `docs/EXPERIMENTS.md`
- Development notes (env, caching, logging): see `docs/DEVELOPMENT.md`

---

For deeper usage and customization, continue in the docs folder.

To run on your own dataset, see the "Use your own dataset" section in `docs/EXPERIMENTS.md`.

## Citing

If you use SLIDERS in your research or production, please cite the repository (paper coming soon):

```bibtex
@misc{sliders2025,
  title        = {SLIDERS: Scalable Question Answering on Long Documents with Divide-and-Conquer},
  author       = {Harshit Joshi, Jadelynn Dao, Monica S. Lam},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/stanford-oval/sliders}
}
```

*Beware: Documentation is generated using LLMs and may contain errors.*