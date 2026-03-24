## Architecture and Components

SLIDERS is organized around configurable systems that operate over chunked documents to extract structured tables and answer questions.

### High-level flow (SlidersAgent)

1. Generate schema for the question and documents
2. Extract tables by applying the schema to document chunks
3. Merge tables across chunks and documents (required reconciliation)
4. Answer the question directly or via SQL execution over DuckDB
5. Summarize results and metrics

### Key Modules

- `sliders/system.py`
  - `System`: abstract base class for all systems
  - `SlidersAgent`: orchestrates schema generation, extraction, merging, and answering

- `sliders/modules/`
  - `generate_schema.py`: proposes classes/fields relevant to the question
  - `extract_schema.py`: extracts instances/tables from chunked documents
  - `merge_schema.py`: merges tables; strategy implementations in `modules/merge_techniques/`

Note: Reconciliation/merging is a crucial, non-optional step of the SLIDERS pipeline. Independent chunk-level extractions often contain redundant, partial, or conflicting values that cannot be reliably combined via prompt-only reasoning. SLIDERS employs an LLM-based reconciliation agent that generates declarative SQL over entity tables to normalize and consolidate results, forming the structured foundation for downstream reasoning and transparent user queries.

- `sliders/llm/`
  - `llm.py`: Azure OpenAI client wrapper, Redis caching, rate limiting
  - `prompts.py`: Jinja2-based prompt block loader and few-shot assembly

- `sliders/llm_tools/`
  - `sql.py`: DuckDB SQL execution utilities
  - `code.py`: Python tool for ReAct agents

- `sliders/experiments/`
  - `base.py`: experiment interface
  - `finance_bench.py`, `loong.py`, `babilong.py`: dataset-specific runners and evaluators

- `sliders/document.py`
  - `Document`: loading and chunking logic (markdown, etc.)
  - `contextualize_document_metadata`: enrich metadata using the question

- `sliders/baselines.py`
  - `LLMWithoutToolUseSystem`: direct prompting
  - `LLMWithToolUseSystem`: ReAct agent with tools
  - `LLMSequentialSystem`: chunk-by-chunk early stopping

### Prompt Templates

All templates are in `sliders/prompts/`, organized by purpose:
- `sliders/`: agent prompts for answer generation, SQL, merging, etc.
- `baselines/`: prompts for direct answering or tool-use baselines
- `evaluators/`: prompts for LLM-as-judge evaluation

Templates are loaded via `sliders/llm/prompts.py` using Jinja2 and a small block DSL for instruction/input/output.

### Execution and Results

- Entry: `sliders/runner.py`
  - Parses the YAML config
  - Builds the experiment and system
  - Runs questions sequentially or in parallel
  - Writes results to `$SLIDERS_RESULTS/<file>_<timestamp>.json`

- Results JSON includes:
  - `results`: per-question evaluation tool outputs
  - `all_metadata`: detailed pipeline metadata per question
  - `results_summary`: aggregated accuracies per evaluation tool

### Extending SLIDERS

- Add a new merge strategy: implement under `modules/merge_techniques/` and wire via `merge_strategy`
- Add a new system: subclass `System` and implement `_setup_chains` and `run`
- Add experiments: implement an `Experiment` in `experiments/` and register in `runner.py`

