## Configuration Reference

This guide explains all configuration knobs used by SLIDERS. Configs are YAML files consumed by `sliders/runner.py`.

### Top-level keys

- `experiment` (string): which benchmark to run
  - Supported: `finance_bench`, `loong`, `babilong`
- `system` (string): which system to run
  - Supported: `sliders`, `direct_tool_use`, `direct_no_tool_use`, `sequential`, `rlm`
- `system_config` (object): component-level configuration for the system
- `experiment_config` (object): dataset paths and evaluation options
- `output_file` (string): basename for the result JSON written to `SLIDERS_RESULTS`

---

## system: sliders

The SLIDERS agent orchestrates:
1) schema generation → 2) extraction → 3) optional merging → 4) answering

```yaml
system: sliders
system_config:
  generate_task_guidelines: false           # generate internal per-task guidance
  rephrase_question: false                  # optionally rewrite the input question
  perform_merge: true                       # merge tables from multiple chunks/docs
  check_if_merge_needed: false              # let an LLM decide per-table if merging is needed
  force_sql: false                          # always answer via SQL path

  generate_schema:
    add_extra_information_class: false      # include an auxiliary class for notes/metadata

  extract_schema:
    decompose_fields: false                 # split composite fields into atomic fields
    decompose_tables: false                 # split composite tables
    dedupe_merged_rows: false               # drop duplicate rows when merging
    num_samples_per_chunk: 1                # multiple samples per chunk (N>1 increases cost)
    is_relevant_chunk: false                # filter chunks using an LLM

  merge_tables:
    merge_strategy: seq_agent               # one of: seq_agent | objectives_based | simple

  models:                                   # LLM roles and parameters
    answer:           { model: gpt-4.1,       max_tokens: 8192, temperature: 0.0 }
    answer_no_table:  { model: gpt-4.1-mini,  max_tokens: 8192, temperature: 0.0 }
    answer_tool_output:{ model: gpt-4.1,      max_tokens: 8192, temperature: 0.0 }
    extract_schema:   { model: gpt-4.1,       max_tokens: 8192, temperature: 0.0 }
    derive_atomic_fields:{ model: gpt-4.1,    max_tokens: 8192, temperature: 0.0 }
    generate_schema:  { model: gpt-4.1,       max_tokens: 8192, temperature: 0.0 }
    merge_tables:     { model: gpt-4.1,       max_tokens: 8192, temperature: 0.2 }
    task_guidelines:  { model: gpt-4.1,       max_tokens: 8192, temperature: 0.0 }
    check_objective_necessity:{ model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
    direct_answer:    { model: gpt-4.1,       max_tokens: 8192, temperature: 0.0 }
    force_answer:     { model: gpt-4.1,       max_tokens: 8192, temperature: 0.0 }
    is_relevant_chunk:{ model: gpt-4.1-mini,  max_tokens: 8192, temperature: 0.0 }
    check_if_merge_needed:{ model: gpt-4.1,   max_tokens: 8192, temperature: 0.0 }
```

Notes:
- All `model` fields are passed to `sliders/llm/llm.py:get_llm_client`. Azure OpenAI variables are picked from `.env`.
- Temperature 0.0 enables Redis-based caching automatically when Redis is available.

---

## system: direct_no_tool_use

Single-shot direct prompting without tools.

```yaml
system: direct_no_tool_use
system_config:
  models:
    answer: { model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
```

---

## system: direct_tool_use

ReAct-style agent that can call tools like the Python interpreter.

```yaml
system: direct_tool_use
system_config:
  models:
    tool_use: { model: gpt-4.1, max_tokens: 4096, temperature: 0.0 }
```

---

## system: sequential

Chunk-by-chunk answering; stops early when the model declares an answer.

```yaml
system: sequential
system_config:
  models:
    answer:
      model: gpt-4.1
      max_tokens: 4096
      temperature: 0.0
      template_file: baselines/direct_without_tool_use.prompt
```

---

## experiment_config (per benchmark)

Common fields:

```yaml
experiment_config:
  benchmark_path: /path/to/benchmark.jsonl      # required
  files_dir: /path/to/markdown-or-docs/         # required for file-backed benchmarks
  gpt_results_path: /path/to/oracle.jsonl       # optional (used by some evaluators)
  soft_evaluator_model: gpt-4.1                 # LLM-as-judge (soft)
  hard_evaluator_model: gpt-4.1                 # LLM-as-judge (hard)
  num_questions: null                           # int or null for full set
  random_state: 42                               # sampling reproducibility
  document_config:                               # how documents are chunked
    chunk_size: 16000
    overlap_size: 0
  filter_by_type: financial                      # loong only (optional)
  filter_by_level: 3                             # loong only (optional)
```

---

## Output control

- `output_file`: Basename for the output JSON. The runner writes to:
  - `$SLIDERS_RESULTS/<output_file>_<timestamp>.json`

Make sure to set the environment variable before running:

```bash
export SLIDERS_RESULTS="$(pwd)/results"
mkdir -p "$SLIDERS_RESULTS"
```

---

## Tips

- Use `--parallel` to process questions concurrently (I/O and LLM latency overlap).
- Start with smaller `num_questions` to validate your paths and credentials.
- If your tables are often empty, try enabling `perform_merge` and/or increasing chunk size.

