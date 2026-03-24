## Experiments Guide

This guide covers the built-in benchmarks and how to run them.

All runs are launched via:

```bash
uv run sliders/runner.py --config <path-to-config.yaml> [--parallel]
```

Make sure to define and create a results directory:

```bash
export SLIDERS_RESULTS="$(pwd)/results"
mkdir -p "$SLIDERS_RESULTS"
```

---

## FinanceBench

Config examples:
- `configs/finance_bench_sliders_agent.yaml`: SLIDERS agent on FinanceBench
- `configs/finance_bench_sliders_sample.yaml`: quick sample with 1 question

Edit the following paths:

```yaml
experiment_config:
  benchmark_path: /path/to/financebench_open_source.jsonl
  files_dir: /path/to/financebench/markdown/pdfs/
  gpt_results_path: /path/to/gpt-4_oracle_reverse.jsonl   # optional
```

Run:

```bash
uv run sliders/runner.py --config configs/finance_bench_sliders_agent.yaml
```

---

## Loong

Config: `configs/loong_sliders.yaml`

Paths:

```yaml
experiment_config:
  benchmark_path: /path/to/loong_processed.jsonl
  files_dir: /path/to/loong/doc/
  # optional filters
  filter_by_type: financial
  # filter_by_level: 3
```

Run:

```bash
uv run sliders/runner.py --config configs/loong_sliders.yaml --parallel
```

---

## BabiLong

`runner.py` includes an example `run_babilong_loong_sliders` for grid runs over token lengths and QA types. Create or point to `babilong_data` files before running.

You can create your own config similar to FinanceBench/Loong and provide:

```yaml
experiment: babilong
experiment_config:
  benchmark_path: /path/to/babilong_<qa>_<token_len>.json
  num_questions: 10
```

---

## Choosing a System

Set `system` in the YAML:
- `sliders`: full pipeline (recommended)
- `direct_no_tool_use`: single LLM prompt
- `direct_tool_use`: ReAct agent with tools
- `sequential`: chunk-by-chunk stopping
- `rlm`: experimental REPL-based agent

---

## Practical Tips

- Start with small samples using `num_questions` to validate setup
- Use `--parallel` to improve throughput (watch provider rate limits)
- Enable `check_if_merge_needed` to reduce unnecessary merging work
- If SQL frequently errors, try `force_sql: false` to allow direct answers

---

## Use your own dataset

There are two easy paths:

### A) Reuse the FinanceBench experiment (no code changes)

Provide your data as a JSONL with the following fields and a directory of markdown files:

```json
{"financebench_id": 1, "question": "What is the cash balance?", "answer": "123", "doc_name": "acme_2020_10k", "evidence": "...optional..."}
{"financebench_id": 2, "question": "What is revenue in 2021?", "answer": null, "doc_name": "acme_2021_10k", "evidence": null}
```

Requirements:
- `doc_name` must match a `.md` file in your `files_dir` (e.g., `acme_2020_10k.md`).
- `answer` and `evidence` can be `null` if you only need predictions.

Then point a config to your paths:

```yaml
experiment: finance_bench
system: sliders
system_config:
  merge_tables:
    merge_strategy: seq_agent
  models:
    answer: { model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
    answer_no_table: { model: gpt-4.1-mini, max_tokens: 8192, temperature: 0.0 }
    answer_tool_output: { model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
    extract_schema: { model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
    generate_schema: { model: gpt-4.1, max_tokens: 8192, temperature: 0.0 }
    merge_tables: { model: gpt-4.1, max_tokens: 8192, temperature: 0.2 }
experiment_config:
  benchmark_path: /path/to/your_dataset.jsonl
  files_dir: /path/to/your_markdown_dir/
  soft_evaluator_model: gpt-4.1
  hard_evaluator_model: gpt-4.1
  num_questions: 50           # or null for all
  random_state: 42
  document_config: { chunk_size: 16000, overlap_size: 0 }
output_file: your_dataset_sliders.json
```

Run it:

```bash
export SLIDERS_RESULTS="$(pwd)/results" && mkdir -p "$SLIDERS_RESULTS"
uv run sliders/runner.py --config configs/your_dataset.yaml --parallel
```

Convert PDFs to markdown with the provided helper if needed:

```bash
uv run scripts/pdf_to_markdown.py --input /path/to/pdfs --output /path/to/your_markdown_dir
```

### B) Create a custom Experiment (advanced)

Implement your own experiment by subclassing `Experiment` and registering it.

```python
from sliders.experiments.base import Experiment
from sliders.document import Document

class MyDataset(Experiment):
    async def _run_row(self, row, system, all_metadata):
        question = row["question"]
        doc_path = f"/path/to/docs/{row['doc_name']}.md"
        document = await Document.from_markdown(doc_path, description="My docs", document_name=row["doc_name"], chunk_size=16000, overlap_size=0)
        answer, metadata = await system.run(question, [document], question_id=row.get("id", ""))
        all_metadata.append(metadata)
        return {"question": question, "predicted_answer": answer, "evaluation_tools": {}}

    async def run(self, system, parallel=False, *_, **__):
        # load your rows and iterate (optionally with asyncio gather if parallel)
        ...
```

Then add it to the registry in `sliders/runner.py` and reference it in your config’s `experiment` field.

