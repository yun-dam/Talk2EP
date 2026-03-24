## Development Notes

### Environment and Credentials

Create `.env` at the repository root:

```bash
AZURE_OPENAI_API_KEY=...
AZARE_URL_ENDPOINT=...
```

`sliders/globals.py` loads this automatically and sets up the prompt environment.

### Running Locally

```bash
uv venv && uv sync
export SLIDERS_RESULTS="$(pwd)/results" && mkdir -p "$SLIDERS_RESULTS"
uv run sliders/runner.py --config configs/finance_bench_sliders_agent.yaml
```

### Logging

All LLM calls are routed through chains with `LoggingHandler` callbacks. This captures:
- Prompt templates and variables
- Tool calls and outputs (e.g., SQL)
- Per-stage metadata (schema, extraction, merging, answering)

### Caching

If temperature is `0.0`, the LLM wrapper (`sliders/llm/llm.py`) uses Redis (if reachable at `localhost:6379`) to cache exact calls keyed by prompt + model + structured output schema.

To enable:

```bash
sudo apt-get install redis-server
sudo service redis-server start
```

### Results

Results are JSON files containing:
- `results`: evaluation tool outputs for each question
- `all_metadata`: detailed run metadata (timings, table stats, errors)
- `results_summary`: accuracy per evaluation tool

### Adding a New Merge Strategy

1. Implement under `sliders/modules/merge_techniques/<strategy>.py`
2. Register it in `merge_schema.py` if needed
3. Select via `system_config.merge_tables.merge_strategy`

### Adding a New System

1. Subclass `System` in `sliders/system.py`
2. Implement `_setup_chains` and `run`
3. Add to `SYSTEM_REGISTRY` in `sliders/runner.py`

### Adding a New Experiment

1. Subclass `Experiment` in `sliders/experiments/base.py`
2. Implement `_run_row` and `run`
3. Add to `EXPERIMENT_REGISTRY` in `sliders/runner.py`

