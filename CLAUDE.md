# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e .

# Run the full integration test suite (mock LLM, no API key needed — 7 tests)
python test_integration.py

# Bundled end-to-end demo on synthetic data, no API key
python -m midas demo
python -m midas demo --provider openai           # uses $OPENAI_API_KEY
python -m midas demo --provider anthropic        # uses $ANTHROPIC_API_KEY
python -m midas demo --report-path ./artifacts/demo_report.md

# CLI subcommands (all accept --kb <path>, default ./midas-kb)
python -m midas status
python -m midas promote <feature_name>
python -m midas demote  <feature_name> --reason "..."
python -m midas reject  <feature_name> --reason "..."
python -m midas learnings --n 5
```

The console entrypoint `midas = "midas.factory:_cli"` is equivalent to `python -m midas`.

There is no separate lint/format config and no pytest setup — `test_integration.py` is a self-contained script. To run a single test, edit the `tests = [...]` list near the bottom of `test_integration.py`.

## Architecture

Midas is a **dual-loop compound-engineering framework** for crypto alpha-feature research. Both loops share a single filesystem-backed knowledge base; every run reads prior learnings and writes new ones.

### Loop 1 — Offline discovery (`midas/loops.py`)
`OfflineCompoundLoop.run()` executes Plan → Write → Assess → Learn → (Refine) per iteration:
- **Plan/Write/Refine** is delegated to `ExpressionProposer` (`midas/proposer.py`), which calls the LLM via `midas/llm.py` using prompt templates seeded in `<kb>/proposer/prompts/`. `DSLValidator` rejects malformed candidates *before* any LLM call to save tokens.
- **Assess** runs `MultiAgentEvaluator` (`midas/evaluator.py`) — six agents in a `ThreadPoolExecutor` covering predictive_power, decay_analysis, trading_cost, diversification, overfit_detection, regime_robustness. Each returns a verdict + 0..1 score + suggestions; suggestions feed the next Refine prompt.
- **Learn** writes a `LearningDocument` to `knowledge/learnings/offline/` and updates feature state markdown.

Caller supplies `data_fn() -> (compute_fn, forward_returns_df, regime_series)`; `compute_fn(expression: str) -> pd.Series` is the integration seam for any external feature engine.

### Loop 2 — Online monitoring (`midas/monitor.py`)
`OnlineMonitor.process_update()` is called per bar. Internally:
- `MonitorEngine` maintains rolling buffers + `FeatureMetrics`.
- `AlertEngine` applies threshold rules (IC decay, slippage, drawdown — see README).
- **Critical** alerts fire `DiagnoseAgent` asynchronously; it writes a learning doc, a fix proposal, and a kill signal to `reports/diagnoses/` and invokes the `on_kill` callback.
- `generate_daily_report(date)` produces `reports/daily/<date>.md`.

### Glue layer
- `midas/factory.py` — `create_midas()` bootstraps the KB tree, seeds skills/prompts/thresholds, and returns a `Midas` container exposing `.offline`, `.promoter`, `.build_online()`. The CLI (`_cli`) lives here.
- `midas/kb.py` — **all** disk I/O goes through `KnowledgeBase`. On first run it seeds `skills/midas-dsl.md`, `skills/factor-patterns.md`, the three prompt templates, and `thresholds.json`. Editing those files in-place changes behaviour on next loop run.
- `midas/promoter.py` — `FeaturePromoter` moves feature markdown between `knowledge/features/{candidates,deployed,archived}/`. `promote/demote/reject` CLI subcommands are thin wrappers.
- `midas/models.py` — all shared dataclasses (`EvaluationResult`, `MultiAgentResult`, `Alert`, `FeatureMetrics`, `LearningDocument`, `DailyReport`, `DiagnoseResult`). Touch this first when changing cross-module data shapes.
- `midas/llm.py` — single provider-resolution boundary; supports `openai` and `anthropic`. Tests substitute a `MockLLM` here.

### Knowledge base (`<kb>/`)
Two example KB directories ship with the repo as **documentation artifacts, not production data**:
- `midas-kb/` — fresh demo KB
- `demo_artifacts/midas-kb/` and `demo_artifacts/midas-kb-online/` — sample offline/online outputs

Layout: `skills/`, `knowledge/{features/{deployed,candidates,archived},learnings/{offline,online},regimes}`, `knowledge/thresholds.json`, `proposer/prompts/{plan,generate,refine}.md`, `reports/{daily,diagnoses}/`.

## Conventions

- Python 3.11+. Dependencies are pinned in `pyproject.toml`; no separate requirements file.
- The integration test patches `midas.llm` with `MockLLM` — when adding new LLM call sites, route them through `midas/llm.py` so tests keep working without API keys.
- Prompt templates and thresholds are *runtime-editable* via the KB. Don't hard-code values that already live in `<kb>/proposer/prompts/` or `<kb>/knowledge/thresholds.json`.
- The DSL syntax accepted by `DSLValidator` is documented in the seeded `skills/midas-dsl.md` — treat that file as the spec.
