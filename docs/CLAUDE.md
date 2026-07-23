# CLAUDE.md — agent instructions

## ⚠️ ENVIRONMENT: ONLY use the `quant` conda env. Never `base`, never `.venv`.

Run **every** Python command in this repo with the `quant` conda interpreter:

```
/opt/anaconda3/envs/quant/bin/python
```

Run the production nowcast with:

```bash
cd /Users/Adam/Documents/home/quant/nowcast
set -a; . ./.env; set +a
PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -u code/production/model.py
```

**Do NOT** use the project `.venv` (Python 3.10) or conda `base`. On this Mac (macOS Sequoia) those
environments hit **pathologically slow Python imports** — `import pandas` takes 10–16 minutes and
runs look like they hang (Gatekeeper re-verifying pip-installed adhoc-signed `.so`s on every load).
The `quant` env imports the whole stack in ~3.5 s and has every dependency (`lightgbm` + `fredapi`
added 2026-07-22; `pmdarima` is **not** needed — `AutoARIMA` uses statsmodels).

If you catch yourself trying to fix, warm, or speed up imports in `.venv`/`base`: **stop and switch
to `quant`.** Do not burn time on the slow envs — this cost ~2 hours once already.

Extra notes:
- Never run many Python processes in parallel here — they serialize on the macOS security scanner
  and each import balloons to many minutes (numpy went 0.15 s → 818 s under that contention).
- Inside the Claude Code sandbox the model's live data fetches (FRED / dbnomics / yfinance) need
  network egress the sandbox blocks. Use `dangerouslyDisableSandbox: true` on **foreground** Bash
  calls for any run that fetches (background-launched runs get re-sandboxed and hang).
- A benign statsmodels `ConvergenceWarning` and one expected-drop factor
  (`global_supply_chain_pressure`, xlsx engine error → 8/9 live) are normal, not a degradation.

See [README.md](README.md#run) for the full run/scoring commands.
