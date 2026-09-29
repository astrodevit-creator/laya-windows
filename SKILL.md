---
name: laya
description: Local, free, offline typed decisions with the Laya model on Windows/Linux (PyTorch, CUDA or CPU). Use when a workflow needs a fast yes/no judgment, a pick-one-of-N choice, or an ordinal score over some text — routing, classification, triage, gating, comment/ad moderation — and a local model is preferred over an API (no credits, no network after first download). Not for open-ended writing, long reasoning, or exact lookups.
---

# Laya (Windows port)

Project: `<path-to>/laya-windows` (port of github.com/mizorewww/laya-coreml, which is macOS-only).
Weights: `convaiinnovations/laya-multilingual` (pinned revision, cached in `~/.cache/huggingface`).
Verified against upstream golden outputs: 16/16 cases, max logit diff ~2e-5 (`tests/parity.py`).

## Run

```bash
<path-to>/laya-windows/.venv/Scripts/laya.exe --offline \
  --state "text to judge" \
  --questions '{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}'
```

- `--questions`: a JSON file path or an inline JSON string.
- `--state-file path` or `--state -` (stdin) for long text. State may be a JSON object too.
- `--device cpu|cuda` (default cuda). Load ~3 s, then ~30 ms per call on GPU.
- For many items, use Python once instead of the CLI per item:
  `from laya_windows import load; agent = load(local_files_only=True); agent.predict(state, questions)`
  with `<path-to>/laya-windows/.venv/Scripts/python.exe`.

## Question types

| type | criteria | answer fields |
|---|---|---|
| `choice` | list of labels, or `{label: description}` | `choice`, `probabilities`, `confidence` |
| `score` | list of level descriptions (index = score) | `score` (expected value), `probabilities` |
| `noul` (yes/no) | optional `{"false": "...", "true": "..."}` | `noul` = P(true), `confidence` |

Max input is 512 tokens (question + options + state); longer state is truncated.

## Rules

- Treat `confidence` < ~0.7 as uncertain: escalate to a stronger model or the user instead of acting.
- It's a small classifier, not a policy engine: never use it as the only gate for spending money, deleting, or publishing.
- A `RuntimeWarning` about clamped temperatures at load is expected (upstream calibration fix).
