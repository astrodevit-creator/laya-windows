"""Compare this Windows port against golden outputs from the unmodified upstream Laya package.

Reference: laya-coreml benchmarks/results/reference.json (upstream torch, float32).
Usage: python tests/parity.py [--device cpu|cuda]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from laya_windows import load
from laya_windows.inputs import collate_items

ref = json.loads((Path(__file__).parent / "reference-multilingual.json").read_text(encoding="utf-8"))
p = argparse.ArgumentParser()
p.add_argument("--device")
args = p.parse_args()
agent = load("convaiinnovations/laya-multilingual", device=args.device)
m = ref["models"]["laya-multilingual"]
worst_logit, worst_prob, failures = 0.0, 0.0, 0
for case in m["cases"]:
    items, _ = agent.prepare(case["state"], case["questions"])
    ids_ok = all(a["ids"] == b["ids"] and a["markers"] == b["markers"]
                 for a, b in zip(items, case["items"])) and len(items) == len(case["items"])
    logits = []
    for item in items:
        batch = collate_items([item], agent.tok.pad_token_id, shape=agent.shape)
        lg, _ = agent.forward(batch)
        logits.append(lg[0, : len(item["markers"])])
    ref_logits = [np.array(r[: len(it["markers"])]) for r, it in zip(case["logits"], items)]
    dl = max(float(np.abs(a - b).max()) for a, b in zip(logits, ref_logits))
    got = agent.predict(case["state"], case["questions"])["answers"]
    exp = case["result"]["answers"]
    dp, same = 0.0, True
    for qid, e in exp.items():
        g = got[qid]
        same &= g.get("choice") == e.get("choice")
        for k, v in e.get("probabilities", {}).items():
            dp = max(dp, abs(g["probabilities"][k] - v))
        if "noul" in e:
            dp = max(dp, abs(g["noul"] - e["noul"]))
    ok = ids_ok and same and dp <= 0.01
    failures += not ok
    worst_logit, worst_prob = max(worst_logit, dl), max(worst_prob, dp)
    print(f"{'PASS' if ok else 'FAIL'} {case['name']:<24} tokens={'ok' if ids_ok else 'DIFF'} "
          f"max|dlogit|={dl:.2e} max|dprob|={dp:.4f}")
print(f"device={agent.device} cases={len(m['cases'])} failures={failures} "
      f"worst_logit={worst_logit:.2e} worst_prob={worst_prob:.4f}")
sys.exit(1 if failures else 0)
