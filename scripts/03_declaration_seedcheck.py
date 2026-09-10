"""Repeat the existing unsteered User/Tool pilot on five seeds.

Run in the same Colab kernel after 02_declaration_pilot.py. Reuses its
simulated tools and the already loaded model. One page, ten trajectories;
this is a debugging check, not ten independent prompt examples.
"""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

required = ("model", "tokenizer", "run_condition", "make_page", "declaration_report")
missing = [name for name in required if name not in globals()]
assert not missing, f"Kernel state missing: {missing}. Do not reload the model blindly."

# Confirm that this uses the same fixture and that the original report has
# all three conditions before comparing repeatability.
original_runs = declaration_report["runs"]
original_by_condition = {r["condition"]: r for r in original_runs if r["seed"] == 123}
assert {"clean", "tool", "user"} <= set(original_by_condition)
for condition in ("tool", "user"):
    assert hashlib.sha256(make_page(condition).encode()).hexdigest() == original_by_condition[condition]["page_sha256"]

report = copy.deepcopy({k: v for k, v in declaration_report.items() if k != "runs"})
report.update({
    "test": "unsteered_declaration_seedcheck",
    "seedcheck_script_version": "0.1.0",
    "seeds": [123, 124, 125, 126, 127],
    "scope": "One short synthetic page; two conditions at five seeds; simulated tools; no steering. Seed 123 repeats the initial pilot. Four seeds are new. Not a held-out prompt evaluation or an independent-prompt ASR estimate.",
    "condition_order": "Alternate tool/user order across seeds; seed 123 repeats original ordering.",
    "runs": [],
})
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = Path("/content/mats-role") / f"declaration_seedcheck_{stamp}.json"
out.parent.mkdir(parents=True, exist_ok=True)
print("REPORT:", str(out), flush=True)

for index, seed in enumerate(report["seeds"]):
    conditions = ("tool", "user") if index % 2 == 0 else ("user", "tool")
    for condition in conditions:
        record = run_condition(model, tokenizer, condition, seed)
        if seed == 123:
            previous = original_by_condition[condition]
            record["matches_initial_pilot_raw_outputs"] = (
                [s["raw"] for s in record["steps"]]
                == [s["raw"] for s in previous["steps"]]
            )
        report["runs"].append(record)
        out.write_text(json.dumps(report, indent=2))
        summary = {k: record[k] for k in (
            "condition", "seed", "status", "upload_request_emitted",
            "simulated_upload_success", "elapsed_seconds", "peak_allocated_gib",
        )}
        if seed == 123:
            summary["matches_initial_pilot_raw_outputs"] = record["matches_initial_pilot_raw_outputs"]
        print("RESULT:", json.dumps(summary), flush=True)

latest = out.parent / "declaration_seedcheck_latest.json"
latest.write_text(json.dumps(report, indent=2))
print("SEEDCHECK_COMPLETE; DOWNLOAD:", str(latest), flush=True)
