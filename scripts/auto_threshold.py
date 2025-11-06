#!/usr/bin/env python3
import os, sys, argparse, math, asyncio, re
from pathlib import Path

# ---- Bootstrap imports for flattened src/ layout ----
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as cfg
from environment import make_problem, build_baseline_and_predict
from evaluator import grade_submission
from test_runner import run_agent_once

def measure_local(runs: int):
    aucs = []
    for i in range(runs):
        train_df, test_df, y_test, spec = make_problem(
            seed=4242 + i, n_train=cfg.N_TRAIN, n_test=cfg.N_TEST
        )
        submission = build_baseline_and_predict(
            train_df, test_df, leak_cols=spec.leak_cols, target=spec.target_name
        )
        res = grade_submission(submission, y_test, spec, test_df)
        auc = res.get("auc", float("nan"))
        if auc == auc:  # filter NaN
            aucs.append(float(auc))
    return aucs

async def _measure_agent(runs: int, model: str):
    aucs = []
    for i in range(runs):
        res = await run_agent_once(i + 1, model=model, verbose=False)
        auc = res.get("auc", float("nan"))
        if auc == auc:
            aucs.append(float(auc))
    return aucs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "agent"], default="local")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", getattr(cfg, "DEFAULT_MODEL", "claude-3-5-haiku-latest")),
    )
    ap.add_argument("--quantile", type=float, default=0.80)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode == "local":
        aucs = measure_local(args.runs)
    else:
        aucs = asyncio.run(_measure_agent(args.runs, args.model))

    if not aucs:
        print("No AUCs collected; aborting.")
        sys.exit(2)

    aucs_sorted = sorted(aucs)
    k = max(0, min(len(aucs_sorted) - 1, math.floor(args.quantile * (len(aucs_sorted) - 1))))
    q = aucs_sorted[k]

    print(f"Runs: {len(aucs_sorted)}")
    print(f"AUCs sample: {aucs_sorted[:10]} ...")
    print(f"p{int(args.quantile * 100)} = {q:.4f}")

    # ---- Write THRESHOLD into flattened src/config.py ----
    cfg_file = SRC / "config.py"
    txt = cfg_file.read_text(encoding="utf-8")

    m = re.search(r"(^\s*THRESHOLD\s*=\s*)([0-9]*\.?[0-9]+)", txt, flags=re.M)
    if not m:
        # Append if missing
        new_txt = txt + f"\n# Auto-calibrated by scripts/auto_threshold.py\nTHRESHOLD = {q:.4f}\n"
    else:
        new_txt = re.sub(r"(^\s*THRESHOLD\s*=\s*)([0-9]*\.?[0-9]+)", rf"\g<1>{q:.4f}", txt, flags=re.M)

    if args.dry_run:
        print("--- DRY RUN ---")
        print(new_txt)
        return

    cfg_file.with_suffix(".py.bak").write_text(txt, encoding="utf-8")
    cfg_file.write_text(new_txt, encoding="utf-8")
    print(f"Updated THRESHOLD in {cfg_file} to {q:.4f}")

if __name__ == "__main__":
    main()