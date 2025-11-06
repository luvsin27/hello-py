#!/usr/bin/env python3
import os, asyncio, sys
from pathlib import Path
from dotenv import load_dotenv

# Make 'src/' importable without installing or PYTHONPATH
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv()

import src.config as cfg
from src.test_runner import run_agent_once

async def _main():
    # Submission default: Anthropic Haiku 4-5 unless MODEL is set
    model = os.environ.get("MODEL", "claude-haiku-4-5")
    runs = getattr(cfg, "NUM_RUNS", 10)
    verbose = getattr(cfg, "DEFAULT_VERBOSE", True)

    passes = 0
    for i in range(runs):
        res = await run_agent_once(i + 1, model=model, verbose=verbose)
        passes += int(res["passed"])
        fails = [k for k, v in res["checks"].items() if not v]
        print(f"Run {i+1}: passed={res['passed']} auc={res['auc']:.3f} fails={fails}")
    print(f"Pass rate: {passes}/{runs} = {passes/runs:.2f}")

if __name__ == "__main__":
    asyncio.run(_main())
