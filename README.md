# RL Task for LLMs (Leakage-Aware Modeling)

**Goal:** Evaluate whether an LLM can build a *leakage-proof* ML pipeline and produce calibrated probabilities that achieve at least a target AUROC on a synthetic binary-classification task.

This repo includes:
- An **agent mode** (Anthropic Claude via tools) for the RL task
- A **local mode** (no API) to sanity-check the dataset, grader, and baseline
- A small **auto-threshold** utility to calibrate the pass rate into the required **10–40%** band

The code is small, auditable, and costs only what you spend on model calls.

---

## Table of Contents
- [What the Task Tests](#what-the-task-tests)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Run the Task](#run-the-task)
  - [Local mode (free)](#local-mode-free)
  - [Agent mode (Anthropic)](#agent-mode-anthropic)
  - [Submission command](#submission-command)
- [Configuration](#configuration)
- [How Grading Works](#how-grading-works)
- [Agent Tools](#agent-tools)
- [Auto-Threshold Tuning](#auto-threshold-tuning)
- [Debugging Tips](#debugging-tips)
- [FAQ](#faq)
- [License](#license)

---

## What the Task Tests

The agent is given **train/test DataFrames** with numeric and categorical features and explicit leaky columns. The objective is to build a leakage-proof ML pipeline, fit it, and produce calibrated probabilities.

**Pass condition:** AUROC >= threshold (0.9471 by default).

The model passes 10–40% of runs to test robustness.

---

## Project Structure

```
hello-py/
├── main.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── rl_llm_exercise.md
├── scripts/
│   └── auto_threshold.py
└── src/
    ├── config.py
    ├── environment.py
    ├── agent.py
    ├── tools.py
    ├── evaluator.py
    └── test_runner.py
```

---

## Setup

### Requirements
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) or pip
- Anthropic API key

### Installation
```bash
uv pip install -e .
cp .env.example .env
# Add ANTHROPIC_API_KEY=sk-ant-...
```

---

## Run the Task

### Local mode (free)
```bash
PYTHONPATH=src uv run python -m test_runner --mode local --runs 1 --verbose
```

### Agent mode (Anthropic)
```bash
PYTHONPATH=src uv run python -m test_runner --mode agent --runs 1 --model claude-3-5-haiku-latest --verbose
```

### Submission (default Haiku 4-5)
```bash
uv run main.py
```

---

## Configuration

All config options live in `src/config.py`.  
Key ones:
```python
THRESHOLD = 0.9471
NUM_RUNS = 10
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_STEPS = 20
MAX_TOKENS = 1000
ENABLE_LEAK_FUTURE_SIGNAL = True
ENABLE_LEAK_GLOBAL_TARGET_MEAN = True
ENABLE_LEAK_CAT0_RATE_FULL = True
```

---

## How Grading Works

- The agent submits probabilities via `submit_answer()`.
- AUROC is computed between predicted probabilities and true labels.
- Passing condition: `AUROC >= THRESHOLD`.

---

## Agent Tools

- **python_expression**: Runs code in sandbox with `train_df`, `test_df`, `pandas`, `numpy`, etc.
- **submit_answer**: Finalizes output. Accepts a dict with `y_pred_proba` and `pipeline`.

---

## Auto-Threshold Tuning

```bash
uv run python scripts/auto_threshold.py --mode local --runs 5 --quantile 0.80
uv run python scripts/auto_threshold.py --mode agent --model claude-haiku-4-5 --runs 20 --quantile 0.85
```

---

## Debugging Tips

- Check `stdout` truncation limits (`STDOUT_MAX_CHARS`).
- Drop leaky columns before feature extraction.
- Use local mode to validate environment quickly.

---

## FAQ

**Q:** Why some runs fail despite valid pipeline?  
**A:** To maintain a 10–40% pass band—variation is intentional.

**Q:** Why AUROC?  
**A:** It’s threshold-free and stable for probabilistic outputs.

**Q:** What leakage sources exist?  
A: `leak_future_signal`, `leak_global_target_mean`, `leak_cat0_rate_full`.

**Q:** How to reduce costs?  
A: Use Haiku 3.5 for dev; switch to 4.5 for evaluation.

---

## License
MIT
