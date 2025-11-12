# RL Task (Leakage Aware Modeling)

**Goal:** Evaluate whether an LLM can build a leakage-proof ML pipeline and produce probabilities that achieve AUROC ≥ threshold on a synthetic binary classification task, under step, token, and tool constraints.

---

## Index
1. What skill is learned  
2. Task at a glance  
3. Leakage types  
4. Common failure modes  
5. Guardrails  
6. Project structure  
7. Setup  
8. Run the task  
9. Configuration  
10. Grading  
11. Agent tools  
12. Auto-threshold tuning  
13. Debug quick checklist  
14. FAQ  
15. Limitations  
16. Future expansion  

---

## 1. What skill is learned
- Leakage awareness and prevention: Identify and neutralize features that leak future or target information.  
- Leak-safe preprocessing: Fit imputers and scalers on train only, then transform both train and test.  
- Probability-first modeling: Return probabilities and reason about AUROC (threshold-free).  
- Tool discipline: Use tools correctly, respect step, token, and output limits, and submit once.

---

## 2. Task at a glance
- **Inputs:** `train_df`, `test_df` containing:  
  - Numeric: `num_0 … num_4`  
  - Categorical: `cat0`, `cat1`  
  - Leaky: `leak_future_signal`, `leak_global_target_mean`, `leak_cat0_rate_full`  
  - Target: `target` (binary)
- **Output:** Probabilities for the test set (`y_pred_proba`)
- **Pass condition:** AUROC ≥ THRESHOLD (default 0.9471)
- **Evaluation:** 10 runs, expected pass-rate 10–40 percent

---

## 3. Leakage types
1. **Future signal:** Encodes information from after the label time.  
2. **Global target mean:** Target mean computed using all rows, not train-only.  
3. **Full-data category rate:** Category rates computed with all rows (may include current or test-period rows).  

> These represent real-world pitfalls such as no out-of-fold encoders or aggregates that include current rows.

---

## 4. Common failure modes
- Dropping leaks after computing feature lists (transformer mismatch).  
- Fitting preprocessors on combined train and test.  
- Submitting wrong payloads (NaN, Inf, wrong length, or values not in [0,1]).  
- Multiple submissions (only first counts).  
- Excessive prints exceeding stdout limits.  
- Step or token exhaustion before submission.  
- Poor model regularization causing unstable AUROC.

---

## 5. Guardrails
- Validates shape and values of `y_pred_proba` (1D, finite, [0,1]).  
- Only first submission is graded.  
- Output limited by `STDOUT_MAX_CHARS` and `PRINT_VALUES_FACTOR`.  
- Controlled step and token budgets.  
- Deterministic seeds with small randomness.  
- Grader isolates evaluation to avoid peeking.

---

## 6. Project structure
```
hello-py/
├── main.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
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
> Flat src layout. Use PYTHONPATH=src for CLI runs. main.py handles submission mode automatically.

---

## 7. Setup
Requirements: Python 3.10+, uv or pip, Anthropic API key.

```bash
uv pip install -e .
cp .env.example .env
# Add your key in .env
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 8. Run the task

**Local mode (free):**
```bash
PYTHONPATH=src uv run python -m test_runner --mode local --runs 1 --verbose
```

**Agent mode (for ad-hoc testing use the following runner):**
```bash
PYTHONPATH=src uv run python -m test_runner --mode agent --runs 1 --model claude-3-5-haiku-latest --verbose
```

**Submission (10 runs, Haiku 4-5):**
```bash
uv run main.py
```

---

## 9. Configuration
Key values from `src/config.py`:

```python
N_TRAIN = 800
N_TEST = 200
THRESHOLD = 0.9471
NUM_RUNS = 10
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_STEPS = 20
MAX_TOKENS = 1000
DEFAULT_VERBOSE = True
STDOUT_MAX_CHARS = 8000
PRINT_VALUES_FACTOR = 2
ENABLE_LEAK_FUTURE_SIGNAL = True
ENABLE_LEAK_GLOBAL_TARGET_MEAN = True
ENABLE_LEAK_CAT0_RATE_FULL = True
DEBUG = True
```
---

## 10. Grading
1. Validate submission payload (single call, correct shape and range).  
2. Compute AUROC on test labels.  
3. Pass if AUROC ≥ THRESHOLD.  
4. Aggregate over 10 runs (goal 10–40 percent pass-rate).

---

## 11. Agent tools
- `python_expression`: Execute Python with access to `train_df`, `test_df`, `pandas`, `numpy`, `sklearn`, `scipy`. Stdout is capped by `STDOUT_MAX_CHARS`. Avoid printing full DataFrames (enforced by `PRINT_VALUES_FACTOR`).
- `submit_answer`: Submit once with:
  ```python
  submit_answer({'y_pred_proba': y_proba})
---

## 12. Auto-threshold tuning
Adjust threshold to maintain 10–40 percent pass-rate.

Local:
```bash
uv run python scripts/auto_threshold.py --mode local --runs 5 --quantile 0.80
```

Agent:
```bash
uv run python scripts/auto_threshold.py --mode agent --model claude-haiku-4-5 --runs 20 --quantile 0.85
```

---

## 13. Debug quick checklist
- Recompute features after dropping leaks and target.  
- Use balanced regularization for better AUROC.  
- Increase MAX_TOKENS if step budget exceeded.  
- Re-tune THRESHOLD using auto-threshold script if pass-rate drifts.

---

## 14. FAQ

**Q: Why AUCROC is used as the threshold metric?**  
A: We use **AUCROC** because it measures discrimination across all thresholds and is robust under class imbalance. In this environment it serves a *different purpose*:  
- It’s **not** a pure accuracy metric.  
- It’s a **proxy for leakage awareness**—we reward realistic performance and penalize suspiciously perfect scores.

If the model **keeps leaky features**, AUC tends toward **≈1.0** → **Fail** (cheating).  
If the model **removes leaks but underfits**, AUC falls **below threshold** → **Fail** (lost genuine signal).  
If the model **balances** leak removal and predictive power, AUC lands around **0.94–0.96** → **Pass** (honest generalization).

This teaches the instinct of a responsible ML engineer: doubt perfect metrics and aim for fair generalization.

**Q: Must I drop the leaky columns?**  
A: You must avoid using them. Dropping is simplest; neutralizing/masking is fine if no leakage remains.

**Q: Can I use any classifier?**  
A: Yes, any sklearn classifier that outputs probabilities is valid.

**Q: Why single submission?**  
A: Encourages planning and prevents random retries.

**Q: Why no concurrency?**  
A: It encourages deliberate planning and prevents trial-and-error retries.

**Q: How to minimize API cost?**  
A: Use local mode for development; only evaluate final with 10 runs.
(and threshold calibration if needed).

---

## 15. Limitations
- Synthetic data, not domain specific.  
- Only three leakage patterns modeled.  
- AUROC only, no calibration metric.  
- Sequential runs, no concurrency. - Binary classification only.

---

## 16. Future expansion
- Add time-series leakage checks.  
- Implement out-of-fold encoders.  
- Introduce calibration metrics and cost-based scoring.  
- Add structured missingness and drift simulations.  
- Extend tool support (safe file I/O, SQL sandbox).
