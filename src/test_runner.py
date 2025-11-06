import os, argparse, asyncio
from dotenv import load_dotenv

import config as cfg
from environment import make_problem, build_baseline_and_predict
from evaluator import grade_submission
from tools import make_tools, python_expression_tool_factory, submit_answer_tool, pop_submission
from agent import run_agent_loop

load_dotenv()

def build_prompt(spec, present_leaks) -> str:
    return (
        "You are given Pandas DataFrames: train_df and test_df.\n"
        "Goal: Build a leakage-proof sklearn baseline and return test-set probabilities.\n"
        f"Drop ALL leaky columns listed in present_leaks: {present_leaks}.\n"
        "Build a Pipeline with train-only preprocessing:\n"
        "  numeric -> impute(median) + scale ; categorical -> OneHotEncoder(handle_unknown='ignore').\n"
        "Train a simple classifier (LogisticRegression is fine). Return PROBABILITIES for test_df in original order.\n"
        "Submission options:\n"
        "  A) Inside python_expression: submit_answer({'y_pred_proba': y_proba, 'pipeline': pipeline})\n"
        "  B) As a tool call: submit_answer with {'answer': {'y_pred_proba': list(y_proba), 'pipeline': pipeline}}\n"
        "Do NOT try to import submit_answer; it is provided in the Python context and also as a separate tool.\n"
        f"Success: AUROC >= {cfg.THRESHOLD:.2f}. After a successful submit, STOP and do not submit again.\n"
    )

async def run_agent_once(run_id: int, model: str, verbose: bool = False):
    train_df, test_df, y_test, spec = make_problem(seed=1000 + run_id, n_train=cfg.N_TRAIN, n_test=cfg.N_TEST)
    present_leaks = [c for c in spec.leak_cols if (c in train_df.columns) or (c in test_df.columns)]
    prompt = build_prompt(spec, present_leaks)

    ctx = {'train_df': train_df, 'test_df': test_df, 'present_leaks': present_leaks}

    tools = make_tools()
    tool_handlers = {
        'python_expression': python_expression_tool_factory(ctx, submit_answer_tool),
        'submit_answer': submit_answer_tool,
    }

    submitted = await run_agent_loop(
        prompt, tools, tool_handlers, max_steps=cfg.DEFAULT_MAX_STEPS, model=model, verbose=verbose
    )
    submission = submitted if isinstance(submitted, dict) else pop_submission()

    if not isinstance(submission, dict):
        print("Submission format invalid; expected dict with 'y_pred_proba' and 'pipeline'.")
        return {'passed': False, 'auc': float('nan'), 'checks': {'submission_format_ok': False}}

    result = grade_submission(submission, y_test, spec, test_df)
    return result

def run_local_once(run_id: int):
    train_df, test_df, y_test, spec = make_problem(seed=42 + run_id, n_train=cfg.N_TRAIN, n_test=cfg.N_TEST)
    submission = build_baseline_and_predict(train_df, test_df, leak_cols=spec.leak_cols, target=spec.target_name)
    result = grade_submission(submission, y_test, spec, test_df)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['local', 'agent'], default='local')
    ap.add_argument('--runs', type=int, default=cfg.NUM_RUNS)
    ap.add_argument('--model', type=str, default=os.environ.get('MODEL', cfg.DEFAULT_MODEL))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    passes = 0
    if args.mode == 'local':
        for i in range(args.runs):
            res = run_local_once(i + 1)
            passes += int(res['passed'])
            print(f"Run {i+1}: passed={res['passed']} auc={res['auc']:.3f} fails={[k for k, v in res['checks'].items() if not v]}")
        print(f"Pass rate: {passes}/{args.runs} = {passes/args.runs:.2f}")
    else:
        async def _a():
            nonlocal passes
            for i in range(args.runs):
                res = await run_agent_once(i + 1, model=args.model, verbose=args.verbose)
                passes += int(res['passed'])
                print(f"Run {i+1}: passed={res['passed']} auc={res['auc']:.3f} fails={[k for k, v in res['checks'].items() if not v]}")
            print(f"Pass rate: {passes}/{args.runs} = {passes/args.runs:.2f}")
        asyncio.run(_a())

if __name__ == '__main__':
    main()
