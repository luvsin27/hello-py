
from typing import Any, Dict, TypedDict, Callable, Optional
from io import StringIO
from contextlib import redirect_stdout
import builtins
import config as cfg

class PythonExpressionToolResult(TypedDict):
    result: Any
    error: Optional[str]

class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool

_LAST_SUBMISSION: Optional[dict] = None

def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    global _LAST_SUBMISSION
    if _LAST_SUBMISSION is not None:
        return {"answer": _LAST_SUBMISSION, "submitted": True}

    if not isinstance(answer, dict):
        return {"answer": "ERROR: answer must be a dict", "submitted": False}

    prob_keys = ("y_pred_proba", "probas", "proba", "pred_proba", "y_pred")
    if not any(k in answer for k in prob_keys):
        return {"answer": "ERROR: include probabilities under y_pred_proba/proba/probas/pred_proba/y_pred", "submitted": False}

    _LAST_SUBMISSION = answer
    return {"answer": _LAST_SUBMISSION, "submitted": True}

def pop_submission() -> Optional[dict]:
    global _LAST_SUBMISSION
    out = _LAST_SUBMISSION
    _LAST_SUBMISSION = None
    return out

def python_expression_tool_factory(context: Dict[str, Any], submit_handler: Callable[[Any], SubmitAnswerToolResult]):
    ns: Dict[str, Any] = dict(context)

    def _run(expression: str) -> PythonExpressionToolResult:
        try:
            stdout = StringIO()

            ncols = None
            try:
                if "train_df" in ns and hasattr(ns["train_df"], "columns"):
                    ncols = len(ns["train_df"].columns)
            except Exception:
                pass
            allowed_values = (cfg.PRINT_VALUES_FACTOR * ncols) if ncols else 100
            printed = {"count": 0}
            _orig_print = builtins.print

            def _guard_print(*args, **kwargs):
                printed["count"] += len(args)
                if printed["count"] > allowed_values:
                    raise RuntimeError("Output limit exceeded: too many values printed")
                return _orig_print(*args, **kwargs)

            ns["print"] = _guard_print

            def _submit(answer):
                return submit_handler(answer)
            ns["submit_answer"] = _submit

            with redirect_stdout(stdout):
                exec(expression, ns, ns)

            out = stdout.getvalue()
            if len(out) > cfg.STDOUT_MAX_CHARS:
                out = out[: cfg.STDOUT_MAX_CHARS] + "\n...[truncated]"
            return {"result": out, "error": None}
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return {"result": None, "error": str(e)}

    return _run

def make_tools() -> list:
    return [
        {
            "name": "python_expression",
            "description": "Executes Python code. Use print() to emit output; returns stdout (capped).",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit {'y_pred_proba' (or 'proba'/'probas'/'pred_proba'/'y_pred'), 'pipeline' (optional)}",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer"}},
                "required": ["answer"],
            },
        },
    ]
