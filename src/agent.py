from typing import Any, Callable, Dict, Optional
import os, json, asyncio, random
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

async def _anthropic_with_retry(client: AsyncAnthropic, **kwargs):
    for attempt in range(5):
        try:
            return await client.messages.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if 'overloaded' in msg or '529' in msg or 'temporarily' in msg or 'timeout' in msg:
                await asyncio.sleep((2 ** attempt) * 0.5 + random.uniform(0, 0.5))
                continue
            raise
    return await client.messages.create(**kwargs)

async def run_agent_loop(prompt: str, tools: list, tool_handlers: Dict[str, Callable[..., Any]],
                         max_steps: int = 10, model: Optional[str] = None, verbose: bool = True) -> Optional[Any]:
    client = AsyncAnthropic()
    model = model or os.environ.get('MODEL', 'claude-haiku-4-5')
    messages = [{'role': 'user', 'content': prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")
        resp = await _anthropic_with_retry(client, model=model, max_tokens=1000, tools=tools, messages=messages)

        has_tool_use = False
        submitted_answer = None
        tool_results = []

        for c in resp.content:
            if c.type == 'text':
                if verbose:
                    print(f"Assistant: {c.text}")
            elif c.type == 'tool_use':
                has_tool_use = True
                name = c.name
                handler = tool_handlers.get(name)
                if not handler:
                    continue
                data = c.input or {}

                if name == 'python_expression':
                    code = data.get('expression', '')
                    if verbose:
                        print("\n[python_expression input]\n```\n" + code + "\n```")
                    result = handler(code)
                    if verbose:
                        print("[python_expression output]\n```\n" + str(result) + "\n```")
                    tool_results.append({'type': 'tool_result', 'tool_use_id': c.id, 'content': json.dumps(result)})

                elif name == 'submit_answer':
                    result = handler(data.get('answer'))
                    submitted = bool(result.get('submitted'))
                    if submitted:
                        submitted_answer = result.get('answer')
                    tool_results.append({'type': 'tool_result', 'tool_use_id': c.id, 'content': json.dumps({'submitted': submitted})})

                else:
                    res = handler(**data) if isinstance(data, dict) else handler(data)
                    tool_results.append({'type': 'tool_result', 'tool_use_id': c.id, 'content': json.dumps(res)})

        if has_tool_use:
            messages.append({'role': 'assistant', 'content': resp.content})
            messages.append({'role': 'user', 'content': tool_results})
            if submitted_answer is not None:
                if verbose:
                    print("\n[agent submitted] dict with y_pred_proba & pipeline")
                return submitted_answer
        else:
            if verbose:
                print("No tool use; stopping.")
            break
    return None
