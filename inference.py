"""
inference.py — Baseline agent for Incident Response Commander
Uses OpenAI-compatible client via validator-injected API_BASE_URL and API_KEY.

Required environment variables (injected by validator):
    API_BASE_URL   → LiteLLM proxy endpoint
    API_KEY        → Validator API key

Optional:
    MODEL_NAME     → Model to use (default: Qwen/Qwen2.5-72B-Instruct)
"""

import os
import json
import re
import httpx # type: ignore
from openai import OpenAI # type: ignore
from env.environment import IncidentResponseEnv
from env.models import Action

# ── Configuration — safe defaults so module loads without crashing ──
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy-token"))
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS   = 10
TEMPERATURE = 0.2

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert on-call engineer responding to a production incident.
Your job is to investigate alerts, find the root cause, fix it, and notify your team.

You can take ONE action per turn. Choose from:
- check_logs      target=<service_name>
- check_metrics   target=<service_name>
- restart_service target=<service_name>
- rollback_deploy target=<service_name>
- scale_up        target=<service_name>
- escalate        target=<reason>
- notify_team     message=<your message>

Rules:
1. Always investigate (check_logs or check_metrics) BEFORE taking a fix action
2. Only fix the service that is actually broken
3. Do NOT end without calling notify_team — it is MANDATORY
4. Call notify_team BEFORE applying the fix
5. Do not restart or rollback services that are healthy

IMPORTANT: Your sequence should always be:
  investigate → notify_team → fix

Respond with ONLY a JSON object like this:
{"action": "check_logs", "target": "payment-service"}
or
{"action": "notify_team", "message": "payment-service is down, fixing now"}

Nothing else. No explanation. Just the JSON.
""".strip()


# ── Build prompt from observation ─────────────────────────────

def build_prompt(obs, step_history):
    services_str = "\n".join([
        f"  - {s.name}: status={s.status}, cpu={s.cpu}%, memory={s.memory}%, error_rate={s.error_rate}"
        for s in obs.services
    ])
    alerts_str = "\n".join([
        f"  - [{a.severity.upper()}] {a.service}: {a.message}"
        for a in obs.alerts
    ])
    history_str = "\n".join(step_history[-5:]) if step_history else "None"

    return f"""
CURRENT INCIDENT — Step {obs.step}
Actions remaining: {obs.actions_remaining}

ACTIVE ALERTS:
{alerts_str}

SERVICE STATUS:
{services_str}

LAST ACTION RESULT:
{obs.last_action_result}

WHAT YOU HAVE DONE SO FAR:
{history_str}

What is your next action? Respond with JSON only.
""".strip()


# ── Parse LLM response into Action ───────────────────────────

def parse_action(response_text: str) -> Action:
    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return Action(
                name=data.get("action", "escalate"),
                target=data.get("target"),
                message=data.get("message"),
            )
    except Exception:
        pass
    return Action(name="escalate", target="parse_error")


# ── Run one episode ───────────────────────────────────────────

def run_episode(task: str) -> dict:
    # ── Create client INSIDE function so validator env vars are available ──
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy-token"))
    model_name   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    client = OpenAI(
    base_url=api_base_url,
    api_key=api_key,
    http_client=httpx.Client(verify=False)
    )

    env          = IncidentResponseEnv(task=task, max_steps=MAX_STEPS)
    obs          = env.reset()
    step_history = []
    grade_result = None
    step         = 0

    print(f"[START] task={task}", flush=True)

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(obs, step_history)

        # ── LLM call — must go through validator proxy ────────
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=100,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] LLM call failed at step {step}: {e}", flush=True)
            response_text = '{"action": "escalate", "target": "llm_error"}'

        action = parse_action(response_text)
        obs, reward, done, info = env.step(action)

        action_str = action.name
        if action.target:
            action_str += f"(target={action.target})"
        elif action.message:
            action_str += f"(message={action.message})"

        print(f"[STEP] step={step} action={action_str} reward={reward.value}", flush=True)

        step_history.append(
            f"Step {step}: {action.name}({action.target or action.message or ''}) "
            f"→ reward {reward.value}"
        )

        if done:
            grade_result = info.get("grade")
            break

    # Force grade if episode didn't finish naturally
    if not grade_result:
        from env.grader import IncidentGrader
        grader       = IncidentGrader(env._scenario)
        grade_result = grader.grade(
            action_history=env._action_history,
            resolved=env._scenario.get("resolved", False)
        )

    print(f"[END] task={task} score={grade_result['score']} steps={step}", flush=True)
    return grade_result


# ── Main ──────────────────────────────────────────────────────

def main():
    print("🚨 Incident Response Commander — Baseline Agent", flush=True)
    print(f"   Model : {os.environ.get('MODEL_NAME', 'Qwen/Qwen2.5-72B-Instruct')}", flush=True)
    print(f"   API   : {os.environ.get('API_BASE_URL', 'https://router.huggingface.co/v1')}", flush=True)

    results = {}
    for task in ["easy", "medium", "hard"]:
        results[task] = run_episode(task)

    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)
    total = 0
    for task, result in results.items():
        score  = result["score"]
        passed = "✅" if result["passed"] else "❌"
        print(f"  {task:<10} {passed}  {score} / 1.0", flush=True)
        total += score

    avg = round(total / len(results), 3)
    print(f"\n  Average score: {avg} / 1.0", flush=True)
    print(f"  Tasks passed : {sum(1 for r in results.values() if r['passed'])} / {len(results)}", flush=True)


if __name__ == "__main__":
    main()