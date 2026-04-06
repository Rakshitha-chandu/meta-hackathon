"""
inference.py — Baseline agent for Incident Response Commander
Uses free Hugging Face Inference API via OpenAI-compatible client.

Required environment variables:
    API_BASE_URL   → https://router.huggingface.co/v1
    MODEL_NAME     → Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       → your Hugging Face token
"""

import os
import json
import re
from openai import OpenAI
from env.environment import IncidentResponseEnv
from env.models import Action

# ── Config ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS    = 10
TEMPERATURE  = 0.2

if not API_KEY:
    raise ValueError("HF_TOKEN environment variable not set.")

# ── OpenAI-compatible client pointing at HF ───────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

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
4. Call notify_team BEFORE applying the fix, like: notify_team("fixing payment-service now")
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

    prompt = f"""
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
    return prompt


# ── Parse LLM response into Action ───────────────────────────

def parse_action(response_text: str) -> Action:
    """Extract JSON from LLM response and convert to Action."""
    try:
        # Try to find JSON in the response
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

    # Fallback if parsing fails
    print(f"  ⚠️  Could not parse response: {response_text[:100]}")
    return Action(name="escalate", target="parse_error")


# ── Run one episode ───────────────────────────────────────────

def run_episode(task: str) -> dict:
    print(f"\n{'='*60}")
    print(f"TASK: {task.upper()}")
    print('='*60)

    env          = IncidentResponseEnv(task=task, max_steps=MAX_STEPS)
    obs          = env.reset()
    step_history = []
    grade_result = None

    print(f"Incident: {env._scenario['description']}")
    print(f"Alerts  : {[a.message for a in obs.alerts]}")

    for step in range(1, MAX_STEPS + 1):
        # Build prompt
        prompt = build_prompt(obs, step_history)

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=100,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  ⚠️  LLM call failed: {e}")
            response_text = '{"action": "escalate", "target": "llm_error"}'

        # Parse action
        action = parse_action(response_text)
        print(f"\n  Step {step}: LLM chose → {action.name}({action.target or action.message or ''})")

        # Step environment
        obs, reward, done, info = env.step(action)
        print(f"  Result  : {obs.last_action_result}")
        print(f"  Reward  : {reward.value} ({reward.reason})")

        # Track history for context
        step_history.append(
            f"Step {step}: {action.name}({action.target or action.message or ''}) "
            f"→ reward {reward.value}"
        )

        if done:
            grade_result = info.get("grade")
            break

    # Force grade if episode didn't finish
    if not grade_result:
        from env.grader import IncidentGrader
        grader       = IncidentGrader(env._scenario)
        grade_result = grader.grade(
            action_history=env._action_history,
            resolved=env._scenario.get("resolved", False)
        )

    # Print final grade
    print(f"\n── FINAL GRADE ────────────────────────────────")
    print(f"  Score   : {grade_result['score']} / 1.0")
    print(f"  Passed  : {'✅ YES' if grade_result['passed'] else '❌ NO'}")
    print(f"  Feedback:")
    for f in grade_result["feedback"]:
        print(f"    {f}")

    return grade_result


# ── Main ──────────────────────────────────────────────────────

def main():
    print("🚨 Incident Response Commander — Baseline Agent")
    print(f"   Model : {MODEL_NAME}")
    print(f"   API   : {API_BASE_URL}")

    results = {}
    for task in ["easy", "medium", "hard"]:
        results[task] = run_episode(task)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    total = 0
    for task, result in results.items():
        score  = result["score"]
        passed = "✅" if result["passed"] else "❌"
        print(f"  {task:<10} {passed}  {score} / 1.0")
        total += score

    avg = round(total / len(results), 3)
    print(f"\n  Average score: {avg} / 1.0")
    print(f"  Tasks passed : {sum(1 for r in results.values() if r['passed'])} / {len(results)}")


if __name__ == "__main__":
    main()