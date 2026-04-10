"""
inference.py — Baseline agent for Incident Response Commander
Uses OpenAI-compatible client to call Hugging Face Inference API.

Required environment variables (no defaults):
    HF_TOKEN       → Your Hugging Face API token (REQUIRED - no default)

Default environment variables (can be overridden):
    API_BASE_URL   → https://router.huggingface.co/v1 (default)
    MODEL_NAME     → Qwen/Qwen2.5-72B-Instruct (default)

Local image support (optional):
    LOCAL_IMAGE_NAME → Local image name for from_docker_image() (optional)
"""

import os
import json
import re
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from openai import OpenAI
from env.environment import IncidentResponseEnv
from env.models import Action

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Incident Response Agent",
    description="Baseline agent using LLM for incident response",
    version="1.0.0"
)

# ── Configuration ─────────────────────────────────────────────
# Defaults ONLY for these two
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# HF_TOKEN - provide dummy if not set to avoid crashing
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-token")

# Optional: Local image name
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Initialize OpenAI client ──────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

MAX_STEPS = 10
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


# ── Request/Response models ───────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None


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
    return Action(name="escalate", target="parse_error")


# ── Run one episode with STRUCTURED LOGGING ───────────────────

def run_episode(task: str) -> dict:
    """Run one episode with START/STEP/END structured logging format."""
    env = IncidentResponseEnv(task=task, max_steps=MAX_STEPS)
    obs = env.reset()
    step_history = []
    grade_result = None

    # START MESSAGE
    print(f"START: Running task={task}")
    
    for step in range(1, MAX_STEPS + 1):
        # Build prompt
        prompt = build_prompt(obs, step_history)

        # Call LLM using OpenAI client
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=100,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = '{"action": "escalate", "target": "llm_error"}'

        # Parse action
        action = parse_action(response_text)

        # Step environment
        obs, reward, done, info = env.step(action)

        # STEP MESSAGE with structured format
        action_str = f"{action.name}"
        if action.target:
            action_str += f"(target={action.target})"
        elif action.message:
            action_str += f"(message={action.message})"
        
        print(f"STEP {step}: action={action_str}, reward={reward.value}, reason={reward.reason}")

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
        grader = IncidentGrader(env._scenario)
        grade_result = grader.grade(
            action_history=env._action_history,
            resolved=env._scenario.get("resolved", False)
        )

    # END MESSAGE
    print(f"END: task={task}, score={grade_result['score']}, passed={grade_result['passed']}")

    return grade_result


# ── API Endpoints ──────────────────────────────────────────────

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Incident Response Agent",
        "version": "1.0.0",
        "description": "Baseline agent for OpenEnv Incident Response",
    }

@app.post("/reset")
def reset(request: Optional[Dict] = None):
    """Start a fresh episode and run full agent loop."""
    task = "easy"
    if request and isinstance(request, dict) and "task" in request:
        task = request.get("task", "easy")
    
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {task}. Choose from: easy, medium, hard"
        )
    
    print(f"START: Baseline agent initialization")
    print(f"STEP 0: model={MODEL_NAME}, api_url={API_BASE_URL}")
    
    result = run_episode(task)
    
    print(f"END: All tasks completed, final_score={result['score']}")
    
    return {
        "task": task,
        "score": result["score"],
        "passed": result["passed"],
        "feedback": result["feedback"],
    }


# ── Main entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=7860, reload=False)
