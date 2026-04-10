"""
app.py — FastAPI server wrapping Incident Response Commander
Exposes reset(), step(), state() as HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

from env.environment import IncidentResponseEnv
from env.models import Action

app = FastAPI(
    title="Incident Response Commander",
    description="OpenEnv environment simulating production incident response.",
    version="1.0.0"
)

# ── In-memory session store ───────────────────────────────────
# Stores one environment per task
sessions: Dict[str, IncidentResponseEnv] = {}


# ── Request/Response models ───────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"     # "easy" | "medium" | "hard"

class StepRequest(BaseModel):
    task: str = "easy"
    action: str            # e.g. "check_logs"
    target: Optional[str] = None
    message: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Incident Response Commander",
        "version": "1.0.0",
        "description": "OpenEnv environment for production incident response",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health"]
    }


@app.get("/health")
def health():
    """Judges ping this to check if the Space is alive."""
    return {"status": "ok"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None, task: str = Query(default="easy")): # type: ignore
    """Start a fresh incident episode."""
    # Use body task if provided, else fall back to query param
    actual_task = (request.task if request else None) or task
    
    if actual_task not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {actual_task}. Choose from: easy, medium, hard"
        )

    env = IncidentResponseEnv(task=actual_task, max_steps=10)
    obs = env.reset()
    sessions[actual_task] = env

    return {
        "observation": obs.model_dump(),
        "task": actual_task,
        "message": "Episode started. Use /step to take actions."
    }

@app.post("/step")
def step(request: StepRequest):
    """Take one action in the environment."""
    if request.task not in sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /reset first."
        )

    env = sessions[request.task]

    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new one."
        )

    action = Action(
        name=request.action,
        target=request.target,
        message=request.message,
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state")
def state(task: str = "easy"):
    """Get current environment state."""
    if task not in sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call /reset first."
        )
    return sessions[task].state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Single Service Crash",
                "difficulty": "easy",
                "description": "One microservice crashed. Find it and restart it.",
                "max_steps": 10,
                "passing_score": 0.5,
            },
            {
                "id": "medium",
                "name": "Cascading Failure from Bad Deploy",
                "difficulty": "medium",
                "description": "Bad deployment caused cascading failures. Rollback it.",
                "max_steps": 10,
                "passing_score": 0.5,
            },
            {
                "id": "hard",
                "name": "Multi-System Incident with Noise",
                "difficulty": "hard",
                "description": "DB overload with noisy alerts. Find real root cause.",
                "max_steps": 10,
                "passing_score": 0.5,
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)