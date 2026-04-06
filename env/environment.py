import copy
from typing import Tuple, Dict, Any, Optional
from env.models import Observation, Action, Reward, Service, Alert

VALID_ACTIONS = [
    "check_logs",
    "check_metrics",
    "restart_service",
    "rollback_deploy",
    "scale_up",
    "escalate",
    "notify_team"
]

class IncidentResponseEnv:
    def __init__(self, task: str = "easy", max_steps: int = 10):
        self.task = task
        self.max_steps = max_steps
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._scenario = None
        self._done = False
        self._action_history = []

    def reset(self) -> Observation:
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._action_history = []
        self._scenario = self._load_scenario(self.task)
        return self._build_observation("Incident detected. Investigate and resolve.")




    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._current_step += 1
        self._action_history.append({
            "name":    action.name,
            "target":  action.target,
            "message": action.message,
        })

        result_message, reward_value, reward_reason = self._process_action(action)
        self._cumulative_reward = round(self._cumulative_reward + reward_value, 3)

        if self._current_step >= self.max_steps:
            self._done = True
            result_message += " [Max steps reached]"

        if self._scenario.get("resolved", False):
            if "notify_team" in [a["name"] for a in self._action_history]:
                self._done = True
            elif self._current_step >= self.max_steps:
                self._done = True
    # else: keep going so agent can notify

        # ── Run grader when episode ends ──────────────────────────
        grade_result = None
        if self._done:
            from env.grader import IncidentGrader
            grader      = IncidentGrader(self._scenario)
            grade_result = grader.grade(
                action_history=self._action_history,
                resolved=self._scenario.get("resolved", False)
            )

        reward = Reward(
            value=reward_value,
            reason=reward_reason,
            cumulative=self._cumulative_reward
        )
        obs  = self._build_observation(result_message)
        info = {
            "action_history": self._action_history,
            "task":           self.task,
            "grade":          grade_result,   # None mid-episode, dict at end
        }
        return obs, reward, self._done, info


    def state(self) -> Dict:
        return {
            "task": self.task,
            "step": self._current_step,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "scenario": self._scenario,
            "action_history": self._action_history
        }

    # ── action handlers ──────────────────────────────────────────

    def _process_action(self, action: Action):
        handlers = {
            "check_logs":       self._handle_check_logs,
            "check_metrics":    self._handle_check_metrics,
            "restart_service":  self._handle_restart_service,
            "rollback_deploy":  self._handle_rollback_deploy,
            "scale_up":         self._handle_scale_up,
            "escalate":         self._handle_escalate,
            "notify_team":      self._handle_notify_team,
        }
        if action.name not in handlers:
            return f"Unknown action: {action.name}", -0.1, "Invalid action"
        return handlers[action.name](action)

    def _handle_check_logs(self, action: Action):
        target = action.target or "all"
        logs = self._scenario["logs"].get(target, ["No logs found for this service"])
        return f"Logs for {target}: {logs}", 0.1, "Investigated logs"

    def _handle_check_metrics(self, action: Action):
        target = action.target or "all"
        services = {s["name"]: s for s in self._scenario["services"]}
        if target in services:
            s = services[target]
            msg = f"Metrics for {target}: CPU={s['cpu']}% Memory={s['memory']}% ErrorRate={s['error_rate']}"
            return msg, 0.1, "Checked metrics"
        return f"No metrics found for {target}", 0.0, "Invalid target"

    def _handle_restart_service(self, action: Action):
        target = action.target
        root_cause = self._scenario["root_cause"]
        if root_cause["type"] == "service_crash" and target == root_cause["service"]:
            self._scenario["resolved"] = True
            return f"Restarted {target}. Service is back online.", 0.5, "Correct fix applied"
        elif root_cause["type"] == "bad_deploy":
            return f"Restarted {target} but it keeps failing. Root cause not addressed.", -0.1, "Symptom treated, not root cause"
        return f"Restarted {target}. No significant change.", -0.05, "Unnecessary restart"

    def _handle_rollback_deploy(self, action: Action):
        target = action.target
        root_cause = self._scenario["root_cause"]
        if root_cause["type"] == "bad_deploy" and target == root_cause["service"]:
            self._scenario["resolved"] = True
            return f"Rolled back {target}. System stabilizing.", 0.5, "Correct fix: rollback applied"
        return f"Rolled back {target} but issue persists.", -0.1, "Wrong fix applied"

    def _handle_scale_up(self, action: Action):
        target = action.target
        root_cause = self._scenario["root_cause"]
        if root_cause["type"] == "overload" and target == root_cause["service"]:
            self._scenario["resolved"] = True
            return f"Scaled up {target}. Load distributed, system recovering.", 0.5, "Correct fix: scaled up"
        return f"Scaled up {target} but no improvement.", -0.05, "Unnecessary scale up"

    def _handle_escalate(self, action: Action):
        if self._current_step < 3:
            return "Escalated too early without investigation.", -0.1, "Premature escalation"
        return "Escalated to senior engineer.", 0.05, "Reasonable escalation"

    def _handle_notify_team(self, action: Action):
        if self._scenario.get("resolved"):
            return "Team notified of resolution.", 0.1, "Timely notification"
        return "Team notified of ongoing incident.", 0.05, "Notification sent"

    # ── helpers ──────────────────────────────────────────────────

    def _build_observation(self, last_action_result: str) -> Observation:
        scenario = self._scenario
        services = [Service(**s) for s in scenario["services"]]
        alerts   = [Alert(**a)   for a in scenario["alerts"]]
        return Observation(
            step=self._current_step,
            services=services,
            alerts=alerts,
            recent_logs=scenario["logs"],
            last_action_result=last_action_result,
            actions_remaining=self.max_steps - self._current_step,
            done=self._done
        )

    def _load_scenario(self, task: str) -> Dict:
        from env.scenario_gen import generate_scenario
        return generate_scenario(task)
    
    