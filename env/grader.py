from typing import Dict, List
from env.models import Reward

# ── Grader weights (must sum to 1.0) ─────────────────────────
WEIGHTS = {
    "root_cause_identified": 0.35,
    "correct_fix_applied":   0.35,
    "no_unnecessary_actions":0.20,
    "team_notified":         0.10,
}

# ── Fix action per root cause type ───────────────────────────
CORRECT_FIX = {
    "service_crash": "restart_service",
    "bad_deploy":    "rollback_deploy",
    "overload":      "scale_up",
}

INVESTIGATION_ACTIONS = {"check_logs", "check_metrics"}
DESTRUCTIVE_ACTIONS   = {"restart_service", "rollback_deploy", "scale_up"}


class IncidentGrader:
    def __init__(self, scenario: Dict):
        self.scenario     = scenario
        self.root_cause   = scenario["root_cause"]
        self.root_service = self.root_cause["service"]
        self.root_type    = self.root_cause["type"]

    def grade(self, action_history: List[Dict], resolved: bool) -> Dict:
        breakdown = {}
        feedback  = []

        # 1. Root cause identified? (checked correct service)
        investigated = self._check_investigation(action_history)
        breakdown["root_cause_identified"] = investigated
        if investigated == 1.0:
            feedback.append("✅ Agent investigated the correct service.")
        elif investigated == 0.5:
            feedback.append("⚠️  Agent investigated services but not root cause.")
        else:
            feedback.append("❌ Agent never investigated any service.")

        # 2. Correct fix applied to correct service?
        fix_score = self._check_fix(action_history, resolved)
        breakdown["correct_fix_applied"] = fix_score
        if fix_score == 1.0:
            feedback.append("✅ Correct fix applied to correct service.")
        elif fix_score == 0.5:
            feedback.append("⚠️  Correct fix action used but on wrong service.")
        else:
            feedback.append("❌ Wrong fix or no fix attempted.")

        # 3. Efficiency — no wasteful destructive actions
        efficiency_score = self._check_efficiency(action_history)
        breakdown["no_unnecessary_actions"] = efficiency_score
        if efficiency_score == 1.0:
            feedback.append("✅ Agent was efficient — no wasteful actions.")
        elif efficiency_score == 0.5:
            feedback.append("⚠️  Agent took some unnecessary actions.")
        else:
            feedback.append("❌ Agent wasted many actions on wrong services.")

        # 4. Team notified?
        notified = 1.0 if any(
            a["name"] == "notify_team" for a in action_history
        ) else 0.0
        breakdown["team_notified"] = notified
        if notified:
            feedback.append("✅ Team was notified.")
        else:
            feedback.append("❌ Team was never notified.")

        # ── Final weighted score ──────────────────────────────
        score = round(sum(
            breakdown[k] * WEIGHTS[k] for k in WEIGHTS
        ), 3)

        return {
            "score":     score,
            "breakdown": breakdown,
            "feedback":  feedback,
            "passed":    score >= 0.5,
            "resolved":  resolved,
        }

    # ── Private helpers ───────────────────────────────────────

    def _check_investigation(self, action_history: List[Dict]) -> float:
        """Did agent investigate the root cause service specifically?"""
        investigated_root = any(
            a["name"] in INVESTIGATION_ACTIONS and
            a.get("target") == self.root_service
            for a in action_history
        )
        investigated_any = any(
            a["name"] in INVESTIGATION_ACTIONS
            for a in action_history
        )

        if investigated_root:
            return 1.0
        if investigated_any:
            return 0.5
        return 0.0

    def _check_fix(self, action_history: List[Dict], resolved: bool) -> float:
        """Did agent apply correct fix to correct service?"""
        correct_action = CORRECT_FIX.get(self.root_type)
        if not correct_action:
            return 0.0

        correct_fix_on_correct_service = any(
            a["name"] == correct_action and
            a.get("target") == self.root_service
            for a in action_history
        )
        correct_fix_on_wrong_service = any(
            a["name"] == correct_action and
            a.get("target") != self.root_service
            for a in action_history
        )

        if correct_fix_on_correct_service and resolved:
            return 1.0
        if correct_fix_on_wrong_service:
            return 0.5
        return 0.0

    def _check_efficiency(self, action_history: List[Dict]) -> float:
        """Penalise destructive actions on wrong services."""
        wrong_destructive = sum(
            1 for a in action_history
            if a["name"] in DESTRUCTIVE_ACTIONS and
            a.get("target") != self.root_service
        )

        if wrong_destructive == 0:
            return 1.0
        if wrong_destructive == 1:
            return 0.5
        return 0.0