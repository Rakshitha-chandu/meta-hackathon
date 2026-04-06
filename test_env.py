from env.environment import IncidentResponseEnv
from env.models import Action

def run_episode(task_name, actions_fn, label="", max_steps=10):
    print(f"\n{'='*55}")
    print(f"TASK: {task_name.upper()}  {label}")
    print('='*55)

    env = IncidentResponseEnv(task=task_name, max_steps=max_steps)  # ← pass it here
    obs = env.reset()
    root_service = env._scenario["root_cause"]["service"]
    root_type    = env._scenario["root_cause"]["type"]
    print(f"Root cause : {root_type} → {root_service}")

    actions      = actions_fn(root_service)
    grade_result = None

    for action in actions:
        obs, reward, done, info = env.step(action)
        print(f"\n  Action : {action.name}({action.target or action.message or ''})")
        print(f"  Result : {obs.last_action_result}")
        print(f"  Reward : {reward.value} — {reward.reason}")
        if done:
            grade_result = info["grade"]
            break

    # Force grade even if episode didn't finish naturally
    if not grade_result:
        from env.grader import IncidentGrader
        grader = IncidentGrader(env._scenario)
        grade_result = grader.grade(
            action_history=env._action_history,
            resolved=env._scenario.get("resolved", False)
        )

    print(f"\n── GRADER RESULT ──────────────────────────")
    print(f"  Final Score : {grade_result['score']} / 1.0")
    print(f"  Passed      : {'✅ YES' if grade_result['passed'] else '❌ NO'}")
    print(f"  Breakdown   :")
    for k, v in grade_result["breakdown"].items():
        bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        print(f"    {k:<28} {bar} {v}")
    print(f"  Feedback    :")
    for f in grade_result["feedback"]:
        print(f"    {f}")

# ── Agent scripts ─────────────────────────────────────────────

def easy_perfect(root_service):
    return [
        Action(name="check_logs",      target=root_service),
        Action(name="notify_team",     message="incident found, fixing"),
        Action(name="restart_service", target=root_service),
    ]

def medium_perfect(root_service):
    return [
        Action(name="check_logs",      target=root_service),
        Action(name="check_metrics",   target=root_service),
        Action(name="notify_team",     message="incident found, fixing"),
        Action(name="rollback_deploy", target=root_service),
    ]

def hard_perfect(root_service):
    return [
        Action(name="check_metrics",   target=root_service),
        Action(name="check_logs",      target=root_service),
        Action(name="notify_team",     message="incident found, fixing"),
        Action(name="scale_up",        target=root_service),
    ]

def bad_agent(root_service):
    return [
        Action(name="restart_service", target="wrong-service"),
        Action(name="restart_service", target="another-wrong"),
        Action(name="scale_up",        target="wrong-service"),
        Action(name="check_logs",      target="wrong-service"),
    ]

def partially_good_agent(root_service):
    """Finds root cause but applies wrong fix."""
    return [
        Action(name="check_logs",      target=root_service),
        Action(name="check_metrics",   target=root_service),
        Action(name="scale_up",        target="wrong-service"),
        Action(name="notify_team",     message="tried to fix"),
    ]

if __name__ == "__main__":
    print("\n🟢 PERFECT AGENTS")
    run_episode("easy",   easy_perfect,   label="— Perfect Agent")
    run_episode("medium", medium_perfect, label="— Perfect Agent")
    run_episode("hard",   hard_perfect,   label="— Perfect Agent")

    print("\n\n🔴 BAD AGENT")
    run_episode("easy", bad_agent, label="— Bad Agent", max_steps=4)

    print("\n\n🟡 PARTIAL AGENT")
    run_episode("medium", partially_good_agent, label="— Partial Agent", max_steps=4)