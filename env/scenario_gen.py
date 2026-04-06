import random
from typing import Dict

# ── Service name pools ────────────────────────────────────────
SERVICES = [
    "payment-service", "auth-service", "order-service",
    "checkout-service", "inventory-service", "notification-service",
    "email-service", "api-gateway", "postgres-db", "cache-service",
    "search-service", "user-service", "billing-service", "shipping-service"
]

# ── Log templates per failure type ───────────────────────────
CRASH_LOGS = [
    "ERROR: OutOfMemoryError at {service} — heap space exhausted",
    "ERROR: Service failed to start after OOM crash",
    "WARN:  Memory usage at {pct}% before crash",
    "ERROR: Segmentation fault in {service} process",
    "ERROR: Process killed by OOMKiller",
]

DEPLOY_LOGS = [
    "ERROR: NullPointerException in Handler after deploy {version}",
    "ERROR: Repeated crashes since deploy {version}",
    "INFO:  Deploy {version} applied at {time}Z",
    "ERROR: Config mismatch introduced in {version}",
    "WARN:  Rollback recommended for {version}",
]

OVERLOAD_LOGS = [
    "ERROR: Max connections reached ({conn}/{conn})",
    "ERROR: Query timeout after {sec}s",
    "WARN:  Slow query log filling up",
    "ERROR: CPU throttling active",
    "WARN:  Request queue depth at {queue}",
]

HEALTHY_LOGS = [
    "INFO: All systems normal",
    "INFO: Heartbeat OK",
    "INFO: No anomalies detected",
]

NOISE_MESSAGES = [
    "Cache miss rate slightly elevated",
    "Email queue slightly delayed",
    "Minor latency spike detected",
    "Disk I/O slightly above baseline",
    "Background job took longer than usual",
]

# ── Helpers ───────────────────────────────────────────────────

def pick(pool, exclude=None):
    exclude = exclude or []
    return random.choice([x for x in pool if x not in exclude])

def fill_logs(templates, service, n=3, **kwargs):
    chosen = random.sample(templates, min(n, len(templates)))
    result = []
    for t in chosen:
        result.append(t.format(
            service=service,
            pct=random.randint(90, 99),
            version=f"v{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}",
            time=f"0{random.randint(1,5)}:{random.randint(10,59)}:00",
            conn=random.choice([200, 500, 1000]),
            sec=random.randint(10, 60),
            queue=random.randint(500, 5000),
            **kwargs
        ))
    return result

# ── Scenario builders ─────────────────────────────────────────

def generate_easy_scenario() -> Dict:
    """Single service crash — obvious from logs."""
    broken = pick(SERVICES)
    others = random.sample([s for s in SERVICES if s != broken], 2)

    services = [
        {"name": broken,    "status": "down",    "cpu": 0.0,
         "memory": round(random.uniform(95, 100), 1), "error_rate": 1.0},
        {"name": others[0], "status": "healthy", "cpu": round(random.uniform(20, 45), 1),
         "memory": round(random.uniform(30, 55), 1),  "error_rate": 0.0},
        {"name": others[1], "status": "healthy", "cpu": round(random.uniform(20, 45), 1),
         "memory": round(random.uniform(30, 55), 1),  "error_rate": 0.0},
    ]

    alerts = [
        {"id": "a1", "service": broken, "severity": "critical",
         "message": f"Service down: {broken}",
         "timestamp": f"2026-04-0{random.randint(1,5)}T0{random.randint(1,5)}:{random.randint(10,59)}:00Z"}
    ]

    logs = {
        broken:    fill_logs(CRASH_LOGS, broken),
        others[0]: [pick(HEALTHY_LOGS)],
        others[1]: [pick(HEALTHY_LOGS)],
    }

    return {
        "name": "Single Service Crash",
        "description": f"{broken} has crashed due to an OOM error.",
        "root_cause": {"type": "service_crash", "service": broken},
        "resolved": False,
        "services": services,
        "alerts": alerts,
        "logs": logs,
    }


def generate_medium_scenario() -> Dict:
    """Bad deployment causing cascading failure."""
    broken   = pick(SERVICES)
    affected = pick(SERVICES, exclude=[broken])
    healthy  = pick(SERVICES, exclude=[broken, affected])
    version  = f"v{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}"
    hour     = f"0{random.randint(1,5)}:{random.randint(10,59)}:00"

    services = [
        {"name": broken,   "status": "down",    "cpu": round(random.uniform(70, 90), 1),
         "memory": round(random.uniform(60, 80), 1), "error_rate": 1.0},
        {"name": affected, "status": "degraded","cpu": round(random.uniform(50, 70), 1),
         "memory": round(random.uniform(50, 70), 1), "error_rate": round(random.uniform(0.5, 0.9), 2)},
        {"name": healthy,  "status": "healthy", "cpu": round(random.uniform(20, 40), 1),
         "memory": round(random.uniform(30, 50), 1), "error_rate": 0.0},
    ]

    alerts = [
        {"id": "a1", "service": broken,   "severity": "critical",
         "message": f"{broken} down after deploy {version}",
         "timestamp": f"2026-04-01T{hour}Z"},
        {"id": "a2", "service": affected, "severity": "high",
         "message": f"{affected} degraded — dependency failure",
         "timestamp": f"2026-04-01T{hour}Z"},
    ]

    deploy_logs = [
        f"ERROR: NullPointerException in Handler after deploy {version}",
        f"ERROR: Repeated crashes since deploy {version}",
        f"INFO:  Deploy {version} applied at {hour}Z",
    ]

    logs = {
        broken:   deploy_logs,
        affected: [
            f"ERROR: Dependency {broken} not responding",
            f"WARN:  Retrying {broken} connection...",
        ],
        healthy: [pick(HEALTHY_LOGS)],
    }

    return {
        "name": "Cascading Failure from Bad Deploy",
        "description": f"Bad deployment {version} to {broken} causing cascading failures.",
        "root_cause": {"type": "bad_deploy", "service": broken},
        "resolved": False,
        "services": services,
        "alerts": alerts,
        "logs": logs,
    }


def generate_hard_scenario() -> Dict:
    """DB overload with noisy unrelated alerts."""
    db      = pick(["postgres-db", "mysql-db", "mongo-db", "redis-db"])
    gateway = pick(["api-gateway", "load-balancer", "nginx-proxy"])
    noisy1  = pick(SERVICES, exclude=[db, gateway])
    noisy2  = pick(SERVICES, exclude=[db, gateway, noisy1])
    conn    = random.choice([200, 500, 1000])

    services = [
        {"name": db,      "status": "degraded", "cpu": round(random.uniform(92, 99), 1),
         "memory": round(random.uniform(88, 97), 1), "error_rate": round(random.uniform(0.7, 0.95), 2)},
        {"name": gateway, "status": "degraded", "cpu": round(random.uniform(60, 80), 1),
         "memory": round(random.uniform(55, 75), 1), "error_rate": round(random.uniform(0.4, 0.7), 2)},
        {"name": noisy1,  "status": "healthy",  "cpu": round(random.uniform(10, 30), 1),
         "memory": round(random.uniform(20, 40), 1), "error_rate": 0.0},
        {"name": noisy2,  "status": "healthy",  "cpu": round(random.uniform(10, 30), 1),
         "memory": round(random.uniform(20, 40), 1), "error_rate": 0.0},
    ]

    alerts = [
        {"id": "a1", "service": db,      "severity": "critical",
         "message": f"DB CPU at {services[0]['cpu']}%, connections maxed",
         "timestamp": "2026-04-01T04:00:00Z"},
        {"id": "a2", "service": gateway, "severity": "high",
         "message": "API response time >5s",
         "timestamp": "2026-04-01T04:01:00Z"},
        {"id": "a3", "service": noisy1,  "severity": "low",
         "message": pick(NOISE_MESSAGES),
         "timestamp": "2026-04-01T04:01:30Z"},
        {"id": "a4", "service": noisy2,  "severity": "low",
         "message": pick(NOISE_MESSAGES),
         "timestamp": "2026-04-01T04:02:00Z"},
    ]

    logs = {
        db: [
            f"ERROR: Max connections reached ({conn}/{conn})",
            f"ERROR: Query timeout after {random.randint(10,60)}s",
            "WARN:  Slow query log filling up",
        ],
        gateway: [
            f"ERROR: Upstream timeout from {db}",
            "WARN:  Request queue growing",
        ],
        noisy1: [pick(HEALTHY_LOGS)],
        noisy2: [pick(HEALTHY_LOGS)],
    }

    return {
        "name": "Multi-System Incident with Noise",
        "description": f"{db} overload causing {gateway} timeouts. Noisy alerts present.",
        "root_cause": {"type": "overload", "service": db},
        "resolved": False,
        "services": services,
        "alerts": alerts,
        "logs": logs,
    }


# ── Main entry point ──────────────────────────────────────────

def generate_scenario(task: str) -> Dict:
    generators = {
        "easy":   generate_easy_scenario,
        "medium": generate_medium_scenario,
        "hard":   generate_hard_scenario,
    }
    if task not in generators:
        raise ValueError(f"Unknown task: {task}. Choose from: easy, medium, hard")
    return generators[task]()