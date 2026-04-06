HARD_SCENARIO = {
    "name": "Multi-System Incident with Noise",
    "description": "Database overload causing API timeouts. Noisy alerts across unrelated services.",
    "root_cause": {
        "type": "overload",
        "service": "postgres-db"
    },
    "resolved": False,
    "services": [
        {"name": "postgres-db",   "status": "degraded", "cpu": 99.0, "memory": 95.0, "error_rate": 0.9},
        {"name": "api-gateway",   "status": "degraded", "cpu": 70.0, "memory": 60.0, "error_rate": 0.6},
        {"name": "cache-service", "status": "healthy",  "cpu": 20.0, "memory": 30.0, "error_rate": 0.0},
        {"name": "email-service", "status": "healthy",  "cpu": 15.0, "memory": 25.0, "error_rate": 0.0},
    ],
    "alerts": [
        {"id": "a1", "service": "postgres-db",   "severity": "critical",
         "message": "DB CPU at 99%, connections maxed",        "timestamp": "2026-04-01T04:00:00Z"},
        {"id": "a2", "service": "api-gateway",   "severity": "high",
         "message": "API response time >5s",                   "timestamp": "2026-04-01T04:01:00Z"},
        {"id": "a3", "service": "cache-service", "severity": "low",
         "message": "Cache miss rate slightly elevated",        "timestamp": "2026-04-01T04:01:30Z"},
        {"id": "a4", "service": "email-service", "severity": "low",
         "message": "Email queue slightly delayed",             "timestamp": "2026-04-01T04:02:00Z"},
    ],
    "logs": {
        "postgres-db":   [
            "ERROR: Max connections reached (500/500)",
            "ERROR: Query timeout after 30s",
            "WARN:  Slow query log filling up",
        ],
        "api-gateway":   [
            "ERROR: Upstream timeout from postgres-db",
            "WARN:  Request queue growing",
        ],
        "cache-service": ["INFO: Normal operation"],
        "email-service": ["INFO: Normal operation"],
    }
}