MEDIUM_SCENARIO = {
    "name": "Cascading Failure from Bad Deploy",
    "description": "A bad deployment to order-service is causing cascading failures.",
    "root_cause": {
        "type": "bad_deploy",
        "service": "order-service"
    },
    "resolved": False,
    "services": [
        {"name": "order-service",    "status": "down",    "cpu": 85.0, "memory": 70.0, "error_rate": 1.0},
        {"name": "checkout-service", "status": "degraded","cpu": 60.0, "memory": 65.0, "error_rate": 0.8},
        {"name": "auth-service",     "status": "healthy", "cpu": 30.0, "memory": 40.0, "error_rate": 0.0},
    ],
    "alerts": [
        {"id": "a1", "service": "order-service",    "severity": "critical",
         "message": "order-service down after deploy v2.4.1", "timestamp": "2026-04-01T02:15:00Z"},
        {"id": "a2", "service": "checkout-service", "severity": "high",
         "message": "checkout-service degraded",              "timestamp": "2026-04-01T02:16:00Z"},
    ],
    "logs": {
        "order-service": [
            "ERROR: NullPointerException in OrderHandler.java:88 after deploy v2.4.1",
            "ERROR: Repeated crashes since deploy v2.4.1",
            "INFO:  Deploy v2.4.1 applied at 02:14:00Z",
        ],
        "checkout-service": [
            "ERROR: Dependency order-service not responding",
            "WARN:  Retrying order-service connection...",
        ],
        "auth-service": ["INFO: All systems normal"],
    }
}