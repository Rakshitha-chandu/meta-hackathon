EASY_SCENARIO = {
    "name": "Single Service Crash",
    "description": "The payment service has crashed due to an OOM error.",
    "root_cause": {
        "type": "service_crash",
        "service": "payment-service"
    },
    "resolved": False,
    "services": [
        {"name": "payment-service", "status": "down",    "cpu": 0.0,  "memory": 99.9, "error_rate": 1.0},
        {"name": "auth-service",    "status": "healthy", "cpu": 32.0, "memory": 45.0, "error_rate": 0.0},
        {"name": "api-gateway",     "status": "healthy", "cpu": 28.0, "memory": 40.0, "error_rate": 0.02},
    ],
    "alerts": [
        {"id": "a1", "service": "payment-service", "severity": "critical",
         "message": "Service down: payment-service", "timestamp": "2026-04-01T03:42:00Z"},
    ],
    "logs": {
        "payment-service": [
            "ERROR: OutOfMemoryError at PaymentProcessor.java:142",
            "ERROR: Service failed to start after OOM",
            "WARN:  Memory usage at 99.9% before crash",
        ],
        "auth-service": ["INFO: All systems normal"],
        "api-gateway":  ["INFO: All systems normal"],
    }
}