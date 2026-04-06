---
title: Incident Response Commander
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# 🚨 Incident Response Commander

An OpenEnv environment simulating production incident response for AI agents.

## What it does
An AI agent acts as an on-call engineer — investigating alerts, diagnosing root causes, and applying the correct fix across 3 difficulty levels.

## Tasks
- Easy: Single service crash — find and restart it
- Medium: Bad deployment causing cascading failures — rollback it
- Hard: Database overload with noisy alerts — find real root cause

## Endpoints
- GET /health — health check
- POST /reset — start new episode
- POST /step — take an action
- GET /state — current environment state
- GET /tasks — list all tasks