ne---
description: Check system and service status
argument-hint: "[service-name]"
allowed-tools: Bash, Read
---

Check the status of SqueezeFlow Trader services.

## Usage
- `/status` - Check all services
- `/status influxdb` - Check specific service
- `/status redis` - Check Redis status

## What it checks
1. Docker container status
2. Service health endpoints
3. Database connectivity
4. Recent errors in logs

The command will show:
- Running services
- Service health status
- Recent log entries
- Any error conditions