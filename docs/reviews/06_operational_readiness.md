# Operational Readiness Review

**Date:** 2026-07-21
**Reviewer:** [Pending Sign-off]
**Status:** Under Review

## 1. Monitoring Strategy
When deployed in a clinical setting, the following metrics must be actively monitored:
- **CPU 95th Percentile Latency:** (`latency_p95_ms` exposed by `DegradationManager.get_status_summary()`).
- **Degradation Level Tracking:** Alerts should fire if the system remains at `LEVEL_3` or higher for more than 5 continuous minutes, as this indicates sustained CPU saturation and loss of critical features.
- **Queue Drop Rate:** Tracking the number of `DropReason.QUEUE_FULL` emissions. Any value > 0 indicates hard data loss due to backpressure.

## 2. Deployment Checklist
- [ ] Hardware verified (Minimum 4 CPU cores, 8GB RAM).
- [ ] OS permissions configured (Read-only access to `~/.cache/` models).
- [ ] HuggingFace token provisioned as a secure environment variable (`HF_TOKEN`).
- [ ] Downstream Parquet storage configured with encryption at rest.

## 3. Incident Response & Rollback
- **Failure Mode:** If the pipeline crashes entirely, the audio is irretrievably lost for that session (no raw audio is persisted to disk).
- **Rollback:** The system operates statelessly per session. Rollbacks to previous pipeline versions simply require restarting the Docker container/Python process before the next clinical session begins.

**Approval:** [ ] Approved
