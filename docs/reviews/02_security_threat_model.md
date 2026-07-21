# Security Threat Model

**Date:** 2026-07-21
**Reviewer:** [Pending Sign-off]
**Status:** Under Review

## 1. Attack Surface

| Threat Vector | Description | Risk Level |
|---------------|-------------|------------|
| **Malicious Audio Input** | Crafted `.wav` files containing NaN/Inf values, extremely high amplitudes, or malformed headers designed to crash extractors or cause buffer overflows in C-libraries. | Medium |
| **Denial of Service (DoS)** | A rapid influx of audio chunks exceeding the real-time processing capacity, aimed at exhausting system memory (OOM). | High |
| **Model Poisoning / Replacement** | Replacing the cached ONNX or PyTorch models in `~/.cache/` with malicious models that execute arbitrary code or skew results. | High |

## 2. Implemented Mitigations

- **Input Sanitization:** Numpy checks and strict dimension bounds are enforced before audio hits C-extensions (like openSMILE).
- **DoS Protection:** The `BoundedQueue` enforces a maximum backlog. `DegradationManager` sheds load to prevent CPU starvation. If the queue is full, new data is dropped, ensuring memory bounds are respected.
- **Model Caches:** By default, YAMNet and other models load from local read-only caches. In a production environment, the cache directory must be secured via OS-level file permissions (`chmod 555`) to prevent runtime modification.

## 3. Residual Risks Accepted

- **Adversarial Audio:** We do not currently implement defenses against sophisticated adversarial audio attacks designed to trick the YAMNet classifier or Silero VAD into false positives. Given the clinical (non-autonomous) nature of the use case, this is deemed an acceptable residual risk.

**Approval:** [ ] Approved
