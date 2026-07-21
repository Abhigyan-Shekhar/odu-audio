# Clinical Intended-Use Review

**Date:** 2026-07-21
**Reviewer:** [Pending Sign-off]
**Status:** Under Review

## 1. Intended Use
The odu-audio pipeline is intended to serve as a high-fidelity acoustic feature extraction engine. It transforms unstructured clinical audio into a structured temporal stream of acoustic parameters (eGeMAPS) and affect embeddings (emotion2vec) specific to the patient. Its primary intended use is to provide data for downstream Machine Learning models and clinical research tools.

## 2. Acoustic Affect vs. Emotion
The system extracts features correlated with *acoustic affect* (e.g., vocal arousal, prosodic variation). It does **not** evaluate, classify, or diagnose a patient's internal emotional or psychiatric state. Clinicians must understand that vocal expressions of affect are highly confounded by language, culture, neurodivergence, and physical illness.

## 3. Prohibited Uses
This pipeline and any downstream models built upon its output **must not** be used for:
- Autonomous clinical diagnosis without expert human review.
- Suicide risk assessments, involuntary commitment decisions, or emergency dispatch without human oversight.
- Insurance coverage, treatment denial, or healthcare rationing decisions.
- Staff surveillance, disciplinary monitoring, or legal/immigration proceedings.

## 4. Model Limitations
- **Diarization Overlap:** The system deliberately defaults to `OVERLAP` or `UNKNOWN` when speakers cross-talk. It is heavily biased against false attribution to the patient, meaning it will drop valid patient speech if a clinician talks over them.
- **YAMNet Distortions:** The YAMNet model was trained on YouTube audio (Audioset). Its performance on clinical distress events (crying, gasping) in noisy hospital rooms is not perfectly calibrated and may yield false positives.

**Approval:** [ ] Approved
