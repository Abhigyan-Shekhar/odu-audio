# Next Actions

## Immediate: Verification (Do This First)

**DO NOT create more documentation. Verify what exists.**

### 1. Navigate to Repository

```bash
cd /Users/abhigyanshekhar/Desktop/odu-audio/
```

### 2. Verify Files Exist

```bash
# List all documentation
ls -lh *.md

# List implementation files
find src/ -name "*.py" -ls

# List configuration
ls -lh configs/*.yaml
```

**Expected**: All files listed in `FILE_INVENTORY.md` should exist.

### 3. Check Python Syntax

```bash
# Test each Python file compiles
python3 -m py_compile src/audio_pipeline/__init__.py
python3 -m py_compile src/audio_pipeline/schemas/feature_record.py
python3 -m py_compile src/audio_pipeline/schemas/speaker_attribution.py  
python3 -m py_compile src/audio_pipeline/runtime/degradation.py
```

**Expected**: No syntax errors.

### 4. Verify Imports Work

```bash
# Set PYTHONPATH
export PYTHONPATH=/Users/abhigyanshekhar/Desktop/odu-audio:$PYTHONPATH

# Test imports
python3 -c "from src.audio_pipeline.schemas.feature_record import AcousticFeatureRecord; print('feature_record: OK')"
python3 -c "from src.audio_pipeline.schemas.speaker_attribution import SpeakerAttribution; print('speaker_attribution: OK')"
python3 -c "from src.audio_pipeline.runtime.degradation import DegradationManager; print('degradation: OK')"
```

**Expected**: All imports succeed with "OK" messages.

### 5. Validate YAML

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/streaming_default.yaml')); print('YAML: OK')"
```

**Expected**: "YAML: OK"

### 6. Count Lines

```bash
# Count Python implementation lines
find src/audio_pipeline -name "*.py" -exec wc -l {} + | tail -1
```

**Expected**: ~610 lines total (verify reported inventory).

### 7. Initialize or Verify Version Control

```bash
# Check if already initialized
if [ -d .git ]; then
    echo "Git repository exists"
    git status
else
    echo "Initializing Git repository"
    git init
fi

# Establish baseline commit
git add .
git commit -m "Architecture proposal and implementation scaffold - awaiting verification"
git log --oneline
```

**Expected**: Repository initialized or verified, baseline commit created.

---

## If Verification Fails

### Syntax Errors
→ Fix syntax errors immediately  
→ Re-run verification

### Import Errors  
→ Check Python version (requires 3.10+)  
→ Check `PYTHONPATH` is set correctly  
→ Check dataclasses are imported correctly

### Missing Files
→ Compare against `FILE_INVENTORY.md`  
→ Identify which files are missing  
→ Do NOT proceed to Milestone 0 until all files exist

---

## If Verification Succeeds

### Recommended Order

1. **Cross-functional contract review** (1-2 days)
   - Schedule meeting with all 7 stakeholder roles
   - Review `CONTRACTS.md` as group
   - Document feedback
   - Approve as v0.1-provisional OR request changes

2. **Assign Milestone 0 owner** (immediate)
   - Identify directly responsible engineer
   - Assign reviewer
   - Set target completion date (1 week)

3. **Begin Milestone 0 execution** (Week 1)
   - Create branch: `git checkout -b milestone-0-engineering-baseline`
   - Set up `pyproject.toml`
   - Configure tooling (black, ruff, mypy, pytest)
   - **Implement** critical behavioral tests
   - Set up CI pipeline
   - Produce evidence bundle
   - Obtain acceptance sign-off

4. **Assign Milestone 1 owner** (after Milestone 0)
   - Identify directly responsible engineer
   - Assign reviewers (data owner, ML owner, tech lead)
   - Set target completion date (2 weeks)

5. **Begin Milestone 1 execution** (Week 2-3)
   - Create branch: `git checkout -b milestone-1-wav-to-parquet`
   - **Implement** WAV-to-Parquet converter
   - **Implement** all 40+ acceptance tests
   - Validate determinism
   - Produce evidence bundle with golden fixtures
   - Stabilize contracts as v1.0
   - Obtain acceptance sign-off

---

## Evidence Bundle Requirements

### Milestone 0 Evidence

Must produce:
```
artifacts/milestone-0/
├── test-report.xml              (pytest JUnit XML)
├── coverage.xml                 (coverage report)
├── dependency-lockfile          (poetry.lock or requirements-lock.txt)
├── configuration-snapshot.yaml  (configs used)
├── checksums.txt                (file hashes)
└── acceptance-report.md         (narrative + sign-offs)
```

### Milestone 1 Evidence

Must produce:
```
artifacts/milestone-1/
├── test-report.xml              (all 40+ tests passing)
├── benchmark.json               (performance measurements)
├── sample-output.parquet        (example conversion)
├── golden-fixtures/             (reference outputs)
│   ├── 8khz-mono.parquet
│   ├── 16khz-mono.parquet
│   ├── 44_1khz-stereo.parquet
│   └── ...
├── schema-v1.0.json             (finalized schema)
├── tolerance-spec.md            (numeric tolerances documented)
└── acceptance-report.md         (narrative + sign-offs)
```

---

## Success Criteria

### Verification Phase (Today)
✅ All files exist  
✅ Python syntax valid  
✅ Imports succeed  
✅ YAML validates  
✅ Line counts match (±10%)  
✅ Git repository initialized

### Milestone 0 (Week 1)
✅ CI passes  
✅ Critical behavioral tests implemented and passing  
✅ ≥80% line coverage  
✅ Evidence bundle complete  
✅ Contracts approved as v0.1-provisional  
✅ Acceptance sign-off obtained

### Milestone 1 (Week 2-3)
✅ 40+ acceptance tests implemented and passing  
✅ CLI tool works: `audio-pipeline wav-to-parquet input.wav output.parquet`  
✅ Deterministic on reference environment  
✅ Evidence bundle with golden fixtures  
✅ Contracts stabilized as v1.0  
✅ Acceptance sign-off obtained

---

## Red Flags

🚩 **More documentation created** → Stop, verify and implement instead  
🚩 **Tests planned but not implemented** → Implement tests, don't just plan  
🚩 **No evidence bundle** → Milestone not complete  
🚩 **No acceptance sign-off** → Milestone not accepted  
🚩 **Claiming "complete" without tests** → Inappropriate claim  
🚩 **Contract frozen before prototype** → Use v0.1-provisional instead  
🚩 **Timeline treated as commitment** → It's a target with caveats  

---

## Contact and Escalation

### For Verification Issues
- **Syntax errors**: Fix immediately, re-run
- **Missing files**: Check `FILE_INVENTORY.md`, do not proceed
- **Import errors**: Check Python version, PYTHONPATH

### For Milestone 0 Issues
- **CI failures**: Debug, fix, commit, repeat
- **Coverage gaps**: Add tests, do not fake coverage
- **Timeline slips**: Communicate early, adjust targets

### For Contract Review Issues
- **Disagreement on schema**: Document alternatives, choose one
- **Missing fields discovered**: Add to v0.1-provisional, note for v1.0
- **Approval delays**: Escalate to technical lead

---

## What NOT to Do

❌ Create more summary documents  
❌ Write more architecture explanations  
❌ Plan tests without implementing them  
❌ Claim completion without evidence  
❌ Freeze contracts before prototype  
❌ Treat targets as commitments  
❌ Skip verification and jump to implementation  
❌ Proceed to Milestone 1 without completing Milestone 0  

---

## What TO Do

✅ Run verification commands **right now**  
✅ Fix any issues found  
✅ Initialize git if verification passes  
✅ Schedule contract review  
✅ Assign Milestone 0 owner  
✅ Begin implementation (not more planning)  
✅ Produce evidence at each milestone  
✅ Obtain sign-offs before proceeding  

---

## Timeline

| Activity | Duration | Start | End |
|----------|----------|-------|-----|
| Verification | 1 hour | Today | Today |
| Contract review | 1-2 days | This week | This week |
| Milestone 0 | 1 week | Next week | Week 1 end |
| Milestone 1 | 2 weeks | Week 2 | Week 3 end |

**First engineering proof point**: Completion of Milestone 1—a passing, deterministic WAV-to-Parquet integration test with an evidence bundle.

---

## Final Checklist

- [ ] Ran all verification commands
- [ ] All verification checks passed
- [ ] Git repository initialized
- [ ] Contract review scheduled
- [ ] Milestone 0 owner assigned
- [ ] Milestone 0 target date set
- [ ] Evidence bundle requirements understood
- [ ] Acceptance process defined
- [ ] Ready to begin implementation

**If all boxes checked**: Proceed to Milestone 0 execution.  
**If any box unchecked**: Stop and complete verification first.

---

**The next activity is verification and implementation, not more documentation.**
