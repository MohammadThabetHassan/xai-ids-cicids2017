# Implementation Audit - Consistency Check

**Date:** March 2026
**Purpose:** Identify and fix inconsistencies between documentation and actual implementation

---

## Feature Status Table

| Feature | Claimed Status | Actual Status | Evidence | Action Needed |
|---------|---------------|---------------|----------|---------------|
| Docker | PROJECT_STATUS: ❌ Not Started | ✅ Implemented | Dockerfile exists | Fix PROJECT_STATUS |
| FastAPI API | PROJECT_STATUS: ❌ Not Started | ✅ Implemented | api/app.py exists, imports OK | Fix PROJECT_STATUS |
| config.yaml | ROADMAP: Complete | ⚠️ Dead code | Not wired into run_pipeline.py | Wire config or remove |
| Per-class metrics | PROJECT_STATUS: ⚠️ Partial | ✅ Done | In README.md lines 177-195 | Fix PROJECT_STATUS |
| Macro F1 | PROJECT_STATUS: ⚠️ Hidden | ✅ Done | In README.md line 173 | Fix PROJECT_STATUS |
| Cross-validation | PROJECT_STATUS: ❌ Not Started | ✅ Done | --cv-folds flag works | Fix PROJECT_STATUS |
| Calibration curves | PROJECT_STATUS: ❌ Not Started | ✅ Done | --calibration flag works | Fix PROJECT_STATUS |
| PR curves | PROJECT_STATUS: ❌ Not Started | ✅ Done | --pr-curves flag works | Fix PROJECT_STATUS |
| Failure analysis | PROJECT_STATUS: ❌ Not Started | ✅ Done | --failure-analysis flag works | Fix PROJECT_STATUS |
| Class weighting | PROJECT_STATUS: ❌ Not Started | ✅ Done | train.py has balanced weights | Fix PROJECT_STATUS |
| Zenodo data | PROJECT_STATUS: ⚠️ Broken | ✅ Done | download.py has fallback | Fix PROJECT_STATUS |
| Issue templates | ROADMAP: Complete | ✅ Done | .github/ISSUE_TEMPLATE/ | Fix ROADMAP status |
| CHANGELOG | ROADMAP: Complete | ✅ Done | CHANGELOG.md exists | Verify |
| CITATION.cff | ROADMAP: Complete | ✅ Done | CITATION.cff exists | Verify |
| Unit tests | PROJECT_STATUS: ❌ None | ✅ Done | test_evaluation.py - 11 tests | Fix PROJECT_STATUS |
| docs/index.md | - | ⚠️ Outdated | Missing Docker/API mention | Update docs |
| docs/methodology.md | - | ⚠️ Outdated | Missing CV/calibration | Update docs |

---

## Key Inconsistencies Found

### 1. PROJECT_STATUS.md (Most Critical)
- Lines 51-52: Docker/API marked "Not Started" - WRONG
- Lines 79-82: Per-class/macro/CV/calibration marked "Partial/Not Started" - WRONG
- Lines 90-94: Testing marked "None" - WRONG (test_evaluation.py has 11 tests)
- Lines 101-124: "Planned" items that are already done

### 2. docs/index.md
- Missing: Docker, FastAPI, config.yaml, new CLI flags
- Missing: Macro F1, per-class metrics, class imbalance discussion

### 3. docs/methodology.md
- Missing: Cross-validation section
- Missing: Calibration analysis
- Missing: Class imbalance handling (class weights)
- Missing: Failure analysis

### 4. config.yaml
- Created but NOT wired into run_pipeline.py
- This is dead code unless integrated

### 5. ROADMAP.md
- Generally aligned but some status updates needed

---

## Action Items

### Phase A: Fix PROJECT_STATUS.md
- Update all completed items to "Complete"
- Remove items from "Planned" that are done

### Phase B: Fix docs/
- Update docs/index.md with new features
- Update docs/methodology.md with CV, calibration, class weights

### Phase C: Fix config.yaml
- Either wire it into pipeline OR mark it as "reference only" / remove

### Phase D: Verify all features
- Test all new CLI flags work
- Verify tests pass
- Verify API imports

---

## Verification Results (Pre-Fix)

| Test | Result |
|------|--------|
| API imports | ✅ OK |
| Evaluation tests (11) | ✅ All pass |
| Dockerfile | ✅ Exists |
| config.yaml | ⚠️ Dead code |
| --cv-folds flag | ✅ Implemented |
| --pr-curves flag | ✅ Implemented |
| --calibration flag | ✅ Implemented |
| --failure-analysis flag | ✅ Implemented |
