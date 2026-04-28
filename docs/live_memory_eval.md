# Live Memory Evaluation Strategy

> **STATUS 2026-04-28:** Mechanical smoke tests pass (state plumbing, disk I/O, import chains). No training or recall evaluation has been run. Functional and coherence tests are scaffolded but unexecuted. THENGPT weights are random. All test results come from untrained (random-weight) models.

This document outlines the testing and evaluation procedures for the Live Memory "Ingest, Don't Train" workflow.

## 1. Mechanical Verification

**Goal:** Ensure the software pipeline correctly handles state persistence, passing, and growth.

**Method:** Unit tests in `tests/test_live_memory.py`.
**Metrics:**

- **State Growth:** Does the memory state size increase as more tokens are ingested?
- **State Persistence:** Can the state be saved to disk and loaded back identically?
- **State Injection:** Does the `query.py` script successfully load and use the state during inference?

**Run Command:**

```bash
python -m tests.test_live_memory
```

## 2. Functional Recall Evaluation (Accuracy)

**Goal:** Measure the model's ability to recall specific facts ingested during Phase 2.

**Method:**

1. **Ingest:** Feed a synthetic dataset containing specific key-value pairs (e.g., "At 10:00, the user drank coffee.").
2. **Query:** Ask questions targeting those facts (e.g., "What did the user drink at 10:00?").
3. **Score:** Exact match or semantic similarity of the answer.

**Prerequisites:**

- A pretrained `THENGPT` model (Phase 1 complete). Random weights will produce near-zero recall accuracy.

## 3. Coherence and Consistency Check

**Goal:** Verify that the ingested narrative remains consistent over long timelines (Phase 2).

**Method:**

- Use `dev/gen_cairo_data.py` to generate "Coherent" data.
- **Check:** The generator enforces logic (e.g., User cannot be in two places at once; Mood changes gradually).
- **Validation:** During ingestion, monitor the "State traces" count in the logs. It should grow linearly with the number of compressed segments (controlled by `ratio`).

## Current Status

- **Mechanical Verification:** Code implemented and smoke tests passing (`tests/test_live_memory.py`). "Passing" means the code does not crash on random weights — it validates state plumbing, save/load round-trips, and import chains. It does NOT validate that the memory mechanism actually works. These are mechanical smoke tests, not functional memory tests.
- **Data Coherence:** Implemented in `dev/gen_cairo_data.py` (persistent user state). Code-level only; no training run has consumed this data.
- **Recall Accuracy:** Pending full pretraining of the `THENGPT` model. Not yet executed.
