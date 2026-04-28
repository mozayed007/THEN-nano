# Changelog: Memory Metadata, Early Recall Benchmark, and Documentation Tightening

## Summary

This update focuses on closing a concrete persistence bug in the disk-backed memory path, improving validation coverage, adding a minimal local benchmark for causal memory recall, and tightening project docs so they do not claim empirical advantages that the current repository has not yet established.

## Why This Change Was Needed

The repository had four practical issues:

1. `DiskTieredMemory` could write traces to disk but did not reliably recover the valid trace count on reload.
2. The existing unit tests validated state mechanics more than actual retrieval-path behavior.
3. There was no small, self-contained local benchmark for testing whether persistent state helps recall on a controlled task.
4. Several docs described Live Memory as if it were already shown to be better than RAG or SFT, even though the current repository mostly validates mechanism and flow rather than benchmarked superiority.

## Code Changes

### 1. Persisted Disk Memory Metadata

Updated:

- `nanochat/memory_manager.py`
- `portable_memory/memory_manager.py`

Changes:

- Added sidecar metadata files at `*.meta.json`
- Persisted:
  - `head`
  - `max_traces`
  - `d_model`
  - `dtype`
- Restored `head` when reloading an existing memory stream
- Added validation checks for metadata mismatches
- Persisted metadata during `append()` and `save()`
- Reload now refreshes both the memmap view and metadata-backed trace count

Impact:

- Query-time loads can now recover the valid memory length instead of defaulting to an empty memory view
- This removes a critical blocker for end-to-end ingest → save → query validation through the disk-backed path

### 2. Stronger Live Memory Tests

Updated:

- `tests/test_live_memory.py`

Changes:

- Increased test model depth from `n_layer=2` to `n_layer=4`
- This ensures at least one layer reaches the retrieval branch under the default `ratio=3`
- Added a disk roundtrip test for `DiskTieredMemory`
- The new roundtrip test verifies:
  - traces are written
  - `head` persists across reload
  - retrieval returns the expected stored vector for a matching query

Impact:

- Test coverage is now better aligned with the actual retrieval architecture
- The most important persistence bug now has direct regression coverage

### 3. Tiny Local Recall Benchmark Scaffold

Added:

- `scripts/tiny_recall_benchmark.py`

What it does:

- Creates small contiguous two-chunk synthetic episodes
- Trains and evaluates four conditions:
  - `GPT` baseline
  - `THENGPT` with reset state
  - `THENGPT` with persistent state
  - `THENGPT` with shuffled-state negative control
- Reports exact-match accuracy on a minimal recall-style task

Intent:

- This is not a final benchmark
- It is a first local harness for checking whether persistent state produces a measurable recall advantage under controlled conditions

## Documentation Changes

Updated:

- `docs/plan/early_validation_and_scaling_plan.md`
- `docs/concepts/then_architecture_public.md`
- `docs/concepts/live_memory_vs_sft.md`
- `docs/concepts/causal_memory_validation.md`
- `docs/critique/critique_and_roadmap.md`
- `docs/critique/critique_loop_4.md`

Changes:

- Refreshed the standing execution plan so it reflects completed reliability work, the existence of the tiny recall benchmark scaffold, and the current near-term decision gate
- Removed or softened wording that implied proven superiority over RAG or SFT
- Reframed comparisons as architectural differences rather than benchmark claims
- Replaced hard claims such as guaranteed economics, zero forgetting, or proven superiority with narrower and more defensible language
- Made the docs explicit that current validation is still mostly mechanical and that stronger causal benchmarks are still needed
- Added a dedicated concept note for causal memory validation so the project has a durable definition of what counts as real evidence versus mechanism-only progress
- Added a new critique loop capturing the improved reliability state and the remaining evidence gap

Impact:

- The docs now better match the current state of the repository
- The project is described as a promising architecture direction rather than a fully validated replacement for existing approaches
- The repository now has a synchronized set of plan, concept, critique, and changelog documents for the current validation-focused session

## Current Status After This Update

The repository is now in a better position to validate the Live Memory thesis because:

- saved disk-backed memory can recover its valid trace count
- retrieval-path behavior has better unit coverage
- a minimal local recall benchmark exists
- the execution plan and assessment docs now clearly define causal validation as the next decision gate
- public-facing project claims are more disciplined and evidence-aligned

## Recommended Next Step

Run the new validation path in order:

1. Run the updated unit tests
2. Run `scripts/tiny_recall_benchmark.py`
3. Inspect whether persistent-state accuracy materially exceeds reset-state and shuffled-state conditions
4. If signal is weak, strengthen the synthetic task before scaling infrastructure work further
