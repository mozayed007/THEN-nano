# Plan: Early Validation and Scaling Path for Live Memory

## Purpose

This document translates the current Live Memory assessment into an exact execution plan.

The priority is not to expand the architecture further before validation. The priority is to run the smallest local experiment that can prove or falsify the central idea:

Can the model use external memory state causally to answer questions that it could not answer from local context alone?

## Strategic Principle

Validation should proceed in this order:

1. Validate the memory mechanism.
2. Validate the training signal.
3. Validate the retrieval infrastructure.
4. Validate scaling behavior.
5. Validate realistic data.
6. Validate usefulness against baselines.

This order matters.

If the mechanism is not proven on a tiny synthetic task, larger infrastructure work will create noise rather than clarity.

## Immediate Goal

Execute the smallest existing local experiment end to end and determine whether the repository shows any causal recall signal from persistent external state.

- Run the updated unit tests and the tiny recall benchmark.
- Check whether `THENGPT` with persistent memory state materially outperforms reset-state and shuffled-state controls.
- Use the result to decide whether to strengthen the task or proceed to broader retrieval diagnostics.

## Current Session Progress

CRITICAL NOTE: All "completed" tasks below refer to code implementation only. The THENGPT model has never been trained. The `tiny_recall_benchmark.py` scaffold exists but has never been executed. The entire validation pipeline remains at Step 0.

- `DiskTieredMemory` now persists and restores valid trace metadata needed to recover the correct memory length on reload.
- Disk roundtrip and retrieval-oriented tests now exist.
- A tiny local recall benchmark scaffold now exists with baseline, reset-state, persistent-state, and shuffled-state conditions.
- The main remaining uncertainty is empirical rather than purely mechanical: the benchmark still needs to be run and inspected.

## Phase 0: Fix Known Blockers Before Benchmarking

This phase is no longer just a design requirement. The repository now contains the main reliability fixes needed to make the earliest mechanism benchmark meaningful.

### Task 0.1: Persist valid trace count for disk-tiered memory

Status:

- Implemented in code (not trained): persisted sidecar metadata.
- Reload can now restore `head` instead of treating disk-backed memory as empty.
- Metadata-backed reload can validate shape and dtype assumptions before use.

### Task 0.2: Add disk-memory roundtrip tests

Status:

- Implemented in code (not trained): in `tests/test_live_memory.py`.
- The test covers ingest, flush, reload, and retrieval against a known stored vector.

### Task 0.3: Add a retrieval-focused test configuration

Status:

- Implemented in code (not trained): via a deeper test model configuration that actually reaches retrieval layers.
- The test path is now better aligned with read-path behavior rather than only non-crash state plumbing.

## Phase 1: Build the Smallest Real Benchmark

This phase is the most important one in the entire project.

The repository now has an initial scaffold for this phase. The remaining work is to run it, inspect the signal, and strengthen it only where the results stay ambiguous.

## Benchmark Objective

Construct a synthetic task where:

- the answer appears earlier in time,
- the answer is not present in the current local chunk,
- and the model must rely on external state to solve the task.

## Task 1.1: Create a contiguous synthetic episodic dataset

Do not begin with noisy realistic logs.

Create a dataset of small episodes with explicit structure. Example pattern:

```text
[MEM] favorite_drink = coffee
[FILLER] ... many irrelevant tokens ...
[Q] what_is_favorite_drink ?
[A] coffee
```

Design requirements:

- The memory fact must occur in an earlier chunk.
- The question must occur in a later chunk.
- The answer must not be trivially inferable from local context.
- Multiple keys and distractors should be possible.

Suggested task types:

- key-value recall,
- passkey retrieval,
- identity and preference recall,
- simple temporal fact recall.

## Task 1.2: Use a contiguous dataloader for this benchmark

Do not use the generic best-fit packed loader for the first memory benchmark.

Reason:

- The benchmark must preserve episode continuity exactly.
- Every training step should continue from the previous chunk of the same episode until the episode ends.

Required properties:

- deterministic episode order,
- deterministic chunk boundaries,
- no cross-document mixing within an episode,
- optional explicit reset between episodes.

## Task 1.3: Define exact evaluation conditions

Run at least four conditions.

### Condition A: Baseline GPT

- No memory mechanism.
- Same data and similar parameter scale where possible.

### Condition B: THENGPT with memory reset

- Memory-capable architecture present.
- State reset every step or episode boundary in a way that removes useful long-range memory.

### Condition C: THENGPT with persistent memory state

- The intended Live Memory training and inference condition.

### Condition D: THENGPT with corrupted or shuffled memory state

- Negative control to verify that correct state content matters.

## Task 1.4: Use simple metrics first

Do not rely on perplexity alone.

Primary metrics should be:

- exact match accuracy on the answer token or answer string,
- accuracy versus temporal gap,
- accuracy with distractors,
- accuracy as the number of stored facts grows.

Success criterion:

- Condition C should materially outperform A, B, and D.

If that does not happen, the mechanism is not yet validated.

## Phase 2: Local Tiny-Model Experiment

## Task 2.1: Train a very small model first

Use a tiny local setup to get a fast signal.

Suggested scale:

- small depth,
- short sequence length,
- synthetic data only,
- a few thousand steps or less.

The point is not to maximize quality. The point is to force the memory mechanism to reveal whether it works at all.

## Task 2.2: Keep evaluation write policy clean

During early validation, do not allow uncontrolled self-writing during generation.

Recommended evaluation protocol:

1. Ingest known facts.
2. Freeze the memory state for evaluation.
3. Query the model.
4. Score answers.

This prevents self-generated hallucinations from contaminating the memory store and confusing the signal.

## Task 2.3: Add debugging visibility to traces

Add lightweight trace metadata so failures can be understood.

Recommended metadata:

- episode id,
- chunk index,
- write order,
- optional source text span,
- timestamp or relative position.

This helps answer:

- Did the model retrieve the right trace?
- Did it retrieve a nearby distractor?
- Did retrieval fail entirely?

## Phase 3: Strengthen the Mechanism Benchmark

Once the smallest benchmark shows a positive signal, expand difficulty gradually.

### Task 3.1: Increase temporal gap

Measure recall as the gap grows larger.

### Task 3.2: Increase distractor density

Insert more irrelevant memories between fact and query.

### Task 3.3: Increase memory load

Store multiple facts and test whether the model retrieves the relevant one.

### Task 3.4: Stress write policy

Compare:

- ingest-only writes,
- user-input-only writes,
- and fully live bidirectional writes.

This will clarify how much self-writing harms memory quality.

## Phase 4: Validate the Retrieval Infrastructure Separately

Only after RAM-backed mechanism validation should the disk-tiered retrieval path become a primary focus.

## Task 4.1: RAM versus disk equivalence

Use identical traces and compare:

- recall accuracy,
- retrieval ranking behavior,
- and output consistency.

The disk path should preserve behavior, not redefine it.

## Task 4.2: Measure systems metrics

Track:

- memory footprint,
- retrieval latency,
- write throughput,
- scaling with number of traces,
- and effects of `top_k` and chunk size.

## Task 4.3: Add failure-mode tests

Stress test:

- truncated state files,
- partial writes,
- mismatched dimensions,
- zero-memory state,
- metadata corruption.

## Phase 5: Move to Noisier Episodic Data

Only after controlled synthetic success should the project expand toward realistic timelines.

## Task 5.1: Use synthetic but noisy narratives

Introduce:

- interleaved irrelevant events,
- contradictory statements,
- topic switching,
- and long idle stretches.

Goal:

- measure whether retrieval remains precise under noise.

## Task 5.2: Test consistency-oriented memory tasks

Good candidates:

- stable preferences,
- location continuity,
- named entity persistence,
- long-gap question answering.

## Task 5.3: Evaluate memory contamination

Test how the system behaves when:

- the user says something false once,
- the assistant says something incorrect,
- or later updates contradict earlier memory.

These are realistic memory-management problems.

## Phase 6: Compare Against Simpler Baselines

This phase is essential before making stronger claims.

## Required baselines

At minimum compare against:

1. Vanilla GPT without memory.
2. Prompt stuffing of the relevant prior fact.
3. A simple retrieval baseline.

The project does not need to beat every baseline on every axis. But it should demonstrate at least one clear advantage such as:

- better recall under fixed prompt budget,
- cheaper per-user persistence,
- cleaner deletion semantics,
- or better long-gap recall.

## Proposed Deliverables

The project should aim to produce the following artifacts in order:

### Deliverable 1: Mechanism benchmark

- synthetic contiguous task,
- local training script,
- exact-match metric,
- baseline and ablation results.

### Deliverable 2: Retrieval diagnostics

- trace metadata,
- trace inspection utilities,
- top-k retrieval debug output.

### Deliverable 3: Disk-tiered reliability

- persisted memory metadata,
- reload tests,
- RAM versus disk parity checks.

### Deliverable 4: Noise benchmark

- realistic synthetic narratives,
- distractor-heavy recall evaluation.

### Deliverable 5: Baseline comparison report

- memory model versus simple alternatives,
- clear statement of where Live Memory helps and where it does not.

## Exact Near-Term Next Steps

The recommended immediate order of work is:

1. Run the updated unit tests, especially the disk roundtrip coverage.
2. Run `scripts/tiny_recall_benchmark.py` and capture exact-match results for all four conditions.
3. Confirm whether persistent-state accuracy materially exceeds reset-state and shuffled-state controls.
4. If signal is weak, strengthen the synthetic task before expanding infrastructure work further.
5. Add or expose trace metadata and retrieval inspection utilities so failures reveal which traces were used.
6. Compare RAM-backed and disk-backed behavior on the same stored traces.
7. Only after mechanism signal is clear, expand to noisier episodic data.
8. Only after that make stronger claims versus prompt stuffing, retrieval baselines, RAG, or SFT.

## Decision Rules

To keep the project disciplined, use these decision rules.

### Continue mechanism development if:

- persistent-memory condition clearly beats controls,
- retrieval appears semantically aligned,
- and failures are understandable and debuggable.

### Pause and redesign if:

- memory-enabled models do not beat reset-state controls,
- shuffled-state controls perform similarly to correct-state controls,
- or retrieval becomes too noisy even on tiny synthetic tasks.

### Scale infrastructure only if:

- the RAM-backed mechanism is already working,
- and disk-tiered retrieval preserves that behavior.

## Final Recommendation

The next win for the project is not a larger architecture narrative. It is a crisp, local, causal memory benchmark.

If the model can prove on a tiny synthetic task that it reads and uses external state to answer correctly while the controls fail, the project becomes substantially more credible.

If it cannot do that, the right response is not to scale harder. The right response is to simplify the task until the failure mode becomes obvious.

This is the shortest path to truth for the Live Memory idea.
