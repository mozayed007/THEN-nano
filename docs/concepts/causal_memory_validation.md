# Concept: Causal Memory Validation

## Purpose

This document defines the validation framing that should govern the next stage of the Live Memory project.

The goal is to separate mechanical progress from actual evidence that the model uses external state causally.

## The Problem

The repository now supports more of the memory mechanism than it did earlier:

- disk-backed memory can persist metadata needed to reload valid trace counts
- retrieval-path tests are stronger than before
- a small local benchmark scaffold exists

However, those improvements do not by themselves prove the central thesis.

A memory architecture is not validated merely because:

- traces can be written
- traces can be reloaded
- retrieval code executes without crashing
- state grows over time

The missing proof is behavioral.

## The Main Idea

Causal memory validation means the project should be judged by whether correct external state changes model behavior in the expected direction under controlled conditions.

The strongest early validation pattern is:

1. The model succeeds when the correct persistent state is available.
2. The same or similar model degrades when memory is reset.
3. The model also degrades when memory is corrupted or shuffled.
4. The answer is not recoverable from local context alone.

Under this framing, the right benchmark question is not "does memory exist?" but:

- does usable external state cause better recall than controls on a task designed to require that state?

## Scope and Boundaries

This concept is about validation discipline, not about a new storage mechanism.

It does not claim that Live Memory is already better than RAG, SFT, or prompt stuffing.

It does not claim that a tiny synthetic benchmark is sufficient for broad product conclusions.

It does define the minimum evidence needed before broader architecture claims become credible.

## Implications

This framing changes how near-term work should be prioritized.

### 1. Mechanical reliability is necessary but not sufficient

Persistence fixes, reload tests, and retrieval plumbing matter because they remove false negatives and false positives.

But once those are in place, the priority should shift from implementation expansion to causal evaluation.

### 2. Small controlled tasks are the right first proof

The first benchmark should preserve temporal structure and make the answer depend on earlier state rather than current context.

That is why contiguous episodic recall tasks and explicit control conditions matter more than noisy large-scale data at this stage.

### 3. Negative controls are part of the claim

Reset-state and shuffled-state conditions are not optional extras.

They are part of the definition of causal evidence because they test whether state content matters rather than the mere presence of a memory-capable architecture.

### 4. Documentation should follow evidence

Project language should stay narrow until the benchmark shows a real signal.

The repository can currently claim a more reliable prototype and a clearer validation path.

It should not yet claim demonstrated superiority over simpler alternatives.

## Open Questions

- Does the tiny recall benchmark actually show a persistent-state advantage when run end to end?
- Is the current synthetic task strong enough to force memory use rather than shortcut learning?
- How much of the remaining risk comes from training continuity versus retrieval quality?
- When a failure occurs, do existing trace diagnostics explain whether the model retrieved the wrong trace or ignored memory entirely?
