# Loop 3 Critique: Intensive Code Review

**Date:** 2026-02-22
**Project:** Live Memory (THEN Phase)
**Subject:** `nanochat-then` Codebase Code Review vs Past Critiques and Plan 0

## Executive Summary

Based on an intensive review of the documentation (`docs/`) and the corresponding source code in `nanochat-then`, the project has made significant strides in addressing the major architectural flaws identified in Loop 2. However, while the theoretical "Ingest, Don't Train" plan (Plan 0) is plausible, critical implementation bugs in the inference script and missing infrastructure for memory scaling pose severe blockers to full realization.

---

## 1. Critique Resolution Status

### ✅ 1. The Pretraining Gap

- **Critique:** The model was trained as a standard stateless Transformer, resetting memory at every batch, meaning the attention layer never learned to read from memory.
- **Resolution in Codebase:** **HANDLED.** `scripts/base_train.py` initializes `state` and passes it sequentially through training iterations. It properly implements Truncated Backpropagation Through Time (TBPTT) by detaching `state['traces']` and `state['buffer']` to prevent gradient explosions while maintaining contextual history. `THENGPT.forward` accepts and returns the `state`.

### ✅ 2. The "Mean" Retrieval (Memory Blurring)

- **Critique:** The model retrieved memory by computing `torch.mean` across all traces, blurring all specific context into a generic centroid.
- **Resolution in Codebase:** **HANDLED.** `nanochat/gpt.py` in `HybridTHENAttention` correctly replaced `torch.mean` with `F.scaled_dot_product_attention`. It now projects queries (`q_mem = self.dsa(x)`) against the stacked memory traces, allowing it to retrieve *specific* past events rather than an unreadable average.

### ✅ 3. The "Granularity" Trap

- **Critique:** Batch ingestion created 1 trace per 512 tokens, but autoregressive generation (`query.py`) created 1 trace per *token*, causing catastrophic RAM expansion and representation skew.
- **Resolution in Codebase:** **HANDLED.** `HybridTHENAttention` now implements a `state['buffer']`. It accumulates compressed tokens and only flushes them to `state['traces']` when `self.chunk_size` (default 16) is reached. This aligns the granularity of batch processing and isolated token generation.

### ❌ 4. The Context Wall (Hardware-Native Memory Manager)

- **Critique:** Sticking traces in an unbounded Python list in VRAM/RAM causes linear growth ($O(N)$) leading to inevitable OOMs and latency issues.
- **Resolution in Codebase:** **NOT HANDLED.** `state['traces']` remains a standard Python list inside VRAM. The "Stage 2 Hardware-Native" plan involving disk offloading via `mmap` or NVMe storage has not been implemented. This was acknowledged in the changelogs as a "Next Step."

---

## 2. Plan Plausibility (Plan 0: Ingest, Don't Train)

**Verdict:** The plan is conceptually sound and *highly plausible*, but mechanically broken in Phase 3 due to a logic bug in the current code preventing successful querying.

### 🚨 Critical Codebase Bug Found in `query.py`

A severe flaw exists in the interactive generation loop in Phase 3 (`scripts/query.py`):

```python
# current logic
curr_tokens = tokens 
for _ in range(args.max_new_tokens):
    logits, _ = model(curr_tokens, state=state, return_state=True)
    ...
    curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
```

**The Issue:**
`query.py` performs naive autoregressive generation by feeding the *entire growing sequence* (`curr_tokens`) to the model on every forward pass, without utilizing `kv_cache`.

Because `THENGPT.forward()` unconditionally accumulates *all* inputs into its memory `state['buffer']`, feeding `[Token A, Token B, Token C]` on Step 3 causes tokens A, B, and C to be added to memory *again*.

- Step 1 input: `[A]` -> Buffer gets 1 token.
- Step 2 input: `[A, B]` -> Buffer gets 2 *additional* tokens. (Total 3)
- Step 3 input: `[A, B, C]` -> Buffer gets 3 *additional* tokens. (Total 6)

This results in an $O(T^2)$ geometric duplication of the prompt and early generated tokens directly into the episodic memory traces, severely corrupting the context and exacerbating the Context Wall RAM issue.

---

## 3. Strategic Action Plan & Next Steps

1. **Immediate Execution (Fix Phase 3):**
   - Refactor `scripts/query.py` to implement a proper **KV Cache** loop.
   - When using a KV cache, `curr_tokens` passed to `model()` should *only* be the `next_token` (length 1), meaning the memory buffer only receives exactly 1 new token per inference step.

2. **Medium-Term Execution (Stage 2 Disk Offloading):**
   - Address the unbounded GPU RAM expansion caused by `torch.stack(state['traces'], dim=1)`. For long timelines, stacking thousands of tensors on the GPU per iteration is economically unviable. Implement L3 NVMe mapped tensors.

3. **Evaluation Metrics:**
   - Add metadata tracking to traces (timestamps, relative offsets) to diagnose exactly what the model is attending to during debug evaluations. This confirms empirical memory retrieval beyond pure perplexity metrics.
