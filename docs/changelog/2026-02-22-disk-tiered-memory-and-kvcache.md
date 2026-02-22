# Changelog: Data-Intensive Memory Architecture & Query Fixes

## Date: 2026-02-22

## Author: Antigravity (AI Assistant)

### Summary

Addressed the severe OOM vulnerability and generation bugs identified in the "Loop 3 Critique". Implemented the Stage 2 Hardware-Native Memory Manager (`DiskTieredMemory`) to scale episodic memory to NVMe speeds. Fixed the catastrophic $O(T^2)$ memory trace duplication bug in the `query.py` interactive loop by properly utilizing key-value (KV) caching.

### Changes

#### 1. `nanochat/memory_manager.py` (NEW)

* **`DiskTieredMemory`**: A data-intensive, hardware-native memory manager.
  * Replaces unbounded Python lists in VRAM.
  * Streams episodic traces directly to an unbuffered `numpy.memmap` (`.dat` file) on disk.
  * Implements `retrieve(query)`, which streams memory chunks back to VRAM, computes local similarities via dot products, and dynamically accumulates a global Top-K retrieval across the entire disk history. This achieves O(1) VRAM footprint regardless of timeline length.

#### 2. `nanochat/gpt.py`

* **`HybridTHENAttention`**:
  * Integrated `memory_manager` hooks into the forward pass.
  * The KDA compression branch now calls `state['memory_manager'].append(trace)` if configured.
  * The DSA retrieval branch now calls `state['memory_manager'].retrieve(q_mem)` to stream attention from disk, resolving the RAM Context Wall. Backward compatibility with legacy `traces` logic is maintained as a fallback.

#### 3. `scripts/ingest.py`

* **Refactored Ingestion**:
  * Upgraded to instantiate and manage a `DiskTieredMemory` instance.
  * Phase 2 now safely serializes all episodic timelines directly to disk (`cairo_memory_state.dat`), detaching the context size limits from GPU memory capabilities.

#### 4. `scripts/query.py`

* **Fixed Phase 3 Interactive Loop**:
  * **The Bug**: Previously, the loop fed the entire growing sequence `curr_tokens` during autoregressive generation to the `THENGPT` model. Since the model caches memories unconditionally during forward passes, tokens were exponentially duplicated into the memory buffer step-by-step ($O(T^2)$ corruption).
  * **The Fix**: Introduced a strict **Prefill** vs **Decode** state machine.
    * **Prefill**: Processes the initial prompt using `nanochat.engine.KVCache`.
    * **Decode**: Feeds only a single, length-1 `next_token` to the model per step while passing the `kv_cache`. The newly generated token is safely and linearly recorded precisely once per step.

### Impact

* **RAM Stability**: Context sizes are no longer bound by VRAM. Billions of tokens of "lived experience" can now be reliably hosted on standard NVMe drives.
* **Accuracy**: The model's episodic memory is uncorrupted from duplicate insertions, ensuring that temporal queries resolve cleanly without self-hallucinated noise.
