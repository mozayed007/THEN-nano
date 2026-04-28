# Project Critique and Roadmap: Live Memory (THEN)

## Executive Summary

The **Live Memory (THEN)** project implements a novel "Ingest, Don't Train" architecture aimed at reducing reliance on retraining for episodic updates. While the **Phase 1 (Pretrain) → Phase 2 (Ingest) → Phase 3 (Query)** workflow is conceptually sound, the current implementation still faces significant scalability, training-alignment, and validation hurdles.

## Critical Analysis

### 1. The Pretraining Gap (Major Risk)

**Issue:** The current pretraining pipeline (`base_train.py`) trains the model as a standard stateless Transformer. It resets the memory state at every batch.
**Implication:** The `HybridTHENAttention` layer is never forced to learn how to *read* from its memory traces during training, because the state is always empty or irrelevant.
**Result:** When we switch to Phase 2/3, the model may fail to attend to the injected traces because it never learned that mechanism. It's like giving a notebook to someone who never learned to read handwriting.

### 2. The Context Wall (Scalability Risk)

**Issue:** `HybridTHENAttention` appends traces to a list: `state['traces'].append(trace)`.
**Implication:** Memory grows linearly with experience ($O(N)$).
**Result:**

* **RAM Explosion:** After days of "life," the state will exceed GPU memory.
* **Retrieval Latency:** Attending to all history becomes slow ($O(N)$).
* **Noise:** As history grows, the signal-to-noise ratio in the retrieval mechanism drops, leading to hallucinations.

### 3. Data Realism

**Issue:** We rely on `synthetic-cairo-episodes.txt`.
**Implication:** The data is clean, structured, and causal. Real user logs are messy, interleaved, and often irrelevant.
**Result:** The model may overfit to the simple narrative structure of the synthetic data and fail on real, noisy user interactions.

---

## Roadmap & Next Stages

> **Update 2026-04-28:** All stages refer to code implementation only. No model has been trained. The training pipeline has never been executed. THENGPT weights are random. All "fixes" are code-level only, not empirically proven.

### Stage 1: Stateful Pretraining (CODE WRITTEN, NOT TRAINED)

We must modify the pretraining loop to simulate the "Live Memory" condition.

* **Action:** Implement **Truncated Backpropagation Through Time (TBPTT)** or simply pass the `state` dictionary from Batch $N$ to Batch $N+1$ when training on long documents.
* **Objective:** Force the model to recall information from the *previous* batch (which is now in `state`) to predict tokens in the *current* batch.
* **Status:** Code exists in `scripts/base_train.py` (2026-02-13). State persistence and gradient detachment are implemented in code. **Zero training steps have been run. The model weights are random.**

### Stage 2: Scalable Memory Infrastructure (Hardware-Native) — NOT STARTED

We will reject external Vector DBs and plugins to maintain a pure, zero-dependency architecture. We will manage the memory hierarchy directly, treating the OS and hardware as the database.

* **Action:** Implement a **Native Hierarchical Memory Manager**.
  * **L1 (GPU VRAM):** Hot traces (Ring Buffer, last ~N steps). Fast access for immediate attention.
  * **L2 (Host RAM):** Warm traces. Paged out from GPU when full.
  * **L3 (Disk/NVMe):** Cold traces. Stored as raw binary files or memory-mapped tensors (`torch.mmap`).
* **Mechanism:**
  * **Direct Addressing:** The model generates "memory pointers" (time-indices) to retrieve specific blocks from disk without loading the whole file.
  * **OS Paging:** Rely on the operating system's virtual memory management (via `mmap`) to handle caching of large trace files transparently.
  * **Zero-Copy:** Use zero-copy tensor views to read from disk directly into tensor processing.

### Stage 3: Mechanism-Specific Tasks — NOT STARTED

Train on tasks that explicitly require memory, not just generic text.

* **Task:** "Passkey Retrieval" (Hide a key at $t=0$, ask for it at $t=10000$).
* **Task:** "Personality Consistency" (Reward the model for maintaining consistent preferences over long contexts).

### Stage 4: Real-World Deployment — NOT STARTED

* **Privacy:** Since memory is a file (`state.pt`), we can offer "Incognito Mode" (don't save traces) or "Memory Wipe" (delete file) features trivially.
* **Personalization:** If the mechanism is validated and the storage path is reliable, one frozen model could potentially serve many users with separate state files.

## Conclusion

The project has strong potential as an alternative architecture for stateful personalization, but it currently exists as a **mechanical prototype**. It validates the *flow* (Ingest/Query) more than the *learning* (Can it actually use the memory?). The next step is to bridge the gap between generic pretraining and stateful inference with controlled benchmarks and ablations.
