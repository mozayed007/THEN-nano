# Conceptual Guide: Live Memory vs. Continued Pretraining

This document clarifies the architectural and functional differences between the **Live Memory (THEN)** workflow and standard **Continued Pretraining** or **Supervised Fine-Tuning (SFT)** used by frontier labs.

## 1. The Core Distinction: Weights vs. State

The fundamental difference lies in **where** the new information is stored and **how** it is written.

| Feature | Continued Pretraining / SFT | Live Memory (THEN) |
| :--- | :--- | :--- |
| **Storage Mechanism** | **Synaptic Weights** (Parameters) | **Episodic State** (Buffer/Cache) |
| **Write Operation** | `loss.backward()` + Optimizer Step | State writes during forward pass |
| **Model Mode** | `model.train()` | `model.eval()` (Frozen) |
| **Permanence** | Persistent until later weight updates change behavior | Persistent only if saved and managed externally |
| **Speed** | Slow (requires gradient calculation) | Potentially fast (forward-pass writes), but still requires validation in realistic settings |
| **Forgetting / Interference** | Risk of overwriting prior behavior during updates | Avoids direct weight overwriting, but retrieval quality and memory management remain open questions |

---

## 2. Standard Approach: Continued Pretraining / SFT

When frontier labs "pause and continue" training on new or higher-quality data, they are modifying the model's **weights**.

### The Process

1. **Input**: New text data (e.g., higher quality books).
2. **Compute**: Calculate loss (prediction error) and gradients.
3. **Update**: Adjust billions of floating-point numbers (weights) slightly to minimize error on the new data.

### The Problem

* **Destructive**: To learn $B$, the model often overwrites parts of $A$. This is "catastrophic forgetting."
* **Opaque**: The knowledge is diffused across the entire network. You cannot point to a specific neuron and say, "This is the memory of user ID 42."
* **Static**: Once training stops, the model is frozen. It cannot learn *during* a conversation without a retraining cycle.

---

## 3. Our Approach: Live Memory (Ingest)

The **THEN (Temporal History Episodic Network)** architecture treats memory as a distinct **state object**, separate from the processing **weights**.

### The Live Memory Process

1. **Phase 1 (Pretrain)**: Train weights *once* to learn the **mechanism** of memory (how to compress, store, and retrieve).
2. **Phase 2 (Ingest)**: Feed episodic data into the frozen model.
    * Instead of updating weights, the `HybridTHENAttention` layer compresses the input into **traces**.
    * These traces are appended to a `state` dictionary.
3. **Phase 3 (Query)**: The model attends to this `state` to answer questions.

### Code Reference (`nanochat/gpt.py`)

In `HybridTHENAttention.forward`:

```python
if layer_idx % (self.ratio + 1) < self.ratio:
    compressed = self.kda(x)
    state['buffer'] = torch.cat([state['buffer'], compressed], dim=1)
    while state['buffer'].size(1) >= self.chunk_size:
        chunk = state['buffer'][:, :self.chunk_size, :]
        trace = torch.mean(chunk, dim=1)
        state['traces'].append(trace)
        state['buffer'] = state['buffer'][:, self.chunk_size:, :]
    return compressed, state
```

### The Advantage

* **Instant**: As soon as the user speaks, it is in the state. No training run required.
* **Surgeon-Precise**: We can delete a specific memory by removing its trace from the list.
* **Weight-Preserving**: The core reasoning weights are not directly updated during ingestion, which makes external memory easier to isolate and manage.

## 4. Important Scope Note

This comparison is architectural, not a benchmark claim.

The current repository does **not** yet establish that Live Memory is empirically better than SFT or RAG on recall quality, robustness, or cost in real deployments. The current value of this comparison is to explain a different design direction: storing new information in external state instead of updating weights.

## Summary

| Metaphor | Continued Pretraining | Live Memory |
| :--- | :--- | :--- |
| **Human Analogy** | **Brain Plasticity**: Physically rewiring neurons to learn a skill (slow, hard to reverse). | **Working Memory**: Writing a note in a notebook (fast, easy to edit). |
| **Computer Analogy** | **Firmware Update**: Flashing the ROM. | **RAM/Disk Write**: Saving a file. |

**Live Memory is best understood here as a dynamic, stateful buffer that the model may learn to read and write to, whereas SFT updates the model's parameters directly.**
