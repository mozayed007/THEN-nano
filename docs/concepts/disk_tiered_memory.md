# Concept: Disk-Tiered Memory Management

## Overview

Live Memory (THEN phase) shifts away from traditional Vector Databases (like Pinecone or Chroma) to maintain a pure, zero-dependency, local-first architecture. Instead of holding unbounded lists of episodic traces in VRAM—which inevitably leads to catastrophic Out-Of-Memory (OOM) errors during long LLM sessions—the project implements a **Hardware-Native Memory Manager** (`nanochat.memory_manager.DiskTieredMemory`).

## The Concept

The core philosophy is to leverage the Operating System's highly optimized virtual memory and paging subsystems rather than reinventing a sluggish user-space database proxy.

### 1. Ingestion via Memory Mapping (`mmap`)

When a trace is compressed by the Knowledge Distillation Attention (KDA) layer, instead of pushing it to a Python list, `DiskTieredMemory` writes the raw `(B, D)` tensor directly into a `numpy.memmap` array backed by an NVMe file.

* **Why?** Writing sequentially to an unbuffered disk file is incredibly fast. Memmap avoids application-level serialization overhead (like `torch.save` dicts), keeping the Python process memory perfectly flat regardless of how many gigabytes of traces are ingested.

### 2. Streaming Attention Retrieval

During the Dense Sparse Attention (DSA) retrieval phase, standard self-attention mechanisms require projecting the `Query` against all `Keys`. If all keys are loaded in VRAM simultaneously, the context window crashes.

**DiskTieredMemory approaches this as a data-streaming problem:**

1. **Chunking:** The manager reads contiguous blocks (e.g., 4096 traces) from the disk array. Because of `mmap`, the OS intelligently predicts and prefetches pages from the NVMe, providing near-RAM latency for sequential accesses.
2. **Local Scoring:** A local dot-product similarity (`Query @ Chunk.T / sqrt(D)`) is computed on the GPU for the loaded chunk.
3. **Global Accumulation:** A running `Top-K` buffer is maintained. After each chunk, the local highest-scoring indices are merged mathematically into the global best pool.
4. **Selective Fetching:** Once the entire disk file has been scanned in chunks, the module possesses the absolute global `Top-K` indices. It reads *only those specific rows* one last time to form the `Values` tensor and computes the final weighted context sum back to the transformer stream.

## Why Not Just Use KV Cache Offloading?

Standard KV cache offloading focuses on retaining raw key and value heads for strict sequence generation. `DiskTieredMemory` focuses on **high-level, semantically compressed episodic traces**.

* Standard KV caching (1 token = ~1.5 KB per layer) explodes to terabytes over months of user interaction.
* THEN's episodic `DiskTieredMemory` compresses concepts over time (chunked mean pooling) and acts specifically as the mechanism for the model to remember overarching context and historical facts decoupled from raw token limits.

## The Interactive Query Bug (O(T^2) Trace Corruption)

Implementing Live Memory introduces uniquely dangerous edge cases compared to standard stateless LLMs.

Because `THENGPT` actively writes to memory during inference (the "Ingest, Don't Train" philosophy), standard naive autoregressive generation (e.g., passing `[Prompt, T1, T2]` -> `[Prompt, T1, T2, T3]` on each step) causes geometric duplication. The model re-ingests the `Prompt` on Step 1, Step 2, and Step 3 repeatedly.

**The Solution:** Strict separation of generation logic.

1. **Prefill:** The model must process the initial user context comprehensively to build KV caches and properly log the prompt event to memory exactly *once*.
2. **Decode:** The generation loop must only pass the absolute `next_token` along with the KV state. Providing a 1-length tensor guarantees the memory buffer receives only 1 new event per interaction, saving storage and guaranteeing coherence. This is rigorously handled in `scripts/query.py`.

## Portable Architecture & Model Agnosticism

While originally integrated exclusively into the bespoke `THENGPT`, the architecture has now been extracted into the `portable_memory` module.

* `DiskTieredMemory` operates purely on PyTorch and NumPy, completely decoupled from the parent LLM.
* `attention_hooks.py` uses `register_forward_hook` to dynamically inject Knowledge Distillation Attention (writing) and Dense Sparse Attention (reading) into standard HuggingFace models (like Llama, Qwen, Mistral) without modifying their source code.
* `model_wrapper.py` (via `InferenceEngine`) securely handles the tricky Prefill vs. Decode state splitting across external open architectures.
