# Changelog: Portable Memory Architecture

## Date: 2026-02-22

## Author: Antigravity (AI Assistant)

### Summary

Extracted the core elements of the Live Memory architecture—previously deeply coupled into the bespoke `THENGPT` model—into a new completely standalone, model-agnostic `portable_memory` module via PyTorch Hooks. This allows the disk-tiered streaming memory to be dropped into any open-source or open-weights HuggingFace model dynamically.

### Changes

#### 1. `portable_memory/memory_manager.py`

- Extracted `DiskTieredMemory` entirely from `nanochat-then`.
- It now functions purely via PyTorch and NumPy (using `.dat` NVMe memmap streaming) with zero dependencies on `nanochat` modules.

#### 2. `portable_memory/attention_hooks.py`

- Created dynamic injection hooks using PyTorch's `register_forward_hook`.
- Enables transparent modification of the attention mechanisms in layers of HuggingFace models (e.g., Llama, Qwen, Mistral) without altering their source code.
- Captures states for Knowledge Distillation Attention (trace compression) and Dense Sparse Attention (memory retrieval).

#### 3. `portable_memory/model_wrapper.py`

- Added the `InferenceEngine` wrapper.
- Abstracted away the complex KV Cache prefill vs. decode state separation to natively prevent the $O(T^2)$ ingestion duplication issue reliably on external models.

#### 4. `portable_memory/scripts/`

- Implemented `ingest.py` and `query.py` that transparently load `AutoModelForCausalLM` directly from the HuggingFace Hub, apply the forward hooks to the transformer layers, and process memory correctly against the `.dat` cache completely agnostic to the model's originating repository architecture.

### Impact

- **Framework Agnosticism:** Live Memory can now be trivially attached to a wide array of new baseline models (like Qwen3) reducing future technical debt and isolating core memory innovations from foundational model source code.
