# LiveMem — Learned Episodic Memory for Frozen Language Models

> **Status:** Architecture prototype. Code complete. No training has been run. All claimed mitigations are code-level, not empirically validated.

**LiveMem** explores a hypothesis: that a language model's weights and its memory should be separate systems. Instead of retraining a model to learn new things, we give it a notebook.

Built on top of [karpathy/nanochat](https://github.com/karpathy/nanochat) — the simplest experimental harness for training LLMs — this project extends it with the **THEN (Temporal History Episodic Network)** architecture: a learned external memory layer that lets a frozen model write, store, and retrieve episodic state without touching its weights.

**Author:** Muhammad Z. Ahmed ([@MoZayed007](https://github.com/MoZayed007))

---

## The Hypothesis

Standard LLMs have three options for incorporating new information:

1. **Continued pretraining / SFT** — slow, expensive, prone to catastrophic forgetting, hard to target per-user.
2. **RAG** — practical but keeps memory outside the model's learned dynamics, creates latency and prompt pressure.
3. **Tool-based memory** — manual, prompting-dependent, not a natural part of the model's forward pass.

LiveMem proposes a fourth path:

1. **Train once** — the model learns the *mechanism* of memory (how to compress, store, and retrieve traces).
2. **Ingest into frozen state** — new information is written to external state via forward passes, not gradient updates.
3. **Query with state** — the model reads its own memory during inference.

If this works, user-specific memory becomes a **state management problem**, not a model update problem.

## Architecture: THEN

The THEN architecture adds lightweight modules to transformer blocks that simulate hippocampal memory dynamics:

```
Input → Encoder → Replay → Abstracter → Integrator → Neurogenesis → Output
```

- **Encoder** — compresses input into memory traces
- **Replay** — reactivates relevant traces during forward passes
- **Abstracter** — consolidates raw traces into higher-level representations
- **Integrator** — merges retrieved memory with current reasoning
- **Neurogenesis** — manages trace lifecycle and capacity

Two retrieval paths:
- **KDA (Key-Driven Attention)** — semantic lookup by content
- **DSA (Dynamic State Attention)** — temporal/contextual retrieval

The system includes a `portable_memory` module using PyTorch `forward_hooks` for drop-in compatibility with standard HuggingFace models.

## Ingest, Don't Train Workflow

```bash
# Phase 1: Pretrain (learn the mechanism, not the memories)
python -m scripts.base_train --model-class THENGPT --depth 8 ...

# Phase 2: Ingest (populate memory state, weights stay frozen)
python -m scripts.ingest --model_path outputs/d8/model_000100.pt --data_path data/episodes.txt

# Phase 3: Query (recall from populated state)
python -m scripts.query --model_path outputs/d8/model_000100.pt --state_path memory_state.pt
```

## What This Is Not

LiveMem is not a claim that LLMs have human-like memory. The architecture is best understood as:

- A learned episodic state layer
- A dynamic notebook for a frozen model
- A per-user external memory substrate

It is a high-potential prototype, not a validated paradigm.

## Validation Criteria

The concept should be considered validated only when:

1. The model answers correctly *because of* the external state
2. That behavior disappears when state is removed or corrupted
3. The effect holds across increasing temporal gaps and distractors
4. The mechanism works comparably to simpler baselines
5. Storage/retrieval remains stable from RAM to disk-tiered state

## Getting Started

### Setup

```bash
cd nanochat-then
uv sync --extra gpu    # or: uv sync --extra cpu
source .venv/bin/activate
```

### Run Tests

```bash
python run_all.py
# or individually:
python -m tests.test_live_memory
```

*High recall accuracy requires a fully pretrained THENGPT model. Tests currently verify mechanical correctness of the state pipeline.*

### Reproduce the Base Model (nanochat speedrun)

The upstream nanochat provides a full pipeline to train a GPT-2 grade model on an 8XH100 node in ~2 hours (~$48). See [runs/speedrun.sh](runs/speedrun.sh):

```bash
bash runs/speedrun.sh
python -m scripts.chat_web
```

See [nanochat's README](https://github.com/karpathy/nanochat) for full documentation on the base training harness, leaderboard, and research workflow.

## Key Documentation

| Document | Description |
|----------|-------------|
| [THEN Architecture](docs/concepts/then_architecture_public.md) | The "Notebook" analogy and how THEN works |
| [Live Memory Thesis](docs/concepts/live_memory_refined_thesis.md) | Refined conceptual foundation and validation criteria |
| [THEN vs SFT vs RAG](docs/concepts/live_memory_vs_sft.md) | Comparison with existing approaches |
| [Critique Loop 5](docs/critique/critique_loop_5.md) | Honest analysis of current limitations |
| [Master Plan](docs/Live%20Memory.md) | Full project blueprint from pretraining to deployment |
| [Causal Memory Validation](docs/concepts/causal_memory_validation.md) | Ablation and validation methodology |

## Project Structure

```text
.
├── nanochat/                    # Core engine (forked from karpathy/nanochat)
│   ├── gpt.py                   # GPT transformer with THENGPT subclass
│   ├── memory_manager.py        # Disk-tiered memory management
│   ├── engine.py                # Inference with KV cache
│   ├── optimizer.py             # AdamW + Muon optimizer
│   └── ...
├── portable_memory/             # Model-agnostic memory via PyTorch hooks
│   ├── memory_manager.py        # DiskTieredMemory
│   └── attention_hooks.py       # KDA/DSA hook implementations
├── scripts/
│   ├── base_train.py            # Train the base model
│   ├── ingest.py                # Populate memory state from data
│   ├── query.py                 # Query with populated memory
│   ├── tiny_recall_benchmark.py # Memory recall validation scaffold
│   └── ...
├── tests/
│   ├── test_live_memory.py      # Memory pipeline tests
│   └── test_engine.py           # Engine tests
├── docs/
│   ├── concepts/                # Architecture docs and thesis
│   ├── critique/                # Honest limitation analysis
│   └── plans/                   # Validation roadmap
└── dev/
```

## Critical Analysis & Roadmap

See [docs/critique/critique_loop_5.md](docs/critique/critique_loop_5.md) for a detailed critique.

Key findings:
- **Cost**: Moving memory to Disk/NVMe is essential — from $0.37/user/hr to $0.0001/user/hr.
- **Architecture**: "Mean Retrieval" must be replaced with "Attention Retrieval" to avoid memory blurring.
- **Validation**: The model has not yet been shown to use memory causally. The central claim remains unproven.

## Upstream: nanochat

This project is built on **[karpathy/nanochat](https://github.com/karpathy/nanochat)** by Andrej Karpathy. Nanochat is the simplest experimental harness for training LLMs — designed to run on a single GPU node with minimal, hackable code covering tokenization, pretraining, finetuning, evaluation, inference, and a chat UI.

The upstream nanochat includes:
- A **Time-to-GPT-2 leaderboard** for pretraining speedruns
- A single `--depth` dial that auto-configures all hyperparameters for compute-optimal models
- Full pipeline from raw data to a ChatGPT-like web UI

For questions about the base nanochat codebase, see [DeepWiki](https://deepwiki.com/karpathy/nanochat) or the [Discussions tab](https://github.com/karpathy/nanochat/discussions).

## Acknowledgements

- **[Andrej Karpathy](https://github.com/karpathy)** — nanochat and nanoGPT, the foundation this project builds on
- **[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)** — leaderboard and gamification ideas
- **[HuggingFace](https://huggingface.co/)** — FineWeb and SmolTalk datasets
- **[Lambda](https://lambda.ai/service/gpu-cloud)** — compute infrastructure

## License

MIT
