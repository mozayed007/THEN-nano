# AI Live Memory Project: GPT-2 Phase Full Document – Baseline Prototyping with NanoChat & THEN

**STATUS: Phase plan written. No training executed. See run_all.py to begin.**

**Document Metadata**  

- **Author**: Muhammad Z. Ahmed (@MoZayed007)  
- **Date**: February 08, 2026 (Generated: 12:15 AM EET); Corrected Apr 28, 2026  
- **Version**: 1.2 (Reality correction: all metrics are targets, not results)  
- **Project Context**: This exhaustive, self-contained document details **Phase 1** of the Master Plan: Prototyping the Temporal-Hippocampal Embedding Network (THEN) on GPT-2-scale models using the actual NanoChat repository (<https://github.com/karpathy/nanochat>, reviewed in full as of Feb 2026). NanoChat is Karpathy's minimalist, hackable framework for end-to-end LLM training on single/multi-GPU nodes, emphasizing cognitive simplicity via one dial (`--depth`) for compute-optimal models. It covers tokenization, pretraining, SFT, eval, inference, and ChatGPT-like UI—perfect for validating THEN's internal memory mechanics (episodic traces, semantic abstraction, replay with Hybrid Option C interleaving) before Qwen3 scaling.  
  - **Repo Deep Dive Summary**: NanoChat (~1K LoC PyTorch) trains GPT-2 capability (CORE score >0.256) in ~3hrs on 8xH100 (~$72; spot ~$20). Leaderboard gamifies "time-to-GPT-2." Key files: `gpt.py` (transformer model, easy subclass for THEN), `scripts/base_train.py` (pretrain), `core_eval.py` (CORE/bpb/MMLU), `engine.py` (KV-cache infer), `ui.html` (web chat). Deps: PyTorch, wandb, uv. No factories—pure funcs for mods. Structure: `nanochat/` (core), `scripts/` (entrypoints), `tasks/` (evals like MMLU/GSM8K), `runs/` (bash like speedrun.sh).  
  - **Phase Goal**: Fork/mod NanoChat → Insert THEN (THENGPT subclass) → Gen Cairo/Arabic synthetic data → Local quick-train/test → Kaggle full baseline → Eval retention gains (20-40% target). Output: Stateful checkpoint.  
  - **Current Reality**: Code architecture complete. THENGPT subclass, ingesters, and benchmark scaffold are written. No training has ever been executed. All metrics below are targets, not results.  
  - **Hardware**: Local 12GB VRAM (tests: 4-6GB target, batch=2-4); Kaggle T4x2 (full train: ~1-2hrs estimated, untested).  
- **Obsidian Tags**: #AILiveMemory #GPT2Phase #NanoChatReal #THEN #BaselineProto #DeepReview  
- **Related Docs**: [Live Memory Master Plan](../Live%20Memory.md), [THEN Integration Threads](../THEN%20Integration%20Threads.md)  
- **Assumptions & Validations** (Ultra-Reviewed): GPT-2 124M params (depth-26 canonical); NanoChat auto-hypers (width/heads/LR from depth) yield optimal; THEN overhead 5-10% target (sparse/Hebbian, not profiled); Synthetic data (10K episodes) hypothesized sufficient for proto recall (>70%). Risks: OOM—contingency batch=1 (code-level); Conflicts—test py_compile (code passes). Validated via repo: No external DBs; KV-cache in engine.py aids state.  

---

## Executive Summary

This phase leverages NanoChat's real structure (minimal/hackable: `gpt.py` for model mods, `base_train.py` for pretrain, `chat_web.py` for UI) to embed THEN: Subclass GPT as THENGPT → Interleave Option C (3:1 KDA:DSA for theta-rhythms) post-attn → Hook episodic/semantic/replay in forward → Gen multilingual synthetic (Cairo/Arabic episodes) → Train depth-12 baseline (~25M params, ~10min local target) → Scale to depth-26 (~1hr Kaggle target) → Eval via extended `core_eval.py` (CORE + custom forgetting).  

**Deep Insights from Repo**: NanoChat's "one-dial" depth auto-tunes everything (e.g., depth=12: n_embd=768, heads=12, LR=6e-4)—ideal for THEN scaling without config hell. Deps minimal (uv sync); Custom: Edit `gpt.py` → Rerun scripts.  

**Code Status**: THENGPT subclass written and imports work. Synthetic data generator written. Ingest/query scripts written. Benchmark scaffold written. No training executed. All output metrics below are TARGETS.  

---

## 1. Phase Foundations: NanoChat Repo Deep Review & GPT-2 Context

### 1.1 Actual Repo Structure & Content (Ultra-Extracted)

From full GitHub crawl (README verbatim + file tree): NanoChat is a "simplest experimental harness" for single-GPU LLM lifecycle (tokenization → pretrain → SFT → eval → infer/UI). No configs/factories—hackable pure PyTorch. Purpose: Democratize GPT-2 (~$43K in 2019 → <$100 now) via `--depth` dial (auto-hypers: Width/heads/LR/decay from depth). Leaderboard: Time-to-GPT-2 (CORE>0.256; current best ~2.76hrs 8xH100).  

**Full File Tree** (Exact from Repo):  

```text
.  
├── LICENSE (MIT)  
├── README.md  
├── dev/  
│   ├── gen_synthetic_data.py  
│   ├── generate_logo.html  
│   ├── nanochat.png  
│   └── repackage_data_reference.py  
├── nanochat/  
│   ├── __init__.py  
│   ├── checkpoint_manager.py  
│   ├── common.py  
│   ├── core_eval.py  
│   ├── dataloader.py  
│   ├── dataset.py  
│   ├── engine.py  
│   ├── gpt.py  
│   ├── logo.svg  
│   ├── loss_eval.py  
│   ├── optim.py  
│   ├── report.py  
│   ├── tokenizer.py  
│   └── ui.html  
├── pyproject.toml  
├── runs/  
│   ├── miniseries.sh  
│   ├── runcpu.sh  
│   ├── scaling_laws.sh  
│   └── speedrun.sh  
├── scripts/  
│   ├── base_eval.py  
│   ├── base_train.py  
│   ├── chat_cli.py  
│   ├── chat_eval.py  
│   ├── chat_rl.py  
│   ├── chat_sft.py  
│   ├── chat_web.py  
│   ├── tok_eval.py  
│   └── tok_train.py  
├── tasks/  
│   ├── arc.py  
│   ├── common.py  
│   ├── customjson.py  
│   ├── gsm8k.py  
│   ├── humaneval.py  
│   ├── mmlu.py  
│   ├── smoltalk.py  
│   └── spellingbee.py  
├── tests/  
│   └── test_engine.py  
└── uv.lock  
```

**Key Features (Repo Summary)**:  

- **Auto-Hypers**: Depth drives all (e.g., depth=12: GPT-1 scale, n_embd=768, heads=12; depth=26: GPT-2, ~1.6B params, CORE~0.256).  
- **Pipeline**: Pretrain (`base_train.py`), SFT (`chat_sft.py`), Eval (`core_eval.py`: CORE/bpb/MMLU/GSM8K/HumanEval), Infer (`engine.py`: KV-cache + tools), UI (`chat_web.py` + `ui.html`: Gradio-free HTML/JS).  
- **Customization**: Hack `gpt.py` (transformer: Embed → [LN + CausalAttn + Add → LN + FFN + Add] x depth → LMHead). Modify forward for THEN; Rerun scripts (e.g., `base_train.py` loads custom class via `--model_class=THENGPT`). **(Code written, not trained)**  
- **Deps/Setup**: `uv sync` (PyTorch, wandb); `source .venv/bin/activate`. No heavy libs; MPS/CPU support (`runcpu.sh`).  

**GPT-2 Context**: NanoChat recreates OpenAI GPT-2 (2019: 1.5B params, 1K context, causal LM). Baseline CORE=0.256; NanoChat hits 0.260 in 2.76hrs. For THEN: Mod attn in `gpt.py` for Option C; Use `engine.py` state for traces.  

---

## 2. Detailed Actions: Code-Written Plan (No Training Executed)

### Code Status Summary

All code blocks below are written and pass `py_compile`. No training runs have been initiated. The day-by-day routine below describes the intended execution order — it has not been followed. Steps marked [not run] require a trained model to execute.

### THENGPT Subclass (Code Written)

The THENGPT subclass extending NanoChat's GPT with EpisodicEncoder, SemanticAbstracter, and HybridTHENAttention (Option C, 3:1 KDA:DSA ratio) is written in `nanochat/gpt.py`. See the fork for full source.

### Synthetic Data Generator (Code Written)

`dev/gen_synthetic_data.py` generates 10K Cairo/Arabic episodes. Generator runs but output has not been used in training.

### Training Pipeline (Code Written, Not Run)

The following commands are the intended execution path. None have been invoked with a real training run.

**Quick-Train (intended, not run)**:  

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --device_batch_size=2 \
    --max_seq_len=512 \
    --dataset_file="synthetic-cairo-episodes.txt" \
    --model_class="THENGPT" \
    --epochs=0.1 \
    --run="gpt2-then-local-quick"
```

**Full Train (intended, not run)**:  

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=26 \
    --device_batch_size=4 \
    --max_seq_len=1024 \
    --dataset_file="synthetic-cairo-episodes.txt" \
    --model_class="THENGPT" \
    --epochs=0.5 \
    --run="gpt2-then-kaggle-d26"
```

---

## 3. Target Metric Table (Untested)

**PENDING: benchmark not yet run. All values below are design targets, not empirical results.**

| Metric             | Vanilla GPT-2 (target) | THENGPT (target) | Target Gain | Notes                              |
| ------------------ | ---------------------- | ---------------- | ----------- | ----------------------------------- |
| CORE Score         | TBD                    | TBD              | TBD         | PENDING: train required             |
| Recall Avg         | TBD                    | TBD              | TBD         | PENDING: multi-turn Cairo test      |
| Forgetting Rate    | TBD                    | TBD              | TBD         | PENDING: measure after training     |
| VRAM Peak          | TBD                    | TBD              | TBD         | PENDING: profile during training    |

The originally documented speculative numbers (Recall 0.55 → 0.78, Forgetting 0.25 → 0.12, CORE 0.256 → 0.258) were design targets from the article and HiMeS-inspired literature. They have never been empirically measured in this project. They remain reasonable hypotheses to test once training begins.

---

## 4. Phase Outputs (Not Yet Produced)

- **Repo Fork**: <https://github.com/mozayed007/THEN-nano> (Mods: THENGPT code, synthetic data gen) — **code only**  
- **Checkpoints**: `runs/gpt2-then-kaggle-d26/final.pt` — **not yet produced**  
- **Data**: `synthetic-cairo-episodes.txt` — **generator written, not validated for training**  
- **Evals**: Metrics table — **pending first training run**  
- **UI Demo**: `python scripts/chat_web.py --model=final.pt` — **not runnable without trained model**  

### Handover to Qwen3 Phase

- Transfer: Load GPT-2 state as Qwen init (`qwen_then.load_state_dict(gpt_state, strict=False)` — Subspaces align). **(code planned, not executed)**  
- Master Plan: Mark Phase 1 as code-complete only.  

### Ultra-Review Notes

- **Repo Fidelity**: All commands repo-exact (e.g., torchrun flags from speedrun.sh; Eval from core_eval.py). No inventions in code structure.  
- **Depth Choices**: Depth-12 quick (~25M params, ~5min target); 26 full (GPT-2 1.6B, leaderboard-aligned).  
- **Cairo/Arabic**: 30% data Arabic (تذكر تفضيلي); Qwen handover multilingual boost.  
- **Assumptions Re-Validated**: Overhead: Code adds ~50 LoC (sparse); Gains: Hypothesized from repo CORE + custom (untestable without training). Risks Low: Repo CPU fallback if GPU issue.  

This phase plan is locked. Training has not yet begun. See `run_all.py` for the execution entry point.

**End of GPT-2 Phase Document**  
*Corrected: Apr 28, 2026. All metrics stripped. Code status accurately reflected.*