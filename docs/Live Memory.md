# AI Live Memory Project: Comprehensive Master Plan Document – From Pretraining to Qwen3 Interface

**STATUS: Architecture prototype. Code complete. No model training has been run. All claimed mitigations are code-level, not empirically validated.**

**Document Metadata**  

- **Author**: Muhammad Z. Ahmed (@MoZayed007)  
- **Date**: February 08, 2026 (Generated: 12:00 AM EET)  
- **Version**: 1.0 (Final Synthesis)  
- **Project Overview**: This all-in-one master plan consolidates our entire discussion thread into a actionable blueprint for the AI Live Memory system. It evolves from neuroscience-inspired memory types (episodic, semantic, procedural, working) and the revolutionary **THEN (Temporal-Hippocampal Embedding Network)** architecture—simulating hippocampal/MTL dynamics internally to cure LLM "anterograde amnesia." We start with GPT-2-scale prototyping (NanoChat for basics), transition to Qwen3-8B (efficient, multilingual scaling), and culminate in a NanoChat-like interface for interactive testing.  
  - **Core Goal**: Build a model-agnostic, stateful learner with real-time consolidation/replay, no external DBs.  
  - **Phases Covered**: Pretraining → Customization → Training → Evaluation → Interface Deployment.  
  - **Personalization**: Tailored for your Cairo setup (12GB VRAM / 30GB RAM rig). ADHD-friendly: Modular steps, checklists.  
- **Obsidian Tags**: #AILiveMemory #MasterPlan #THEN #Qwen3 #NanoChat #FullPipeline  
- **Related Docs**: [causal_memory_validation.md](concepts/causal_memory_validation.md), [live_memory_refined_thesis.md](concepts/live_memory_refined_thesis.md), [early_validation_and_scaling_plan.md](plan/early_validation_and_scaling_plan.md)  
- **Assumptions & Validations**: See Section 2. This doc is your "routine bible"—print/PDF for desk reference; update via Obsidian links.  

---

## Executive Summary

This plan operationalizes our thread: From amnesia analogy → THEN design (with Hybrid Option C interleaving) → NanoChat prototyping → Model pivot to Qwen3-8B → Full pipeline. Budget: $0 (Kaggle free tier; local rig).  

**High-Level Phases**:  

1. **Pretraining Baseline** (GPT-2/NanoChat: Validate THEN basics).  
2. **Customization** (Insert THEN into Qwen3).  
3. **Training** (Local quick-tests → Kaggle full runs).  
4. **Evaluation** (Memory-specific metrics).  
5. **Interface** (NanoChat-like UI for Qwen3 demos).  

**Expected Outcomes**: A persistent Qwen3-THEN model with 25-40% better multi-turn retention; Arabic/Cairo-personalized episodes; open-source ready. — **All pending training validation.**

**Current Reality**: Code architecture is written and imports work. All THEN modules, HybridTHENAttention, ingesters, and benchmark scaffold exist. No training has ever been executed. THENGPT weights are random. Memory mechanism has zero empirical validation.

---

## 1. Project Foundations: Recap of Discussion Synthesis

### 1.1 Core Concepts from Thread

- **Amnesia Problem**: LLMs/VLMs (e.g., GPT-2, Qwen3) are "smart but forgetful"—static weights + limited context mimic anterograde amnesia. Solution: Internal memory via THEN.  
- **Memory Types** (Neuroscience-Inspired):  
  - Episodic: Timestamped events (hippocampus binding).  
  - Semantic: Fact graphs (MTL abstraction).  
  - Procedural: Habit rules (Hebbian updates).  
  - Working: Dynamic buffers (prefrontal fusion).  
- **THEN Architecture**: Lightweight modules (5-10% overhead) embedded in transformer blocks: Encoder → Replay → Abstracter → Integrator → Neurogenesis. Stateful forward passes form/consolidate traces.  
- **Hybrid Attention (Option C)**: Layer-wise interleaving (3:1 KDA:DSA) for theta-rhythmic encoding/retrieval—simulates hippocampal cycles; targets O(N) efficiency.  
- **Model Path**: GPT-2 (NanoChat proto) → Qwen3-8B (scale: 8B params, 128K context, multilingual).  
- **Tools/Workflow**: Local mods (PyTorch) → GitHub fork → Kaggle train (T4x2, 30hr quota) → Interface (Gradio/NanoChat UI).  
- **Hardware**: Your 12GB VRAM rig (local tests: 4-6GB); Kaggle for trains (32GB total).  

### 1.2 Key Assumptions & Validations

Deep-reviewed thread; corrected where needed (e.g., param sizes). — **All validation statuses are code-level assumptions. Nothing has been empirically verified.**

| Assumption | Validation | Status |
|------------|------------|--------|
| **Model Sizes**: Qwen3-8B ~8B params (not 7B mix-up). | Confirmed via HF docs/arXiv (Feb 2026 release). | Valid (docs) |
| **VRAM Fits**: 4-5GB quantized on your rig; 20GB max for Kaggle. | nvidia-smi estimates in thread sketches. | Valid (calculations) |
| **THEN Overhead**: 5-10% params/compute. | Code sketches: Sparse ops; Hebbian gradient-free. | Code-level, not profiled |
| **Efficiency**: O(N) via Option C; 25-40% retention gain. | Derived from article + HiMeS-inspired evals. | Not validated — requires training |
| **Quotas**: Kaggle 30hr/week GPU free. | Current policy (2026 stable). | Valid (platform) |
| **Multilingual**: Qwen3 strong in Arabic for Cairo episodes. | HF benchmarks: 85% Arabic MMLU. | Valid (reported) |
| **No External Deps**: THEN internal (no DBs). | Core design; state in model.state_dict(). | Code-level |

**Risks**: OOM on Kaggle—mitigate with batch=2 (code-level). Interference—orthogonal subspaces in code (code-level, not verified).

---

## 2. Detailed Action Plan: Phases from Pretraining to Interface

### Phase 1: Pretraining Baseline (GPT-2 via NanoChat) – Validate THEN Basics

**Status**: Code complete. No training run.  
**Goal**: Quick prototype to test THEN inserts/Option C on simple causal decoder. ~4-6hrs total (untested estimate).  
**Why First?**: NanoChat's minimalism (1K LoC) confirms mechanics before Qwen3 scale.  

**Actions**:

1. **Setup & Fork** (code complete).  
   - Clone: `cd ~/projects; git clone https://github.com/karpathy/nanochat.git nanochat-gpt2`.  
   - Fork on GitHub (<https://github.com/mozayed007/THEN-nano>).  
   - Mod `gpt.py`: Insert THENGPT subclass. **(code written, not trained)**  
   - Gen synthetic data: Run `python gen_synthetic_data.py` with 10K episodes. **(generator written, data not validated for training)**  

2. **Local Quick-Train** [not run].  
   - Command: `torchrun --standalone --nproc_per_node=1 scripts/base_train ...`  
   - Test chat: `python scripts/chat_cli.py ...`  
   - **Check**: VRAM <6GB; Recall acc >70% (target, untested).  

3. **Kaggle Baseline Train** [not run].  
   - Upload fork/dataset to Kaggle.  
   - Full run: `--depth=12 --epochs=1`.  
   - Eval: `python core_eval.py` + custom forgetting metrics.  

**Milestone**: Baseline checkpoint — **not yet produced**. Assumptions: NanoChat runs unmodified except subclass (untested).

### Phase 2: Customization for Qwen3 – Scale THEN

**Status**: Code complete. No training run.  

**Actions**:

1. **Load & Subclass** (code written).  
   - Install: `pip install transformers bitsandbytes accelerate`.  
   - Code: QwenTHEN subclass from prior doc. **(code written, not trained)**  
   - Add Arabic synthetic data gen.  
   - **Check**: `print(model)` confirms modules. VRAM ~5GB (estimate, not measured).  

2. **Integration Test** [not run].  
   - Forward test: Multi-turn Arabic/English.  
   - Hook replay: Trigger on salience >1.0; log state['traces'].  
   - **Check**: Recall test: 85% fidelity target (untested).  

**Milestone**: Qwen3-THEN ready in code — **not yet validated**.

### Phase 3: Training Pipeline – Local to Kaggle

**Status**: Code complete. No training run.  

**Actions**:

1. **Prep Data/LoRA** (code written).  
   - Dataset: 100M tokens target (synthetic + OpenWebText Arabic subset).  
   - LoRA config: `peft` targeting attn/THEN modules (r=16). **(config written, not applied)**  

2. **Quick Local Fine-Tune** [not run].  
   - `transformers.Trainer` with LoRA: 1 epoch subsample.  

3. **Full Kaggle Train** [not run].  
   - Notebook: Clone fork; Train: `trainer.train()` with batch=4, epochs=3.  

**Milestone**: Trained model zip — **not yet produced**.

### Phase 4: Evaluation & Refinements – Metrics & Ablations

**Status**: Code complete. No benchmarks run.  

**Actions**:

1. **Run Benchmarks** [not run].  
   - Datasets: MMLU (HF) + Custom multi-session (100 turns).  
   - Metrics: Retrieval acc (Precision@5), Forgetting rate (<10%), Consolidation eff (<1s/update).  
   - Code: Extend NanoChat `core_eval.py` for Qwen. **(code written, never executed)**  
   - Ablate Option C: Train vanilla vs. interleaved; Compare gains. **(ablations not run)**  

2. **Debug/Refine** [not applicable — no results to debug].  
   - Profile: `torch.profiler` for VRAM/latency.  
   - Neurogenesis test: Simulate load>50 traces → Expand params.  

**Milestone**: Eval report — **not yet produced**. Assumptions: 20-50% gains are targets only, not confirmed.

### Phase 5: Interface Deployment – NanoChat-Style UI for Qwen3

**Status**: Code complete. No integration test with trained model.  

**Actions**:

1. **Build UI** (code written).  
   - Gradio interface with stateful THEN chat. **(code written, not tested with trained model)**  
   - Trace viewer sidebar.  
   - **Check**: Multi-turn with trained model TBD.  

2. **Polish & Deploy** [not applicable].  
   - Host: Gradio public (free) or local Streamlit.  

**Milestone**: Live demo link — **not yet produced**.

---

## 3. Risks, Contingencies & Final Assumptions

**Risks**:  

- OOM: Drop batch/seq_len; Use CPU offload. **(code mitigations written, not tested)**  
- Quota Exhaust: Fallback to local subsample (slower).  
- Arabic Fidelity: Fine-tune extra on Cairo data if <80%.  
- **Pretraining Gap**: Without stateful pretraining, the model might ignore memory. **Mitigation**: Implemented Truncated BPTT in `base_train.py` (2026-02-13) — **code-level only, not validated through training**.  
- **Memory Blurring**: Averaging traces destroys detail. **Mitigation**: Replaced `torch.mean` with Dot-Product Attention (2026-02-13) — **code-level only, not validated through training**.  
- **Model Portability**: Modifying the source code of every model is unscalable. **Mitigation**: Abstracted THEN into a PyTorch-hook based `portable_memory` module, making it drop-in compatible with standard HuggingFace wrappers (2026-02-22) — **code-level only, not validated through training**.  

**Contingencies**: If Qwen3 issues, revert to Gemma-3-9B (similar size).  

**Final Assumptions**:  

- Tools: PyTorch 2.2+, Transformers 4.40+ (valid).  
- Data: 100M-1B tokens sufficient for proto (untested).  
- Gains: 25-40% from Option C (target, not confirmed).  
- **State Granularity**: Buffering logic ensures 1 trace per chunk (16 tokens) regardless of batch size — **code-level, not validated**.  

This is your executable blueprint—code architecture is ready. Training has not yet begun.

**End of Master Plan**  
*Last Updated: Apr 28, 2026. Status corrected to reflect pre-training reality.*