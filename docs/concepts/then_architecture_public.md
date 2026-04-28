# The THEN Architecture: A "Notebook" for Language Models

## The Problem: "Amnesia" in AI

Most Large Language Models (LLMs) suffer from **anterograde amnesia**.

* **Training (Phase 1)**: They learn general knowledge (e.g., "The sky is blue").
* **Inference (Chat)**: They cannot "learn" new things permanently. If you tell them "My name is Sarah," they only remember it for the duration of the context window. Once the chat closes or the window fills up, the information is gone forever.

### Why Current Solutions Have Tradeoffs

1. **SFT / Fine-Tuning**:
    * **The Mechanism**: Re-training the model's weights on new data.
    * **The Pain**: It's slow, expensive, and destructive. Teaching the model "Sarah likes coffee" might overwrite "Sarah is a programmer" (Catastrophic Forgetting). It creates a static snapshot, not a living memory.

2. **RAG (Retrieval-Augmented Generation)**:
    * **The Mechanism**: Using a search engine (Vector DB) to find documents and pasting them into the prompt.
    * **The Pain**: It keeps memory outside the model's learned forward dynamics and can introduce latency and prompt budget pressure.
        * **External Retrieval**: The model reads retrieved text rather than using a dedicated learned memory state.
        * **Latency**: Requires a retrieval step and additional prompt construction.
        * **Context Pressure**: Large retrieved histories compete for context window space and increase token cost.

3. **Plugins / Tools**:
    * **The Mechanism**: Giving the model a "Save to File" tool.
    * **The Pain**: It is **manual and clunky**. The model has to *decide* to call a tool, format a JSON, and hope it wrote the right thing. It's not a natural cognitive process; it's a bureaucratic task.

## The Solution: Temporal History Episodic Network (THEN)

The **THEN** architecture (implemented in this fork) separates **Processing** (Weights) from **Memory** (State).

### 1. The "Notebook" Analogy

Imagine the model is a student taking a test.

* **Weights (Frozen)**: The student's brain. It knows how to read, write, and reason. We train this *once* to be smart.
* **State (Live)**: A notebook on the desk.
* **Ingest Phase**: When you tell the model something new ("My name is Sarah"), it doesn't change its brain. Instead, it **writes a note** in the notebook.
* **Query Phase**: When you ask "What is my name?", the model uses its brain to **read the note** and answer "Sarah."

### 2. How It Works (Simplified)

Instead of updating the neural network's parameters (which requires a massive GPU cluster), we append compressed vectors to a `state` file.

* **Ingest**: `User Input -> [Compress] -> Memory Trace`
* **Recall**: `Query -> [Attention] -> Retrieve Trace -> Answer`

### 3. Key Benefits

1. **Instant Learning**: New facts are available immediately. No waiting for a training run.
2. **Weight Stability**: Since the design writes to state rather than weights during ingestion, it avoids direct weight overwriting for new episodic information.
3. **Privacy Control**: The "notebook" is just a file. In principle, you can delete specific pages (memories) or burn the whole book without changing the model's core weights.
4. **Potential Cost Benefits**: If the mechanism works reliably, one frozen model could potentially serve many users with separate state files.
5. **Portable Implementation Path**: The current prototype includes PyTorch `forward_hooks` intended to wrap standard HuggingFace models without invasive source-code modifications.
---

CURRENT STATUS (2026-04-28): The THEN architecture has been implemented as code modifications to nanochat with a portable hook-based path for HuggingFace models. All unit tests pass for mechanical plumbing. However, no model training has been executed — the architecture has not yet been shown to produce causal recall advantages in any benchmark. The `tiny_recall_benchmark.py` scaffold is the next step.

*This architecture is intended as a path from static weights toward a more stateful form of personalization without continuous retraining, but its practical advantages still require controlled validation.*
