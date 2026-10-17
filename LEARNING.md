# Real-Time Learning, Neurogenesis & The Dream Cycle

## 1. Executive Summary & Foundational Philosophy

Traditional Large Language Models (LLMs) operate under a frozen paradigm: once trained, model weights are strictly read-only. Standard runtimes (e.g. `llama.cpp`, vLLM, Ollama) treat all post-training adaptation through two crude mechanisms:
1. **Context Stuffing (RAG):** Prepending raw text chunks into the prompt, which exhausts VRAM, introduces quadratic attention overhead ($O(S^2)$), and vanishes the moment the session ends.
2. **Offline Fine-Tuning / LoRA:** Heavyweight offline batch training requiring reverse automatic differentiation (autodiff), backward gradient descent pipelines, and complex adapter switching.

When viewed from a systems engineering and neurophysiological perspective, both approaches miss the fundamental nature of biological intelligence:
- Mammals do not re-read their entire autobiography every morning to remember how to do their jobs.
- Mammals do not run full-brain backpropagation during waking hours.
- Instead, the brain employs a **Dual-Stage Memory & Consolidation Lifecycle**:
  - **Awake (Fast Episodic Plasticity):** The hippocampus rapidly captures high-salience experiences with zero-copy efficiency into working memory.
  - **Asleep (The Dream Cycle):** During slow-wave sleep and rest, the brain replays high-salience episodic trajectories, extracts recurring patterns, and consolidates them into the neocortex via **neurogenesis and synaptic plasticity**, converting temporary memories into **innate, permanent instincts**.

**Channel** implements this biological lifecycle natively in silicon:
- **Private Per-Agent Model Weights:** Each agent (e.g. `~/agents/hopper`) owns its personal, unquantized model weights (`model.safetensors`).
- **Neurogenesis over Weight Mutation:** Rather than mutating existing pre-trained weights and risking catastrophic forgetting, Channel expands intermediate MLP layer dimensions by appending dedicated **Agent Memory Neurons**.
- **The Dream Cycle:** An offline, deterministic consolidation pass that processes write-ahead logs (`.stream.jsonl`) and memory slabs (`.episodic.mem`), compiles recurring action trajectories into motor chunks, and bakes them directly into disk weights.
- **Zero-FLOP Inference Decay:** Synaptic decay and spaced repetition are tracked in a lightweight metadata registry (`.agent/neurons.json`) and baked directly into weight scaling factors during dream cycles, costing literally zero compute during live inference.

---

## 2. Why LoRA Was Rejected for Channel

Low-Rank Adaptation (LoRA) has become the default fine-tuning standard in the wider AI ecosystem. However, within Channel's architecture, LoRA is an architectural anti-pattern.

### The Problem LoRA Was Invented to Solve
LoRA was created as an escape hatch for developers who were **completely locked out of the inference engine**. In black-box environments (PyTorch, closed APIs, proprietary cloud engines), users cannot inspect intermediate activations, cannot manipulate physical ring buffers, and cannot stream multi-layer KV slabs. LoRA allowed them to monkey-patch weight matrices without retraining the entire 70-billion-parameter network.

### The Channel Reality
In Channel, we built the engine from bare metal:
* We own the Vulkan 1.3 compute pipeline.
* We own the unified physical ring buffers across all 48 layers.
* We stream multi-layer KV slabs to NVMe via zero-copy unified memory (`OP_MEM_COMMIT`).
* We rehydrate associative episodic memories directly into live attention in sub-millisecond time (`OP_MEM_QUERY`).

#### Trade-Off Analysis: LoRA vs. Episodic Memory Injection

| Dimension | Micro-LoRA Overlays | Channel Episodic Injection |
| :--- | :--- | :--- |
| **Engine Complexity** | **Severe:** Requires full backward autodiff graph, activation caching, and reverse Vulkan shaders in Zig. | **Zero:** Uses already-built forward compute pipeline and UMA memory readback. |
| **Adaptation Latency** | Minutes to hours of gradient descent backprop. | **Instantaneous:** Injected in $\sim 0.05\text{ms}$ upon query match. |
| **Multi-Layer Depth** | Typically applied only to attention projections ($Q, V$). | Injects rich, pre-computed $K, V$ activations across **all 48 layers** simultaneously. |
| **Memory Footprint** | Additional GPU descriptor sets and adapter buffers. | Native utilization of existing 4,096-slot dynamic ring buffer. |

**Architectural Decision:** Channel completely rejects LoRA for runtime adaptation. Runtime factual and contextual adaptation is handled natively via **Episodic Memory Injection**, while permanent structural consolidation is handled via **The Dream Cycle**.

---

## 3. Neurogenesis: Appending Neurons vs. Mutating Weights

When updating model weights on disk to consolidate learning, there are two distinct engineering paths:
1. **Synaptic Mutation:** Modifying the existing 14,336 intermediate neurons in the pre-trained MLP layers.
2. **Biological Neurogenesis:** Appending brand-new intermediate neurons to the layer geometry (e.g. expanding $14,336 \to 14,400$).

```
                         MLP LAYER EXPANSION (Neurogenesis)
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[ Base Pre-Trained Weights (14,336 Neurons) ]             [ Appended Agent Neurons (64 Neurons) ]
- Pure Google QAT Pre-Training                            - Hopper's Unique Lifetime Experience
- 100% Frozen, Bit-Exact, Immutable                       - Procedural action macros & custom facts
- Guarantees zero regression in coding/reasoning          - Dynamically pruned or reinforced over time
```

### Why Neurogenesis is the Superior Architecture

#### 1. Mathematical Guarantee Against Catastrophic Forgetting
In standard neural networks, modifying pre-existing weights to learn a new task invariably degrades performance on previously learned tasks (catastrophic interference). By leaving the original 14,336 neurons completely untouched, the model's core linguistic fluency, reasoning capability, and foundational coding competence remain **100% bit-exact and pristine**.

#### 2. Trivial Vulkan Compute Scaling
In modern Vulkan compute shaders, matrix dimensions are passed as dynamic **Push Constants** at dispatch time:
```zig
const PushConstants = extern struct {
    dim_m: u32,
    dim_k: u32,
    dim_n: u32, // Passed dynamically as 14336, 14400, etc.
};
```
Expanding the intermediate dimension requires zero shader recompilation and zero pipeline recreation. The GPU simply executes a few additional workgroups during the fused MLP dispatch.

#### 3. Clean Binary Persistence in `model.safetensors`
A `.safetensors` file is an uncompressed, memory-mappable binary layout. Expanding a tensor involves:
1. Updating the JSON header property: `"shape": [3840, 14400]` instead of `[3840, 14336]`.
2. Appending the raw 16-bit floating-point bytes (`BF16`) to the file.
3. Updating byte offset pointers for subsequent tensors.

---

## 4. The Anatomy of an Appended Neuron

In Gemma 4 12B Unified, each transformer layer's MLP operates as an associative key-value memory:
* `gate_proj` & `up_proj` ($3840 \to D_{\text{mlp}}$): **Key Detectors** (detecting specific concepts in the thought vector).
* `down_proj` ($D_{\text{mlp}} \to 3840$): **Value Injectors** (injecting associated facts or subsequent actions into the residual stream).

When an appended neuron $k$ is added to Layer $l$:
1. **The Detector Row ($w_{\text{gate}}[k], w_{\text{up}}[k]$):** Set to the unit-normalized activation vector of the triggering concept (e.g. the latent thought pattern for *"Natasha"* or *"Spreadsheet Cell B4"*).
2. **The Injector Column ($w_{\text{down}}[k]$):** Set to the target conceptual payload (e.g. the vector representing *"Bookkeeping / Sullux"* or *"Type =SUM("*).

### The "Runway" Effect: Pushing Enrichment Down
By baking an association into an intermediate layer (e.g. Layer 25) rather than relying on upper-layer resolution:
* When the trigger concept enters Layer 25, the appended neuron fires, injecting the associated knowledge immediately.
* **The token now enjoys 23 full layers of deep reasoning (Layers 26 through 48)** to process deductions on top of that fact.
* The model can reason about complex implications of learned facts even when the factual relationship was **never stated in the active context prompt**.

---

## 5. The Dream Cycle Lifecycle

The Dream Cycle is a detached, offline consolidation process executed during system idle periods, overnight maintenance, or explicit administrative sleep states.

```
[ Waking Hours: Streaming Execution ]
  - Actions logged to .stream.jsonl
  - High-salience episodes saved to .episodic.mem
                         │
                         ▼
[ Dream Cycle Trigger (Idle / Scheduled) ]
                         │
                         ├── Phase 1: Sequence & Habit Mining (Trie Analysis)
                         │   - Scan .stream.jsonl for deterministic paths
                         │   - Validate recurrence, entropy, and success
                         │
                         ├── Phase 2: Neurogenesis Compilation
                         │   - Append new neurons for qualified action chunks
                         │   - Write updated tensors to model.safetensors
                         │
                         ├── Phase 3: Synaptic Activity Auditing
                         │   - Dot-product scan over historical .episodic.mem
                         │   - Update last_fired_at in .agent/neurons.json
                         │
                         └── Phase 4: Synaptic Decay & Pruning
                             - Apply Ebbinghaus decay multiplier to down_proj
                             - Prune dead probationary neurons
```

### Phase 1: Sequence Mining via Trie Analysis
The Dream Sequencer processes the waking action logs in `.stream.jsonl` using deterministic Trie path analysis:
1. **Recurrence Threshold:** The trajectory must have occurred $\ge N$ times (default: 3).
2. **Low Branching Entropy:** The transition from Step $A \to B \to C$ must have an empirical conditional probability $P(B \mid A) \ge 0.85$.
3. **Positive Terminal Outcome:** The sequence must have terminated in a verified success state (subshell exit code 0, explicit user confirmation, task marked completed) with **zero barge-in interruptions or error rollbacks**.

### Phase 2: Compiling Action Chunks into Weights
When an action trajectory qualifies, the sequencer compiles the transition:
* The state at Step 1 is encoded into the detector vector of an appended neuron.
* The action at Step 2 is encoded into the injector vector.
* The sequence becomes a chain of internal neural domino triggers, enabling the agent to execute the multi-step macro as a single reflex.

---

## 6. Synaptic Decay, Spaced Repetition & The Neuron Registry

In biological intelligence, forgetting is **not a hardware failure**; it is an active protective mechanism designed to **unlearn obsolete habits** (e.g. avoiding clicking on an old, decommissioned login portal button).

### The Neuron Registry (`.agent/neurons.json`)
The agent maintains a lightweight metadata registry tracking the health and lifespan of every appended neuron:

```json
{
  "version": 1,
  "neurons": {
    "layer_28_neuron_14337": {
      "id": "macro_spreadsheet_sum",
      "layer": 28,
      "index": 14337,
      "created_at": 1776000000000,
      "last_fired_at": 1776500000000,
      "fire_count": 28,
      "half_life_days": 60.0,
      "strength": 1.0,
      "status": "consolidated"
    },
    "layer_25_neuron_14338": {
      "id": "temp_project_staging_url",
      "layer": 25,
      "index": 14338,
      "created_at": 1776400000000,
      "last_fired_at": 1776410000000,
      "fire_count": 3,
      "half_life_days": 7.0,
      "strength": 0.85,
      "status": "probationary"
    }
  }
}
```

### Zero-Overhead Activity Auditing via Stored Memories
Rather than burdening live streaming inference with GPU profiling counters or turn-taking bitmasks, **the Dream Sequencer calculates `last_fired_at` directly from historical memory slabs**:
* For each appended neuron $k$ at Layer $l$, the sequencer takes its detector vector $w_{\text{gate}}[k]$.
* It computes the dot product against stored episode activation vectors in `.episodic.mem`:
  $$\text{Activation} = w_{\text{gate}}[k] \cdot x_E^{(l)}$$
* If $\text{Activation} > 0.50$, the neuron participated in that episode.
* The sequencer sets `last_fired_at = episode.created_timestamp`.

### Zero-FLOP Inference Decay via Weight Scaling
During the Dream Cycle, the sequencer evaluates the time elapsed since the neuron last fired:
$$\Delta t = \text{Date.now}() - \text{last\_fired\_at}$$

If $\Delta t > \text{half\_life}$:
$$\text{strength}_{\text{new}} = \text{strength}_{\text{old}} \times \lambda_{\text{decay}} \quad (\text{e.g. } \lambda = 0.75)$$

To apply this decay during live inference:
* The sequencer **multiplies the neuron's row in `down_proj` directly by $\lambda_{\text{decay}}$** in `model.safetensors`:
  ```zig
  for (down_proj_row) |*weight| {
      weight.* *= strength_decay;
  }
  ```
* **During live inference:** The Vulkan compute shader executes its standard, unencumbered matrix multiplication. The decay is physically baked into the magnitude of the weights. The signal injected into the token is naturally and effortlessly weaker, **costing exactly zero runtime FLOPs**.

---

## 7. Two-Tiered Retention Policy: Habits vs. Foundational Wisdom

Channel addresses the risk of catastrophic forgetting versus memory bloat through a **Two-Tiered Retention Policy**:

```
┌────────────────────────────────────────────────────────────────────────┐
│ TIER 1: PROBATIONARY HABITS (Fire Count < 20)                          │
├────────────────────────────────────────────────────────────────────────┤
│ • Short, dynamic half-life (3 days ──► 7 days ──► 14 days)             │
│ • Designed for transient tasks, temporary URLs, and project scratchpads│
│ • If disused, decays below threshold (strength < 0.10)                 │
│ • ACTION: Pruned and deleted from model.safetensors                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ TIER 2: CONSOLIDATED WISDOM (Fire Count ≥ 20)                          │
├────────────────────────────────────────────────────────────────────────┤
│ • Half-life expands exponentially (months to years)                    │
│ • Represents foundational life experience, core identity, and mastery  │
│ • If disused, decays to an Experience Floor (e.g. strength = 0.20)      │
│ • ACTION: IMMUNE TO PRUNING. Never deleted from model.safetensors.     │
│   Preserves latent historical priors for future design and reasoning   │
└────────────────────────────────────────────────────────────────────────┘
```

### Configurable Hardware Governance (`tui/config.json`)
The host allows fine-grained governance over growth and pruning policies based on local hardware resources:

```json
{
  "learning": {
    "enableDreamCycle": true,
    "maxAppendedNeuronsPerLayer": 256,
    "consolidationThreshold": 20,
    "decayFloor": 0.20,
    "probationaryPruneThreshold": 0.10,
    "decayHalfLifeBaseDays": 3.0,
    "hardwareBudget": "conservative"
  }
}
```

On resource-constrained devices, `"hardwareBudget": "conservative"` enforces strict limits on layer expansion and prunes aggressively. On 96GB workstation hardware, `"hardwareBudget": "unbounded"` permits the agent to grow thousands of lifelong neurons, evolving into a uniquely skilled, permanently adapted individual intelligence.
