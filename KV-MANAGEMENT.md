# KV Cache Management, Compaction & Episodic Consolidation

## 1. Executive Summary & Architectural Convergence

In earlier iterations of the **Channel Inference Engine**, memory and context management were partitioned across three loosely coupled subsystems:
1. **Tier 3 Physical Partition:** Exactly 128 physical KV slots ($3968 \dots 4095$) permanently carved out of the 4,096-slot ring to hold recalled episodic memories.
2. **Hippocampal Debounce Loop:** A detached 6-second timer attempting to predict conversational turn completion and flush staging buffers to `.episodic.mem`.
3. **Dynamic Ring GC:** An independent, ad-hoc slot-eviction routine running low-salience pruning whenever the sliding ring encountered boundary pressure.

While functional, this design suffered from rigid capacity ceilings (episodes $> 128$ tokens overflowed; episodes $< 128$ wasted VRAM), redundant background thread contention, and duplicate salience calculations.

Channel refactors this into a **Single Unified Consolidation & Compaction Architecture**:

```
                       LEGACY ARCHITECTURE (3 Subsystems)
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│ Tier 3 Partition      │  │ Hippocampus Debounce  │  │ Ring Buffer GC        │
│ Fixed slots 3968..4095│  │ 6-second timer to NVMe│  │ Evicting low-salience │
└───────────────────────┘  └───────────────────────┘  └───────────────────────┘
                                   │
                                   ▼
                       NEW UNIFIED ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED CONSOLIDATION & COMPACTION ENGINE                 │
│  - One homogeneous physical ring (Tier 1 Anchors + Tier 2 Dynamic Ring)     │
│  - Recalled memories stream directly into active context as first-class span│
│  - High-water mark triggers single pass: Evict noise, Consolidate to disk   │
│  - Full working state backed by atomic NVMe snapshots (sub-100ms restore)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Homogeneous Physical Geometry (Abolishing Tier 3)

The physical KV cache allocation remains strictly bounded to **4,096 physical slots per layer across all 48 layers**, eliminating memory fragmentation and out-of-memory faults.

However, the internal partitioning is drastically simplified:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Total Fixed Allocation: 4,096 Physical Slots per Layer                 │
├─────────────────────────┬──────────────────────────────────────────────┤
│ Tier 1: Immutable       │ Tier 2: Unified Dynamic Sliding Ring         │
│ System Anchors          │ (Conversational dialogue, sensory streams,   │
│ [Slots 0 .. N_anchor-1] │  and rehydrated episodic memory spans)       │
│                         │ [Slots N_anchor .. 4095]                     │
└─────────────────────────┴──────────────────────────────────────────────┘
```

### Key Architectural Invariants:
1. **No Dedicated Recall Partition:** Rehydrated memories are streamed directly into Tier 2 as a standard continuous span prefixed by a synthetic temporal token (e.g. `[SYNTH_START_MEM] [SYNTH_TIME_TODAY]`).
2. **First-Class Ring Citizen:** Recalled memories participate naturally in causal self-attention alongside active dialogue. If the model and user continue discussing the recalled topic, its attention mass increases and it remains resident in VRAM. If the topic changes, its attention mass decays and it is naturally pruned during the next compaction pass.
3. **Dynamic Variable Length:** A recalled memory can be 8 tokens or 350 tokens; it is no longer artificially shoehorned into a static 128-slot partition.

---

## 3. System Prompt Compaction Before Pinning

In traditional runtimes, the system prompt and tool definitions are ingested in full verbatim fidelity into early slots ($0 \dots N-1$) and permanently pinned. For complex agent prompts with verbose guidelines and multi-tool schemas, this pins 400 to 800 slots—consuming up to 20% of the entire 4,096-slot working context.

Channel introduces **Pre-Pinning System Prompt Compaction**:

```
[ Cold Boot: Ingest Raw System Markdown Prompt (~450 Tokens) ]
                              │
                              ▼
        [ Forward Pass Through Layer 47 (Initial Prefill) ]
                              │
                              ▼
        [ Attention Sink Invariant: Permanently Lock Slots 0..3 ]
                              │
                              ▼
        [ Layer 48 Greedy MMR Submodular Compaction Pass ]
        - Prunes syntax boilerplate ("You are an agent...", "Ensure you...")
        - Retains dense semantic tokens (core tools, entities, constraints)
        - Compresses 450 tokens down to ~120 dense slots
                              │
                              ▼
        [ Permanently Pin Surviving ~120 Slots as Tier 1 Anchors ]
```

### The Attention Sink Law
Under the *StreamingLLM* mathematical invariant, autoregressive transformers require **Token 0** (and the first 2–4 tokens) to absorb baseline softmax probability mass. If Token 0 is pruned or shifted, generation perplexity explodes.
* **Invariant:** Channel permanently locks **Slots 0..3** without compaction.
* Compaction operates strictly across slots $4 \dots N_{\text{sys}}-1$, distilling the instructions down to their mathematical essentials before the write pointer is opened to Tier 2.

---

## 4. The High-Water "Compact-and-Persist" Lifecycle

Channel replaces both the arbitrary 6-second debounce timer and the independent ring GC with a single, deterministically triggered **High-Water Compaction & Consolidation Pass**:

```
[ Active Stream Ingestion / Token Decode Loop ]
                       │
                       ▼
        [ Saturation Check (isWorkingSetSaturated) ]
        - Condition A: Fill Ratio >= 80% (3,276 active slots)
        - Condition B: Gini Coefficient < 0.25 (Diffuse Attention)
                       │
          ┌────────────┴────────────┐
          ▼ (No)                    ▼ (Yes: Saturated)
     [ Continue ]             [ Trigger Compaction Pass ]
                                    │
                                    ├── Zone A (Slots T - 1024 .. T):
                                    │   UNTOUCHABLE FULL FIDELITY
                                    │   (Immediate conversational rhythm)
                                    │
                                    └── Zone B (Older Slots N_anchor .. T - 1025):
                                        1. Calculate Layer 48 Attention Mass
                                        2. Calculate Latent Subspace Residuals
                                        3. Select Top-K Salient/Novel Slots
                                        4. COMMIT surviving slots to .episodic.mem
                                        5. DEACTIVATE evicted slots in active_slots
```

---

## 5. Mathematical Algorithms for Zone B Compaction

### 5.1 Accumulated Attention Mass ($S_j$)
At Layer 48, the attention mass of historical slot $j$ is the cumulative attention directed at it by all subsequent tokens $i > j$:
$$\text{Mass}(j) = \sum_{i > j} A_{i, j}^{(48)}$$
Tokens with $\text{Mass}(j) < 0.10$ represent transient grammatical glue (e.g. `the`, `is`, `of`) and are flagged for immediate eviction.

### 5.2 Latent Subspace Redundancy Elimination (MMR)
To prevent keeping multiple tokens that encode identical information (e.g. *Natasha* when *Natasha Senior* appeared later), Channel applies **Orthogonal Subspace Projection**:

For candidate slot $j$ with Layer 48 Value vector $V_j$ against already selected slots $S$:
$$\text{Novelty}(j \mid S) = 1.0 - \max_{s \in S} \left( \frac{V_j \cdot V_s}{\|V_j\| \|V_s\|} \right)$$

The selection score balances importance and novelty:
$$\text{Score}(j) = \text{Mass}(j) + \lambda \cdot \text{Novelty}(j \mid S)$$

The engine greedily selects the top $K$ slots that maximize $\text{Score}(j)$, producing a distilled set of dense semantic vectors that span the cognitive space of the historical conversation.

---

## 6. Episodic Storage & Hebbian Denormalization

When surviving tokens from Zone B are consolidated:
1. **Multi-Layer Extraction:** The key ($K$) and value ($V$) vectors across all 48 layers for the selected slots are extracted directly from Vulkan UMA memory.
2. **Centroid Unit Vector:** The mean vector of their Layer 48 hidden states is computed and normalized to form the episode's search centroid.
3. **Appended to `.episodic.mem`:** Written sequentially to the persistent memory-mapped file alongside token count, start clock, and epoch timestamp.

### Natural Hebbian Reinforcement (Denormalization)
If a previously rehydrated episodic memory is still receiving high attention in Zone B when the compactor runs:
* It is **persisted again** to `.episodic.mem` with an updated timestamp and incremented `access_count`.
* This natural denormalization acts as biological synaptic reinforcement: facts that remain relevant across multiple conversational epochs are strengthened in the associative index.
* If a recalled memory was not referenced, its attention mass remains zero; it is simply deactivated and forgotten from working VRAM without redundant disk writes.

---

## 7. Crash Recovery via Zero-Copy Snapshots

The legacy 6-second debounce was originally implemented as an insurance policy against unexpected server crashes. 

Under the new architecture, **Working State Snapshots (`OP_SNAPSHOT_SAVE`) completely subsume this requirement**:
* At natural conversational pauses (`STOP_END_OF_TURN`), the engine serializes compacted active KV slots to `.memory/.snapshot.bin` in ~5ms via 1MB buffered background I/O.
* If a power outage or crash occurs:
  - Cold boot prefill is completely bypassed ($0.0\text{s}$).
  - Working memory, dynamic ring pointers, and active slot maps are restored in **$< 100\text{ms}$** with 100% exact bit-level tensor parity.
  - Any unpersisted Zone B memories are restored into active context exactly as they were before the interruption.
