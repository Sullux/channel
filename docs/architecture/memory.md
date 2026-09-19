# Episodic Memory & Associative Recall

## 1. Overview & Conceptual Foundations

Traditional Large Language Models are fundamentally amnesic between inference calls and brittle within long contexts. Standard Retrieval-Augmented Generation (RAG) and naive sliding context windows suffer from three fatal flaws:
1. **Context Window Exhaustion:** Attention complexity scales quadratically with history ($O(S^2)$), degrading reasoning speed and diluting attention.
2. **Textual Impedance Mismatch:** Serializing state into natural language text strings strips away high-dimensional latent nuance and requires costly re-tokenization and matrix prefill re-computation.
3. **Flat, Monolithic Perception:** Standard architectures treat all historical data identically, failing to distinguish immediate sensory perception from subconscious associations and deliberate retrospection.

The **Channel Inference Engine** models memory on the **Dual-Process Cognitive Architecture** of mammalian neurobiology:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DUAL-PROCESS MEMORY ARCHITECTURE                     │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. IMPLICIT PASSIVE RESONANCE     │ 2. EXPLICIT FOREGROUND REPLAY      │
│    (Subconscious Background)      │    (Deliberate Mental Simulation)  │
│    - Tier-3 Peripheral KV Slots   │    - Primary Token Stream          │
│    - Autonomous Salience Gating   │    - Full 48-Layer Reasoning       │
│    - Low-bandwidth "tag-along"    │    - Replaces sensory focus        │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 2. Hardware Geometry & Storage Architecture

* **Host Platform:** AMD Ryzen AI Max+ with Radeon 890M/800-series (RDNA 3.5 Vulkan Compute).
* **Memory Model:** Unified Memory Architecture (UMA) with 96 GB shared physical RAM.
* **Storage Bus:** PCIe 4.0 NVMe SSD ($\ge 7{,}000\text{ MB/s}$ sequential read/write).
* **Target Model:** Gemma 4 12B Unified:
  * Hidden size: $H = 3840$
  * Layers: $N_{\text{layers}} = 48$
  * KV dimension per layer: $H_{\text{KV}} = 1024$ (8 heads $\times$ 128 head dim)
  * Attention configuration: 8 full-attention layers + 40 sliding-window layers ($W = 1024$)

### Tensor Memory Footprint:
* **Per-Token KV Activation:**
  $$48\text{ layers} \times 2 \,(K, V) \times 1024\text{ elements} \times 2\text{ bytes (FP16)} = 196.6\text{ KB per token}$$
* **Consolidated Episode Block (64 tokens):**
  $$64\text{ tokens} \times 196.6\text{ KB} \approx 12.58\text{ MB per episode}$$
* **Direct NVMe Rehydration:** Rehydrating a 64-token episode takes $\approx \mathbf{1.8\text{ms}}$ over PCIe 4.0, restoring full-rank latent activations with **zero matrix prefill FLOPs**.

---

## 3. The Hippocampal Debounce & Consolidation Window

### Biological Staging
In mammalian neurobiology, raw sensory transients in echoic/working memory require approximately 2–6 seconds of neural rehearsal before hippocampal consolidation commits them into long-term episodic memory.

### Silicon Implementation (`src/hippocampus.zig`)
Writing every individual token activation to disk introduces severe GPU pipeline stalls and memory bus contention. Furthermore, individual sub-tokens carry virtually zero episodic semantic meaning.

Channel introduces an in-memory **Hippocampal Staging Buffer** operating as a temporal debounce:

```
[ Active Decode Stream / Ingestion ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2 GPU Ring Buffer (Sliding Working Window)       │
│  - KV Cache updated per token on GPU                   │
│  - Final hidden state x retained in staging buffer     │
└────────────────────────┬───────────────────────────────┘
                         │
        [ Consolidation Debounce Trigger ]
        - Condition A: Inactivity / Turn boundary (<turn|>)
        - Condition B: Staging buffer capacity (e.g. 64 tokens)
        - Condition C: High activation velocity (||Δx|| > threshold)
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Episodic Consolidation Pass (Background / Non-blocking)│
│  1. Compute consolidated state diff: Δx = x_t - x_prev │
│  2. Compute unit-normalized centroid vector            │
│  3. Read back multi-layer KV slab via zero-copy UMA   │
│  4. Async-flush binary episode record to storage       │
└────────────────────────────────────────────────────────┘
```

---

## 4. Binary File Format Specification (`.episodic.mem`)

The memory store uses a deterministic, 64-byte-aligned, zero-copy memory-mapped file structure.

```
┌────────────────────────────────────────────────────────────────────────┐
│ .episodic.mem BINARY FILE LAYOUT                                       │
├────────────────────────────────────────────────────────────────────────┤
│ 1. FileHeader (64 Bytes, Fixed Offset 0)                               │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Centroid Vector Index Table [Capacity × (64B Meta + 7680B Vector)]  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Multi-Layer KV Tensor Slab Storage [Capacity × Variable Block Slab] │
└────────────────────────────────────────────────────────────────────────┘
```

### A. File Header (64 Bytes)
```zig
pub const FileHeader = extern struct {
    magic: [4]u8,            // 'E', 'M', 'E', 'M' (0x4D454D45)
    version: u32,          // 1
    file_flags: u32,       // Bit 0: FP16 tensors, Bit 1: Q8_0 tensors, Bit 2: Compressed
    num_layers: u32,       // 48
    hidden_size: u32,      // 3840
    kv_dim: u32,           // 1024
    max_episodes: u32,     // Maximum allocated episode count (e.g. 16384)
    total_episodes: u64,   // Total committed episodes written to date
    write_head: u64,       // Next circular append slot index
    model_id_hash: u64,    // 64-bit FNV-1a hash of model weight signature
    reserved: [16]u8,      // Future expansion / alignment padding
};
```

### B. Episode Metadata Header (64 Bytes)
```zig
pub const EpisodeHeader = extern struct {
    episode_id: u64,         // Monotonic globally unique episode index
    parent_episode_id: u64,  // Lineage pointer to ancestor episode
    created_timestamp: u64,  // Milliseconds from epoch (Date.now())
    last_accessed: u64,      // Epoch timestamp of most recent recall
    start_clock: u64,        // Monotonic token clock t at inception
    token_count: u32,        // Number of sequential token slots in KV slab
    access_count: u32,       // Total times recalled into working memory
    child_count: u32,        // Derived episode lineage count
    salience_norm: f32,      // Directional activation magnitude ||Δx||
    continuation_token: u32, // Next predicted token ID after episode
    flags: u16,              // Bit 0: is_interrupted, Bit 1: has_tool, Bit 2: pinned
    summary_len: u16,        // Length of UTF-8 summary slice (≤ 256 B)
};
```

---

## 5. 2D Givens RoPE Delta Re-Rotation

When historical KV tokens are injected into active memory slots ($[3968 \dots 4095]$), their original position embeddings must be adapted to the active coordinate frame without destroying attention legibility.

Channel implements **2D Givens RoPE Delta Re-Rotation**:
* Let the original token clock be $t_{\text{orig}}$ and the target injection clock be $t_{\text{target}}$.
* The delta position offset is:
  $$\Delta t = t_{\text{target}} - t_{\text{orig}}$$
* For each 2D component $(k_{2i}, k_{2i+1})$ with base frequency $\theta_i = 10000^{-2i/D}$:
  $$\begin{pmatrix} k'_{2i} \\ k'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(\theta_i \Delta t) & -\sin(\theta_i \Delta t) \\ \sin(\theta_i \Delta t) & \cos(\theta_i \Delta t) \end{pmatrix} \begin{pmatrix} k_{2i} \\ k_{2i+1} \end{pmatrix}$$
* Values ($V$) remain completely invariant to positional rotation.

This ensures that injected memories participate in self-attention dot products with correct relative temporal distance, completely eliminating RoPE phase decoherence.
