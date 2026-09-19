# Model Selection: Gemma 4 12B Unified

## Executive Summary

For the development and optimization of the **Channel Inference Engine**, we selected **Gemma 4 12B Unified** (with **Gemma 4 E2B** as a rapid iteration testbed) as the primary target foundation model.

This model represents the optimal balance of reasoning capability, clean dense architecture, native multimodal streaming, and hardware efficiency for our target integrated AMD GPU environment (AMD Unified Memory Architecture with 96 GB DDR5 RAM and RDNA 3.5 Vulkan Compute).

---

## 1. Gemma 4 Lineup Comparative Analysis

| Feature / Model | Gemma 4 E2B | Gemma 4 E4B | **Gemma 4 12B Unified** | Gemma 4 26B A4B (MoE) | Gemma 4 31B Dense |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 2.3B (5.1B w/ PLE) | 4.5B (8.0B w/ PLE) | **11.95B** | 25.2B | 30.7B |
| **Active Parameters** | 2.3B | 4.5B | **11.95B** | 3.8B (8 of 128 experts) | 30.7B |
| **Layer Count** | 35 | 42 | **48** | 30 | 60 |
| **Architecture Type** | Dense + PLE | Dense + PLE | **Dense Unified (Encoder-Free)**| Mixture of Experts | Standard Dense |
| **Modalities** | Text, Image, Audio | Text, Image, Audio | **Text, Image, Audio** | Text, Image | Text, Image (No Audio)|
| **Encoder Design** | External Encoders | External Encoders | **Direct Linear Projections** | External Encoders | External Encoders |
| **MMLU Pro** | 60.0% | 69.4% | **77.2%** | 82.6% | 85.2% |
| **LiveCodeBench v6** | 44.0% | 52.0% | **72.0%** | 77.1% | 80.0% |
| **Vocabulary Size** | 262,144 | 262,144 | **262,144** | 262,144 | 262,144 |

---

## 2. Key Architectural Decision Drivers

### A. The "Unified" Encoder-Free Breakthrough
In standard multimodal architectures, processing audio and vision requires running heavy external neural networks (e.g. Vision Transformers / ViTs or Audio Conformers) before feeding intermediate representations to the LLM.

**Gemma 4 12B Unified eliminates external encoders entirely:**
* Raw audio waveforms and visual patches are projected **directly into the LLM's embedding space via lightweight linear projection layers**.
* All modalities are processed natively by the same 48 decoder transformer layers.
* **Impact on Channel:** Drastically simplifies our inference runtime. We only need to implement standard matrix-vector operations and a simple linear ingress projection, rather than porting complex external vision/audio neural pipelines.

### B. Standard Clean Dense Weights
* **Avoids MoE Routing Overhead:** While the 26B A4B MoE is efficient, dynamic gating, top-k expert dispatching, and scattered memory lookups across 128 expert matrices add significant GPU cache thrashing on integrated hardware.
* **Uniform 48-Layer Structure:** Gemma 4 12B provides a standard, predictable 48-layer stack ideal for static Vulkan command recording and 4-layer command chunking.

### C. Native Audio Streaming Capability
* Unlike the 31B Dense model—which drops audio support—the 12B Unified model natively processes audio streams.
* This allows Channel to test **true full-duplex sensory streaming** directly inside our streaming ingestion pipeline.

### D. Strong Reasoning Baseline
With a **77.2% MMLU Pro** and **72.0% LiveCodeBench** score, the 12B model possesses high-fidelity semantic and code understanding. It natively supports the `<|channel>thought` channel and structured function calling contracts.

---

## 3. Hardware & Memory Fit (Integrated AMD Hardware)

### Target Host Specs:
* **Memory Architecture:** AMD Unified Memory Architecture (UMA)
* **Allocated Compute Memory:** 96 GB DDR5 RAM
* **Compute Device:** AMD Radeon 890M / 800-series (RDNA 3.5, Vulkan Compute 1.3)
* **Storage:** PCIe 4.0 NVMe SSD (~7,000 MB/s sequential read)

### Quantization & Memory Footprint:

| Precision / Quantization | Memory Required | Remaining RAM for Ring Buffers & Diffs | Sustained Decode Speed |
| :--- | :--- | :--- | :--- |
| **`bfloat16` (Unquantized)** | ~24.0 GB | ~72.0 GB Headroom | 12–15 tok/s |
| **`Q8_0`** | ~12.5 GB | ~83.5 GB Headroom | 18–22 tok/s |
| **`Q4_0` (Channel Target)** | ~7.2 GB | **~88.8 GB Headroom** | **25–28 tok/s** |

### Architectural Implication:
In pure symmetric zero-centered `Q4_0`, the model occupies just ~7.2 GB of unified memory. Over **88 GB of unified DDR5 RAM remains available** to hold:
* 48-layer unified dynamic ring buffers (4,096 physical slots per layer).
* In-memory episodic Hippocampal KV caches and UMA memory snapshots.
* OS disk cache and VFS files.

---

## 4. Implementation Parameters

* **Model Family:** Gemma 4
* **Target Variant:** 12B Unified
* **Number of Layers ($N$):** 48
* **Hidden Dimension ($D$):** 4,096
* **KV Heads:** 8 Grouped-Query Attention (GQA)
* **Attention Heads:** 32 Query Heads (Head Dim: 128)
* **Vocabulary Size ($V$):** 262,144
* **Attention Mechanism:** RoPE with full 128-dim rotary embedding
* **MLP Non-Linearity:** SwiGLU gated linear unit
* **Reasoning Mode:** Native `<|channel>thought` channel support with autonomic Thinking Gate control
