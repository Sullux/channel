# Channel Engineering Strategy & Invariant Log

## Overview & Philosophy

The **Channel Inference Engine** was developed through rigorous milestone-driven discipline:
1. **Step 1: Foundational Baseline Model Runner** (verified pure Zig execution, mathematical parity, Vulkan compute dispatch, and Q4_0 memory mapping).
2. **Step 2: Streaming Architecture & Dynamic Ring Integration** (unified 4,096-slot geometry, episodic Hippocampal UMA readback, associative RoPE injection, continuous pull-streaming).
3. **Step 3: Autonomic Reflex Plane & Multi-Channel Transduction** (1-token logit probes, Shannon entropy/confidence gating, Thinking Gate, and A-FSM focus management).

---

## 1. Core Engineering Invariants

### A. Pure Zero-Centered Q4_0 Linear Projections
* Channel uses **Pure Symmetric Zero-Centered Q4_0 Quantization** across all linear projections ($Q, K, V, O$, `gate_proj`, `up_proj`, `down_proj`), while maintaining the output vocabulary classifier in **Q8_0** and input token embeddings in **BF16**.
* This matches official Google QAT weights and standard `llama.cpp` quantization geometry, keeping weight memory read per token to $\approx 6.00\text{ GB}$ on 12B Unified, yielding $\ge 24\text{ tok/s}$ sustained decode throughput on AMD UMA hardware.

### B. Fixed 4,096-Slot Physical Ring Geometry
* Physical KV storage is strictly 4,096 slots per layer across all 48 layers:
  - **`0 .. N_sys-1`:** Tier 1 anchors (system prompt & tool contracts), permanently locked.
  - **`N_sys .. 3967`:** Dynamic sliding FIFO working ring.
  - **`3968 .. 4095`:** Dedicated Tier 3 associative recall slots.
* **Prefill Slot Bounding:** Prefill slot lists passed to GPU kernels must always remain $\le 4,096$. Slots cannot collide with Tier 3 recall space.

### C. Vulkan Compute Queue & Driver Watchdog Stability
* All GPU compute dispatches execute on a dedicated asynchronous compute queue.
* **4-Layer Command Chunking:** Command buffers are submitted in 4-layer chunks with fences rather than all 48 layers in one dispatch. This completely eliminates AMDGPU Linux kernel driver watchdog resets (`VK_ERROR_DEVICE_LOST`) under continuous decode workloads.

### D. Single-Thinking-Phase Invariant & Token 100 Suppression
* In Gemma 4, turns permit at most one thinking phase.
* Once token 101 (`<channel|>`) is sampled, the thinking channel is closed for the remainder of the turn.
* Token 100 (`<|channel>`) must be actively suppressed in the sampler during response generation to prevent hallucinated recursive thinking loops.

### E. Explicit Thinking Authority (Thinking Gate as Single Authority)
* Thinking is disabled by default: `suppress_thinking = true` is locked across prefill and decode unless explicitly requested via `INPUT_FLAG_REASON` (`0x02`).
* When deliberate reasoning is engaged, token 101 is masked on token 1 inside the thought channel to prevent premature channel collapse.
* Adaptive thinking depth probes evaluate reasoning sufficiency at soft yields, capping thought early once confidence $\ge 0.80$.

### F. Server-Side Turn Envelopes & Structural Invariants
* The engine tracks `turn_open` state. If a client delivers an interrupt turn while a prior turn was open or yielding, the engine automatically commits token 106 (`<turn|>`), maintaining KV structural validity.
* Delimiters (`<|turn>`, `<|channel>`) belong exclusively to the engine; the host client communicates purely through model-agnostic binary opcodes and semantic flags.

### G. Zero Delimiter Stutter in Inbound Paths
* User turn paths use raw numerical filenames (`msg/user/1001.txt`, not `msg_1001.txt`).
* Notification IDs use uniform typed strings without underscores (`not101`, `not102`) to prevent pseudo-token stutter and hallucinated formatting loops.

### H. Non-Blocking File I/O & Working State Snapshots
* Working state snapshots serialize active slots asynchronously in a detached thread with 1 MB buffered I/O, completing in ~5ms without interrupting active token decode.
* Zero-delta snapshot gating returns status code 2 when the clock has not advanced, eliminating duplicate checkpoints.

---

## 2. Milestone Evolution & Verified Metrics

* **Phase 1 Baseline:** Verified mathematical parity with `llama.cpp` at $\ge 24.3\text{ tok/s}$ and $\le 2.0\text{s}$ prefill.
* **Phase 2 Memory Subsystem:** UMA zero-copy episodic commits (`OP_MEM_COMMIT`), Givens 2D RoPE delta re-rotation, and sub-millisecond centroid cosine scanning.
* **Phase 3 Continuous Streaming:** Elastic syntactic unit gating (`STOP_ELASTIC_YIELD`), in-flight barge-in staging, and VFS pull-streams.
* **Phase 4 Autonomic Architecture:** Discrete 1-token logit probes (`OP_PROBE_AUTONOMIC`) evaluating in ~1.5ms with $O(1)$ clock rollback, Shannon entropy scoring, and A-FSM focus arbitration.
