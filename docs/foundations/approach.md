# Approach & Philosophy

## Overview

Rather than scaling monolithic Transformer context windows toward infinity or paging multi-gigabyte dense models across slow memory buses, the **Channel Inference Engine** inverts the fundamental relationship between **layers**, **context**, and **execution**:

> **Instead of a rigid, deep stack of layers processing an unbounded, growing context window with turn-based batch latency, Channel introduces a fixed, compact context allocation per layer with continuous, interruptible streaming transduction and sub-millisecond autonomic reflexes.**

```
┌────────────────────────────────────────────────────────────────────────┐
│ Channel Host / Agent Mind (Node.js)                                    │
│  - A-FSM State Machines & Reducers                                     │
│  - Channel Focus Arbitration (chat/user, subshell logs, terminals)     │
│  - Unix-Style VFS Storage & Asynchronous Subshell/Terminal Manager     │
└────────────────────────────────────────────────────────────────────────┘
                                 ▲
                     Binary Wire Protocol (16-byte frames)
                     OP_STREAM_INPUT, OP_PROBE_AUTONOMIC, OP_RESUME, ...
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Channel Tensor Brainstem (Zig & Vulkan Compute 1.3)                    │
│  - Unified 4,096-Slot Physical Ring Buffer per Layer (0..4095)        │
│  - Pure Symmetric Zero-Centered Q4_0 Linear Projections               │
│  - On-Device Top-64 Softcapped Sampler & Elastic Syntactic Yielding   │
│  - Sub-Millisecond 1-Token Autonomic Probes (Entropy & Confidence)    │
│  - UMA Zero-Copy Episodic Memory & Working State Snapshots             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Core Engineering Principles

### 1. Fixed Context Geometry with Tiered Anchors
Instead of maintaining an unbounded global context window that expands indefinitely ($S \to \infty$) with quadratic memory growth ($O(S^2)$), every layer in Channel maintains a **strictly fixed 4,096-slot allocation** in physical GPU memory.

* **Tier 1 (Immutable Anchors: slots `0..N-1`):** System identity, core prompt, and YAML frontmatter tool contracts, permanently prefilled and never evicted.
* **Tier 2 (Dynamic FIFO Ring: slots `N..3967`):** Working dialogue and stream transduction context managed via a circular write pointer.
* **Tier 3 (Associative Recall: slots `3968..4095`):** Reserved for dynamic episodic memory injection, prime subconscious associative recall, and task switching.

### 2. Autonomic Reflexes Over Generative Function Calling
Traditional agents invoke tools by autoregressively generating 20–60 tokens of JSON or pseudo-code, which the host parses, executes, and appends into the transcript. This consumes 2–5 seconds and permanently bloats the active KV context.

Channel separates conscious generative cognition from subconscious reflexes:
* **Discrete 1-Token Probes:** State machine transitions, channel focus, task triage, and reasoning gates are evaluated via constrained 1-token logit probes directly on the GPU in under 2ms.
* **Mathematical Certainty:** Channel extracts softmax confidence and Shannon entropy directly from raw GPU logits to decide state transitions deterministically.
* **Instant $O(1)$ Clock Rollback:** Transient probe evaluations roll back the logical token clock and ring buffer deactivations immediately, leaving zero pollution in the conversational KV history.

### 3. Elastic Syntactic Unit Gating & Causal Interruption
In real-time environments, user speech, commands, or background terminal output arrive asynchronously. 
* Channel yields execution at natural syntactic boundaries (periods, question marks, exclamation marks, or closing code fences) via `STOP_ELASTIC_YIELD`.
* When a user barges in, active generation pauses cleanly without dropping tokens. The engine tracks `turn_open` state and automatically closes open channel envelopes (`<turn|>`), ensuring KV cache validity.
* Responses and barge-in turns are staged monotonically, guaranteeing strict chronological and causal consistency across panels and logs.

### 4. Zero-Copy Episodic Memory with 2D Givens RoPE Delta Rotation
Rather than degrading historical interactions to plain-text summaries or stuffing thousands of tokens into prompts, Channel retains the rich neural activations of historical turns:
* **UMA Readback:** Historical KV slabs are committed directly from unified memory to disk via `OP_MEM_COMMIT`.
* **Givens RoPE Rotation:** When an episodic memory is recalled (`OP_MEM_QUERY`), keys are re-rotated in 2D coordinate space to match their new relative position in the active attention window, allowing past cognitive states to resonate natively in self-attention.

### 5. Minimal Dependencies & Strict Separation of Concerns
* **Zig for the Brainstem:** Zero external runtime dependencies; pure Vulkan 1.3 compute shaders (WGSL/SPIR-V), direct POSIX file descriptors, and manual memory management.
* **Vanilla JavaScript for the Mind:** High-readability functional factories, declarative controllers, zero transpilation, and strict adherence to low-cognitive-load engineering principles.
* **Model-Agnostic Wire Framing:** The client never formats or parses raw model-specific delimiters (`<|turn>`, `<|channel>`). All turn envelopes belong in the engine; all UI presentations belong in YAML declarations.
