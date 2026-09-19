# Channel Architecture Overview

The **Channel Inference Engine** decouples modern LLM execution into a **Dual-Plane Architecture**: a high-speed, hardware-accelerated **Tensor Brainstem** running directly on the GPU in C++/Zig, and an autonomous **Cognitive Mind** running on the host in Node.js.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE MIND (Host Runtime)                   │
├────────────────────────────────────────────────────────────────────────┤
│ • Autonomic Finite State Machines (A-FSM Reducers & Action Grammars)   │
│ • Channel Focus Arbitration (chat/user, subshell streams, terminals)   │
│ • Unix-Style VFS Storage & Asynchronous Subshell/Terminal Manager      │
│ • LIFO Notification Interrupt Stack & Turn Ordering Controller        │
└────────────────────────────────────────────────────────────────────────┘
                                   ▲
               16-Byte Binary Wire Protocol (TCP / Local Socket)
               OP_STREAM_INPUT, OP_PROBE_AUTONOMIC, OP_RESUME, ...
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      TENSOR BRAINSTEM (Engine Runtime)                 │
├────────────────────────────────────────────────────────────────────────┤
│ • Unified 4,096-Slot Physical KV Ring Buffer per Layer (0..4095)       │
│ • Pure Symmetric Zero-Centered Q4_0 Linear Projections (Vulkan 1.3)    │
│ • On-Device Softcapped Top-64 Sampler & Elastic Syntactic Yielding     │
│ • Sub-Millisecond 1-Token Autonomic Probes (Entropy & Confidence)      │
│ • UMA Zero-Copy Episodic Memory & Working State Snapshots              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Architectural Subsystems

### 1. [Autonomic Reflex Plane](autonomic.md)
Separates conscious generative cognition from subconscious reflexes. Channel evaluates internal routing, state transitions, task triage, and reasoning depth via constrained 1-token logit probes directly on the GPU in under 2 milliseconds. Using on-device Shannon entropy and softmax confidence scoring, the engine executes reflexes with mathematical certainty and rolls back the KV clock in $O(1)$ without polluting conversational context.

### 2. [Episodic Memory & Associative Recall](memory.md)
Maintains long-term continuity without prompt-stuffing or text-based RAG. Channel captures full-rank latent activation slabs from the KV cache via zero-copy UMA readbacks (`OP_MEM_COMMIT`). When associative resonance matches a historical episode (`OP_MEM_QUERY`), keys are re-rotated in 2D coordinate space via Givens delta rotation and injected directly into attention slots, restoring past cognitive state in ~0.05ms without matrix prefill FLOPs.

### 3. [Physical KV Ring Buffer Geometry](ring-buffer.md)
Enforces a strictly bounded 4,096-slot physical geometry across all 48 transformer layers. System prompts and immutable tool contracts are locked permanently into Tier 1 anchors (`0..N-1`), while working dialogue flows through a sliding FIFO ring (`N..3967`). Dynamic recall slots (`3968..4095`) provide dedicated capacity for episodic injection without memory copy or slot collision.

### 4. [Continuous Streaming Transduction](streaming.md)
Replaces turn-based batch prefill and decode with continuous, pull-stream ingestion over 512-byte blocks. The engine yields execution at natural syntactic units (`STOP_ELASTIC_YIELD`), allowing user barge-ins to be captured mid-sentence without losing in-flight thought or response state, and automatically enforces canonical turn envelope closure (`<turn|>`).

### 5. [Zero-Copy Working State Snapshots](snapshots.md)
Enables instantaneous session pause, resume, and checkpointing. Active KV cache slots are compacted into a single contiguous binary slab alongside logical clock and anchor metadata, writing to disk asynchronously in a background thread with 1MB buffered I/O, completing checkpoints in ~5ms without blocking active decode.

### 6. [Reference TUI Architecture](tui.md)
A production-grade reference implementation showing how to construct an autonomous, multi-channel agent on Channel. Demonstrates zero-markup declarative UI patterns, Markdown AST rendering via `@sullux/markdown-compiler`, a Unix-style Virtual File Subsystem (VFS) with `chmod 0444` user turn sealing, bounded 512-character reading slices, asynchronous subshell execution (`cmd`), persistent terminal sessions (`trm`), and a LIFO interrupt controller.

### 7. [Client Implementation Guide](../api/client-guide.md)
Architectural guide for building any host harness or agent runtime (robotics, cloud microservices, voice agents) that drives Channel over its binary wire protocol.

### 8. [Binary Wire Protocol](../api/binary-protocol.md)
The model-agnostic binary communication layer between host applications and the Tensor Brainstem. Operates over a fixed 16-byte header with big-endian opcodes, decoupling all model-specific tokens (`<|turn>`, `<|channel>`) from the client runtime.
