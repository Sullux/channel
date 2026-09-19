# Research Milestones & Roadmap

This document records the strategic order of operations and future research horizons for the **Channel Inference Engine**.

---

## 1. Completed Foundational Milestones

- [x] **Pure Q4_0 Linear Projection GPU Pipeline:** Fused QKV and SwiGLU shaders with workgroup tiling on Vulkan 1.3 compute.
- [x] **On-Device Top-64 Argmax Reduction & Softcapping:** Replaced 1MB CPU transfers with GPU-side reduction, lifting decode speed to $\ge 24\text{ tok/s}$.
- [x] **Unified 4,096-Slot Physical KV Ring Buffer:** Fixed per-layer geometry with Tier 1 anchors, sliding FIFO ring, and dedicated Tier 3 recall space.
- [x] **Hippocampal Episodic Memory Subsystem:** Zero-copy UMA readbacks (`OP_MEM_COMMIT`), 2D Givens RoPE delta rotation, and sub-millisecond centroid scanning.
- [x] **Continuous Streaming Transduction & Elastic Yielding:** Addressable 512-byte block pull-streams and syntactic unit pauses (`STOP_ELASTIC_YIELD`).
- [x] **Causal In-Flight Barge-In & Interrupt Controller:** Server-side `turn_open` tracking, turn envelope recovery, and LIFO interrupt queueing.
- [x] **Zero-Copy Working State Snapshots:** Compacted active-slot binary checkpointing with detached 1MB buffered background I/O.
- [x] **Autonomic Reflex Plane (`OP_PROBE_AUTONOMIC`):** Discrete 1-token logit probes (~1.5ms) with Shannon entropy/confidence gating, Thinking Gate, and $O(1)$ clock rollback.
- [x] **Unix-Style VFS & Subshell / Terminal Management:** Bounded 512-char slices, async detached `cmd` subshells, and persistent `trm` PTY sessions.
- [x] **Responsive 3-Panel TUI Reference Implementation:** Tiered layout adaptation, abstract Markdown control, and zero-markup controller architecture.

---

## 2. Active Research Horizons

### A. Autonomous Agent Capabilities
- [ ] Implement hierarchical agent task planning and decomposition on top of the A-FSM.
- [ ] Deepen multi-channel sensory arbitration across simultaneous terminals, background processes, and conversation.
- [ ] Develop autonomous self-healing and code editing loops powered by sub-millisecond probe reflexes.

### B. Streaming, Quiescence & Memory Scaling
- [ ] **Large Repository Ingestion Benchmark:** Stream massive multi-file codebases into Channel and evaluate synthesis speed and memory stability.
- [ ] **Quiescence Magnitude Sweeps:** Benchmark throughput and reasoning quality across varying quiescence thresholds ($\tau = 0.001 \dots 0.35$).
- [ ] **Multi-Hop Retrospective Associative Recall:** Test multi-hop associative queries across deep historical episode archives.

### C. Real-Time Online Plasticity (Synaptic Adaptation)
- [ ] Track recurrent memory activation diffs ($\sum \text{Working Memory Tax} > \text{Plasticity Risk Tax}$).
- [ ] Implement fast online micro-LoRA / Hebbian weight updates ($\Delta W$) directly into MLP projection matrices.
- [ ] Verify permanent retention of consolidated facts with zero active context consumption.

### D. Audio Streaming & Real-Time Multimodal Synthesis
- [ ] Stream raw audio PCM frames into Layer 0 via `OP_STREAM_INPUT`.
- [ ] Integrate lightweight real-time speech synthesis (TTS) for full-duplex verbal communication.
