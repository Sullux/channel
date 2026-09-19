<p align="center">
  <img src="logo.svg" alt="Channel Logo" width="160" height="160" />
</p>

# Channel Inference Engine

**Channel** is a continuous streaming inference engine with autonomic reflexes, tailored for the Gemma 4 model family on integrated AMD GPU hardware (RDNA 3.5 / Vulkan Compute).

Unlike traditional turn-based inference servers (e.g. Ollama, llama.cpp, vLLM) that operate on monolithic prompt dumps and batch outputs, Channel models LLM execution as a continuous, stateful, interruptible sensory stream.

---

## Core Pillars

1. **Continuous Streaming Transduction**
   - Ingests raw inputs via addressable 512-byte block pull-streams.
   - Decodes tokens with elastic syntactic unit yielding (`STOP_ELASTIC_YIELD`) at natural punctuation resting points, allowing true mid-stream user barge-in without dropping in-progress thought or response state.
2. **Sub-Millisecond Autonomic Reflexes (`OP_PROBE_AUTONOMIC`)**
   - Bypasses the costly multi-second LLM tool-calling cycle for kernel operations.
   - Evaluates discrete 1-token logit probes directly on the GPU in under 2 milliseconds, extracting Shannon entropy and softmax confidence, followed by instantaneous $O(1)$ KV clock rollback without polluting conversational context.
   - Features the **Thinking Gate** (bypassing deliberate reasoning when simple response certainty is high) and **Adaptive Thinking Depth** (capping thought at elastic yields once a solution is found).
3. **Unified 4,096-Slot Physical KV Ring Buffer**
   - Preserves Tier 1 anchor tokens (system prompt & tool definitions) permanently at slots `0..N-1`.
   - Uses an active dynamic FIFO ring for working conversation context.
   - Dynamically routes slots `3968..4095` for high-speed associative recall without CPU memory copies.
4. **Episodic Associative Memory Subsystem**
   - Zero-copy UMA readback for Hippocampal episodic KV cache commits (`OP_MEM_COMMIT`).
   - 2D Givens RoPE delta re-rotation for seamless episodic memory injection (`OP_MEM_QUERY`) into live attention streams.
   - Continuous salience tracking via attention mass accumulation and Gini coefficient distribution analysis.
5. **Pure Q4_0 Vulkan Compute Pipeline**
   - Hardware-tailored for RDNA 3.5 AMD integrated GPUs using pure symmetric zero-centered Q4_0 linear projections across all layers.
   - Fused QKV, fused SwiGLU MLP, and batch causal attention WGSL/SPIR-V shaders.
   - 4-layer command submission chunking ensuring 100% OS and AMDGPU driver watchdog stability under continuous sustained loads.
6. **Decoupled Dual-Plane Architecture**
   - **Tensor Brainstem (Zig on GPU):** High-speed compute, KV ring maintenance, UMA synchronization, autonomic probes, and raw token streaming.
   - **Cognitive Mind (Host Runtime / TUI on Node.js):** Autonomic finite state machines (A-FSM), channel focus arbitration, task scheduling, VFS storage, and asynchronous subshell/terminal process orchestration.
   - Connected via a compact, model-agnostic binary wire protocol (`OP_STREAM_INPUT`, `OP_PROBE_AUTONOMIC`, `OP_RESUME`, `OP_TOOL_RETURN`, `OP_SNAPSHOT_SAVE`, `OP_SNAPSHOT_LOAD`).

---

## Documentation Roadmap

* **[Quick Start Guide](quick-start.md)**: Fast track installation, prerequisites, and running the reference TUI or CLI in minutes.
* **[Foundations](foundations/problem.md)**: Explore the motivation behind Channel—context window saturation, prefill latency, and the LLM tool-calling tax.
* **[Model Selection](foundations/model-selection.md)**: Why Gemma 4 (E2B and 12B-it) was chosen as the architectural baseline.
* **[Philosophy & Approach](foundations/approach.md)**: Core design principles—local-first, minimal dependencies, and hardware parity.
* **[Architecture Hub](architecture/README.md)**: Deep dive into the Dual-Plane system, memory topology, and streaming pipelines.
* **[Reference TUI Architecture](architecture/tui.md)**: Production-grade reference agent implementation showcasing zero-markup YAML controllers, Markdown AST rendering, VFS storage, and LIFO interrupts.
* **[Command Line Interface (CLI)](api/cli.md)**: Exhaustive reference for command-line arguments, operational modes, and environment variables.
* **[Binary Wire Protocol](api/binary-protocol.md)**: Complete byte-level reference for the 16-byte framing protocol and opcodes.
* **[Client Implementation Guide](api/client-guide.md)**: Architectural guide for building custom clients (robotics, cloud microservices, voice agents) on Channel.
* **[Engineering Log](reference/engineering.md)**: Authoritative record of system invariants, architectural decisions, and historical post-mortems.
* **[Research Milestones](reference/next.md)**: Strategic implementation roadmap and future research horizons.
