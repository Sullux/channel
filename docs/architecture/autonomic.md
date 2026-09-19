# Autonomic Finite State Machine (A-FSM)

## 1. Executive Summary & Biological Analogy

Contemporary Large Language Model (LLM) agent frameworks operate under a flawed assumption: that every decision, transition, and evaluation must pass through the model's deliberative, conscious reasoning channel. In standard agentic loops (e.g. ReAct, LangChain, AutoGPT), an LLM is prompted with a monolithic textual context and asked to produce structured JSON or code to decide its next step. This design introduces multi-second round-trip latencies, inflates context windows with ephemeral clutter, introduces severe non-determinism, and induces reasoning fatigue.

In biological nervous systems, the brain does not operate as a single monolithic deliberative loop. Instead, it is partitioned into two distinct systems:
1. **The Autonomic Nervous System (Brainstem & Reflex Arcs):** Subconscious, millisecond-level feedback loops that regulate pupil dilation, cardiac rhythm, swallowing reflexes, and motor coordination. The autonomic system acts on sensory input rapidly and deterministically without conscious intervention.
2. **The Cerebral Cortex (Conscious Deliberation):** Slow, high-energy, deliberative thought used for novel problem-solving, strategic planning, and reflective reasoning.

The **Autonomic Finite State Machine (A-FSM)** equips the **Channel Inference Engine** and host orchestrator with an instinctual nervous system. By executing transient, constrained logit probes directly on the model's latent activations—and immediately rolling back the KV clock—Channel evaluates transitions, routes events, gates thinking channels, and steers progressive stream ingestion in sub-2ms intervals without polluting the KV cache. The deliberative cortex is engaged only when genuine reasoning is required.

---

## 2. Division of Labor: Brainstem Engine vs. Host Orchestrator

To maintain clean separation of concerns and avoid bloated software design:

```
┌─────────────────────────────────────────────────────────────┐
│                       HOST (Node.js)                        │
│                                                             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │    A-FSM Interpreter    │   │   Persistent Context    │  │
│  │  (State Graph & Guards) │◄──┤ (Tasks, Notes, Cursors) │  │
│  └────────────┬────────────┘   └─────────────────────────┘  │
│               │                                             │
│               │ client.probe(prompt, candDigits, 1)         │
│               ▼                                             │
├───────────────┼─────────────────────────────────────────────┤
│  Unix Socket  │ Binary Wire Protocol (OP_PROBE_AUTONOMIC)   │
├───────────────┼─────────────────────────────────────────────┤
│               ▼                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │             CHANNEL BRAINSTEM ENGINE (Zig / GPU)      │  │
│  │                                                       │  │
│  │  • Physical KV Ring Buffer & RoPE (4096 slots)        │  │
│  │  • Fast Constrained Logit Softmax & Entropy           │  │
│  │  • O(1) rollbackClock() Non-Destructive Invariant     │  │
│  │  • Outer TemplateState Turn Conformance               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

1. **The Channel Brainstem Engine (Zig / Vulkan GPU Compute):**
   * High-performance tensor execution, physical ring buffer geometry, UMA memory synchronization, and canonical chat template framing.
   * Exposes a single, universal reflex primitive over the wire: `OP_PROBE_AUTONOMIC` (`client.probe()`).
   * Evaluates 1-token constrained probes in ~1.5 ms and micro-decodes in ~200–400 ms.
   * Strictly agnostic to application state machines, YAML parsing, and task tree topologies.
2. **The Host Orchestrator (Node.js / TUI):**
   * Evaluates the A-FSM state graph, manages persistent context dictionaries, executes reducers/ops, and binds I/O streams.
   * Compiles state machines in pure JavaScript closures (< 1 ms).
   * Drives progressive reading and soft-yield check-in loops.

---

## 3. Disambiguation: Operations (`op`) vs. Tools (`tool`)

Overloading the term "tool" causes severe cognitive confusion across documentation, prompt design, and code. The A-FSM establishes a strict architectural boundary:

* **Tools (`<|tool_call>`)**: Heavyweight, generative operations. The LLM must enter an explicit tool channel, generate structured JSON arguments, and wait for execution. Tools take hundreds of tokens and hundreds of milliseconds of decode time.
* **Operations (`op`)**: Lightweight state transitions and micro-actions executed deterministically by the A-FSM host (`yield`, `reflect`, `resume`, `ask_user`, `emit`, `note`, `recall`, `edit_cell`). They execute in sub-millisecond time without generative overhead.

---

## 4. Mathematical Foundations of `probeAutonomic`

All autonomic decisions are grounded in transient logit evaluation directly from the model's KV attention state.

```
+-------------------------------------------------------------+
|                     probeAutonomic                          |
+-------------------------------------------------------------+
| 1. Record snapshot clock: saved_clock = engine.clock        |
| 2. Wrap prompt in canonical template (formatProbeFrame)     |
| 3. Prefill probe prompt (transient slots)                   |
| 4. If discrete (max_decode_tokens == 1):                    |
|    a. Mask logits to candidate token set                    |
|    b. Compute candidate softmax probabilities and entropy   |
|    c. Select winning candidate via argmax / sample          |
| 5. If generative (max_decode_tokens > 1):                   |
|    a. Autoregressively decode up to max_decode_tokens       |
|    b. Extract decoded text slice                            |
| 6. Rollback KV ring buffer: rollbackClock(saved_clock)      |
| 7. Return winning index, token, confidence, entropy, cost   |
+-------------------------------------------------------------+
```

### 4.1 Normalized Probability (Confidence) & Shannon Entropy
Given unnormalized logits $z_{c_i}$ for each candidate token $c_i \in C$ at temperature $T$:

$$P(c_i \mid C) = \frac{\exp(z_{c_i} / T)}{\sum_{j=1}^K \exp(z_{c_j} / T)}, \quad \text{confidence} = \max_i P(c_i \mid C)$$

$$H(C) = - \sum_{i=1}^K P(c_i \mid C) \ln P(c_i \mid C)$$

* **Low Entropy ($H \to 0$):** Clear consensus; the model has an unambiguous latent bias toward one transition.
* **High Entropy ($H \to \ln K$):** Ambivalence; triggers fallback cascades or conscious deliberation.

---

## 5. Applied Reflexes in Channel

### 5.1 The Thinking Gate
Before decoding token 1 of a response, Channel executes an autonomic probe to assess whether deliberative reasoning is required:
* `[0: Direct response, 1: Deliberative reasoning]`
* If `0` wins with confidence $\ge 0.80$, Channel immediately commits closed thought tokens (`<|channel>thought\n<channel|>`) and suppresses token 100, streaming the direct response instantly.
* If uncertainty or complexity is high, the deliberate reasoning pass is engaged.

### 5.2 Adaptive Thinking Depth
During active deliberate reasoning, at each elastic syntactic soft-yield (`STOP_ELASTIC_YIELD`), Channel assesses whether reasoning has reached sufficient resolution:
* `[0: Sufficient, 1: Deliberation needed]`
* If confidence $\ge 0.80$ for `0`, Channel issues `OP_RESUME` with `RESUME_ACTION_CLOSE_THOUGHT`, terminating thought generation early and transitioning to response output, saving significant compute and latency.

### 5.3 Channel Focus Arbitration
When background streams (terminal output, subshell execution, timers) produce activity, the A-FSM evaluates whether the incoming signal warrants shifting the active channel focus or should remain background telemetry, eliminating synthetic task creation during normal conversational dialogue.
