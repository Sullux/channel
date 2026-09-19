# Client Implementation Guide

## 1. Overview & Architecture

The **Channel Inference Engine** is intentionally designed as an asynchronous, single-purpose **Tensor Brainstem**. It does not embed web frameworks, HTTP servers, or UI toolkits. Instead, it exposes a lean, full-duplex binary wire protocol over standard POSIX anonymous pipes (STDIN / STDOUT).

Any application—whether a cloud microservice, a command-line agent, an interactive desktop harness, or a real-time robotics controller—can integrate with Channel by spawning the engine binary with `--serve` and exchanging 16-byte binary frames:

```
┌────────────────────────────────────────────────────────────────────────┐
│ CUSTOM CLIENT APPLICATION (Go, Rust, Python, Node.js, C++)             │
├────────────────────────────────────────────────────────────────────────┤
│ • Child Process Lifecycle Manager (spawns channel --serve)             │
│ • Binary Frame Encoder & Decoder (16-byte fixed headers)               │
│ • Asynchronous Event Dispatcher (content, thoughts, status)            │
│ • State Machine / Cognitive Loop (A-FSM or custom agent logic)         │
└────────────────────────────────────────────────────────────────────────┘
                                 ▲ │
                     STDIN Pipe  │ │  STDOUT Pipe
                     (Inbound)   │ │  (Outbound)
                                 │ ▼
┌────────────────────────────────────────────────────────────────────────┐
│ CHANNEL INFERENCE ENGINE (Tensor Brainstem)                            │
│ • Physical 4,096-Slot KV Ring Buffer                                   │
│ • Autonomic 1-Token Reflex Probes (< 2ms)                              │
│ • Elastic Syntactic Yielding & Barge-In Recovery                       │
│ • Zero-Copy Working State Snapshots                                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Process Lifecycle & IPC Setup

### Spawning the Engine Process
To launch Channel as a subprocess from your application (for example, in Node.js, Go, or Python):

```bash
channel --model <path_to_model> --gpu --q4 --serve [OPTIONS]
```

### IPC Rules & Invariants
1. **Piped STDIN / STDOUT:** The engine reads incoming frames exclusively from STDIN and writes response frames exclusively to STDOUT.
2. **Diagnostic Logs to STDERR:** Human-readable logs, timing benchmarks, and debug traces are emitted strictly to STDERR to ensure binary frame integrity on STDOUT.
3. **Parent Process Binding:** Channel monitors its STDIN pipe for `EOF` (end-of-file). If the parent process crashes or terminates, Channel flushes its memory structures and exits cleanly, preventing zombie processes.

---

## 3. Frame Encoding & Protocol Dispatch

Every packet transmitted between host and engine starts with a **16-byte header**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Magic (0x53554C58)                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Version (1)          |           Msg ID              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Opcode             |          Reserved (0)         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Payload Length                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Implementing the Reader Loop
1. Read exactly 16 bytes to extract `magic`, `version`, `msg_id`, `opcode`, and `payload_len`.
2. Verify `magic == 0x53554C58` (`'SULX'`) and `version == 1`.
3. If `payload_len > 0`, read exactly `payload_len` bytes.
4. Dispatch to your event handler based on `opcode`.

For the full specification of all opcodes and payload structures, see the [Binary Wire Protocol](binary-protocol.md).

---

## 4. Driving the Elastic Transduction Lifecycle

In standard batch engines, generation is a black-box forward pass that cannot be interrupted. In Channel, generation yields at natural grammatical boundaries via **Elastic Syntactic Yielding**.

```
Host -> OP_STREAM_INPUT (User Prompt)
  <- OP_STREAM_THOUGHT (Thought tokens)
  <- OP_STREAM_CONTENT (Response tokens)
  <- OP_STATUS (reason = STOP_ELASTIC_YIELD)
```

When receiving `OP_STATUS` with reason `STOP_ELASTIC_YIELD` (`0x03`):

### Choice A: Uninterrupted Continuation
If no background events or user interrupts are pending, the client immediately issues `OP_RESUME`:
```
Host -> OP_RESUME (action = RESUME_ACTION_CONTINUE)
```
The engine advances from its last token and continues decoding without re-prefill.

### Choice B: Thought Channel Capping
If the client's autonomic probes indicate that sufficient reasoning has taken place, the client sends `OP_RESUME` with `RESUME_ACTION_CLOSE_THOUGHT` (`0x01`):
```
Host -> OP_RESUME (action = RESUME_ACTION_CLOSE_THOUGHT)
```
The engine commits token 101 (`<channel|>`), locks thinking suppression, and begins streaming response content immediately.

### Choice C: In-Flight User Barge-In
If a user submits new input while generation is yielding or decoding, the client delivers the barge-in via `OP_STREAM_INPUT`:
```
Host -> OP_STREAM_INPUT (payload = User Barge-in)
```
The engine detects that `turn_open` is active, automatically writes token 106 (`<turn|>`) to seal the assistant's partial turn, and immediately starts prefilling the user's interjection in canonical turn grammar.

---

## 5. Harnessing Sub-Millisecond Autonomic Probes

Clients can execute transient 1-token logit probes directly over the active KV cache using `OP_PROBE_AUTONOMIC` (`0x0012`):

```
Host -> OP_PROBE_AUTONOMIC {
  max_decode_tokens: 1,
  candidates: [token_id_0, token_id_1, token_id_2],
  prompt: "Evaluate next state..."
}
Engine -> OP_AUTONOMIC_RESULT {
  selected_index: 0,
  confidence: 0.942,
  entropy: 0.081,
  cost_ms: 1.42
}
```

* **Latency:** Executes on the GPU in **~1.5 milliseconds**.
* **Zero Pollution:** The engine executes an internal $O(1)$ clock rollback (`rollbackClock`), deactivating transient slots and restoring the exact KV state prior to the probe.
* **Mathematical Certainty:** The client uses `confidence` and `entropy` to make deterministic routing decisions without asking the LLM to output sentences or JSON.

---

## 6. Real-World Client Archetypes

### Archetype 1: Robotics & Physical AI Control Loop
A robotic agent needs continuous, high-frequency motor control and sensory arbitration without multi-second JSON parsing latency:
* **Discrete Action Grammars:** Define candidate token IDs corresponding to primitive motor actions: `[0: HALT, 1: FORWARD, 2: TURN_LEFT, 3: TURN_RIGHT, 4: REVERSE, 5: GRASP]`.
* **Sub-2ms Reflex Loop:** At 50 Hz, feed current LIDAR / sensor telemetry into an autonomic probe:
  ```
  Is immediate obstacle avoidance required? Options: 0: NO, 1: YES
  ```
* If high entropy or YES is detected, the host halts motor forward motion immediately, engaging the deliberate reasoning pass only when an obstacle requires path replanning.

### Archetype 2: Serverless Cloud Agent Microservices
In cloud environments, maintaining multi-gigabyte KV caches resident in VRAM for thousands of idle sessions is cost-prohibitive:
* **Instant Pause:** When a user turn finishes (`STOP_END_OF_TURN`), the client issues `OP_SNAPSHOT_SAVE`.
* **Zero-Copy Disk Flushing:** Channel serializes active slots to disk in ~5ms via 1MB buffered I/O.
* **Instant Wake-Up:** When the user reconnects 10 minutes later, the client spawns Channel and dispatches `OP_SNAPSHOT_LOAD`. Working memory is restored in **< 100 milliseconds**, bypassing the multi-second GPU prefill penalty.

### Archetype 3: Full-Duplex Conversational Voice Assistant
* **Continuous Audio Ingestion:** Stream raw audio PCM frames into Layer 0 via `OP_STREAM_INPUT`.
* **Speech Activity Detection (VAD):** As the assistant speaks, incoming user voice audio triggers an elastic yield.
* **Natural Interruption:** The model pauses mid-phrase, closes turn delimiters cleanly, and addresses the user's interruption seamlessly.

---

## 7. Reference Client Library

The Channel repository provides a clean, well-tested Node.js reference client implementation:
* **`tui/lib/protocol/framing.js`**: Pure binary frame encoders and parsers for all inbound and outbound opcodes.
* **`tui/lib/client/index.js`**: Promise-based client wrapper managing subprocess communication, event emitters, configuration, and autonomic probes (`client.probe()`).
