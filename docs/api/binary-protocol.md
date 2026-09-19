# Channel Binary Wire Protocol

## 1. Overview & Transport Model

The **Channel Inference Engine** provides a native, low-overhead binary wire protocol designed for high-throughput, low-latency, full-duplex communication over **Standard Input / Standard Output (STDIN / STDOUT)** when launched in server mode:

```bash
./zig-out/bin/channel --serve [OPTIONS]
```

### Core Design Principles:
* **Pure Opcode Protocol (Zero Header Flags):** Every event, channel stream, and boundary condition is represented by a discrete opcode. Host dispatch is a simple, direct `switch (frame.opcode)` with zero bitmask checks.
* **Native Multimodal Streaming:** Supports text, discrete tokens, continuous soft embeddings, raw/encoded images, audio waveforms, and video frames.
* **Stream-Injection Memory Model:** Memory queries act as in-engine stream injection side-effects; the host receives lightweight telemetry badges rather than raw vector records.
* **Transport:** Standard POSIX anonymous pipes (STDIN / STDOUT) with lifecycle bound directly to the parent process.
* **Byte Ordering:** Little-Endian for all multi-byte integers and IEEE 754 floating-point values.
* **Fixed 16-Byte Envelope:** Every frame begins with a uniform 16-byte header followed by an optional payload.
* **Full-Duplex:** Host $\to$ Engine (Inbound) and Engine $\to$ Host (Outbound) frames operate concurrently and asynchronously without turn-taking locks.

---

## 2. Binary Message Frame Envelope

Every message transmitted in either direction begins with a fixed **16-byte Header**:

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
|                        Payload Data...                        |
|                            (...)                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Field Definitions:

| Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..3` | **`magic`** | `u32` | Protocol magic constant: `0x53554C58` (ASCII `'SULX'`). |
| `4..5` | **`version`** | `u16` | Protocol version (`0x0001`). |
| `6..7` | **`msg_id`** | `u16` | Correlation ID for matching requests and responses (or `0` for unsolicited events). |
| `8..9` | **`opcode`** | `u16` | Operation code specifying the exact message type and payload structure. |
| `10..11`| **`reserved`** | `u16` | Reserved for 32-bit alignment padding (set to `0x0000`). |
| `12..15`| **`payload_len`**| `u32` | Length in bytes of the payload immediately following the header ($0 \le N \le 67,108,864$). |

---

## 3. Protocol Opcode Catalog

### Inbound (Host $\to$ Engine) Opcodes: `0x0001 .. 0x00FF`

| Opcode | Name | Description |
| :--- | :--- | :--- |
| `0x0001` | **`OP_STREAM_INPUT`** | Stream text, tokens, continuous soft vectors, audio PCM, images, or video frames into Layer 0. |
| `0x0002` | **`OP_ABORT`** | Administrative emergency brake: halts decoding immediately; preserves active state as `is_interrupted`. |
| `0x0003` | **`OP_MEM_QUERY`** | Explicit memory search (`keywords`, `fulltext`, temporal range, or pagination cursor). |
| `0x0004` | **`OP_SET_CONFIG`** | Configure runtime parameters (thinking budget, temperature, quiescence threshold, stop tokens). |
| `0x0005` | **`OP_TOOL_RETURN`** | Return tool execution result back into the model stream. |
| `0x0006` | **`OP_MEM_COMMIT`** | Force immediate consolidation of staging buffer to NVMe storage. |
| `0x0007` | **`OP_SET_SYSTEM`** | Initialize and prefill session system prompt with instructions and abstract tool definitions (JSON). |
| `0x0008` | **`OP_SNAPSHOT_SAVE`** | Request atomic working state checkpoint save (`path`, `stream_id`). |
| `0x0009` | **`OP_SNAPSHOT_LOAD`** | Request working state restore from snapshot (`path`). |
| `0x000A` | **`OP_RESUME`** | Continue decoding from an elastic yield (`STOP_ELASTIC_YIELD`) without new turn prefill. |
| `0x000E` | **`OP_PING`** | Keepalive / round-trip latency probe. |
| `0x000F` | **`OP_SHUTDOWN`** | Gracefully flush stores, release GPU memory, and exit. |
| `0x0012` | **`OP_PROBE_AUTONOMIC`** | Execute transient constrained logit probe or generative micro-decode with non-destructive clock rollback. |

---

### Outbound (Engine $\to$ Host) Opcodes: `0x0101 .. 0x01FF`

| Opcode | Name | Description |
| :--- | :--- | :--- |
| `0x0101` | **`OP_STREAM_CONTENT`** | Generated conversational / assistant text token or discrete multimedia output token. |
| `0x0102` | **`OP_STREAM_THOUGHT`** | Internal reasoning / thought channel token (`<channel>thought`). |
| `0x0103` | **`OP_TURN_COMPLETE`** | Signals end of turn (`<turn|>`), returning token count, duration, and average tok/s. |
| `0x0104` | **`OP_TOOL_CALL`** | Model-generated tool call request (tool name + JSON arguments). |
| `0x0105` | **`OP_MEM_RESPONSE`** | Results of an `OP_MEM_QUERY` returning injected episode counts, timestamps, and cursor. |
| `0x0106` | **`OP_STATUS`** | Live engine telemetry (tok/s, active vs quiescent layer breakdown, ring slots, VRAM). |
| `0x0107` | **`OP_SNAPSHOT_STATUS`** | Working state snapshot status (status code, clock, active slots, stream anchor ID). |
| `0x010C` | **`OP_AUTONOMIC_RESULT`** | Autonomic probe result (winning index, confidence f32, entropy f32, cost ms f32, decoded text). |
| `0x010E` | **`OP_PONG`** | Reply to `OP_PING`. |
| `0x01FF` | **`OP_ERROR`** | Structured error notification. |

---

## 4. Detailed Payload Specifications

### 4.1. `OP_STREAM_INPUT` (`0x0001`) — Inbound Multimodal Stream
Ingests text, tokens, soft vectors, audio PCM, images, or video frames into Layer 0.

* **Payload Layout:**
  - `data_type: u8`:
    - `0x00`: Raw UTF-8 text string
    - `0x01`: Token IDs (`[]u32` little-endian)
    - `0x02`: Soft prompt embeddings (`[][hidden_dim]f16`)
    - `0x03`: Audio PCM samples (`16kHz 16-bit mono`)
    - `0x04`: Encoded Image (`JPEG/PNG`)
    - `0x05`: Raw RGB/RGBA patch tensors
  - `flags: u8`:
    - `0x01`: `INPUT_FLAG_DIRECT` (bypasses outer chat turn wrappers)
    - `0x02`: `INPUT_FLAG_REASON` (explicitly requests deliberate reasoning)
  - `data: [payload_len - 2]u8`

### 4.2. `OP_PROBE_AUTONOMIC` (`0x0012`) — Autonomic Reflex Probe
Executes a transient constrained logit probe or generative micro-decode directly on the model's KV attention state, followed by $O(1)$ clock rollback.

* **Payload Layout:**
  - `max_decode_tokens: u16`: Number of tokens to decode (`1` for discrete logit probe; `> 1` for micro-decode).
  - `num_candidates: u16`: Number of candidate token IDs provided in `candidates`.
  - `candidates: [num_candidates]u32`: Array of token IDs to mask logits over.
  - `prompt_len: u32`: Length of prompt text in bytes.
  - `prompt: [prompt_len]u8`: Raw prompt text.

### 4.3. `OP_AUTONOMIC_RESULT` (`0x010C`) — Autonomic Probe Result
* **Payload Layout:**
  - `selected_index: u16`: 0-based index of the winning token in `candidates` (or `0xFFFF` for generative).
  - `winning_token: u32`: Token ID selected by sampler / argmax.
  - `confidence: f32`: Softmax probability of winning candidate ($0.0 \dots 1.0$).
  - `entropy: f32`: Normalized Shannon entropy over the candidate distribution ($0.0 \dots \ln K$).
  - `cost_ms: f32`: Elapsed evaluation latency in milliseconds on GPU.
  - `text_len: u16`: Length in bytes of decoded text (if `max_decode_tokens > 1`).
  - `text: [text_len]u8`: Decoded UTF-8 string slice.

### 4.4. `OP_RESUME` (`0x000A`) — Elastic Yield Continuation
Resumes autoregressive generation after an elastic soft yield (`STOP_ELASTIC_YIELD`).

* **Payload Layout:**
  - Optional `action: u8`:
    - `0x00`: `RESUME_ACTION_CONTINUE` (resumes token generation along active trajectory)
    - `0x01`: `RESUME_ACTION_CLOSE_THOUGHT` (caps thinking channel, advances `<channel|>`, and transitions immediately to response generation)

### 4.5. `OP_TOOL_RETURN` (`0x0005`) — Binary Tool Return
Delivers structured tool execution output back into the engine.

* **Payload Layout:**
  - `call_id_len: u16`: Length of tool call ID string.
  - `call_id: [call_id_len]u8`: Identifier of tool call being answered.
  - `is_error: u8`: `0` for success, `1` for execution failure.
  - `result_len: u32`: Length of result text.
  - `result: [result_len]u8`: Text or JSON output of tool execution.

### 4.6. `OP_SET_CONFIG` (`0x0004`) — Runtime Configuration (41 Bytes)
* **Payload Layout:**
  - `temperature: f32` (default: `0.7`)
  - `top_p: f32` (default: `0.95`)
  - `min_p: f32` (default: `0.05`)
  - `top_k: u32` (default: `64`)
  - `max_tokens: u32` (safety runaway circuit breaker, default: `4096`)
  - `repeat_penalty: f32` (default: `1.10`)
  - `repeat_window: u32` (default: `64`)
  - `quiescence_threshold: f32` (layer diff gating threshold, default: `0.02`)
  - `thinking_budget: u32` (maximum thinking tokens, default: `512`)
  - `flags: u8`:
    - Bit 0: `enable_critique`
    - Bit 1: `enable_quiescence`
    - Bit 2: `enable_memory_injection`
    - Bit 3: `enable_thinking_gate`
