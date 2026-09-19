# Continuous Streaming Transduction

## 1. The Streaming Paradigm

Traditional Large Language Model architectures treat inference as a **monolithic document synthesis task**: a user submits a prompt, the engine prefills the entire sequence in one shot, and decodes an open-ended wall of text until hit by `<end_of_turn>`.

The **Channel Inference Engine** replaces this batch model with **Continuous Streaming Transduction**:
* Inputs stream into the engine in addressable 512-byte blocks.
* Generation executes in elastic micro-bursts, yielding execution at natural grammatical and semantic resting points via `STOP_ELASTIC_YIELD`.
* External systems (user keystrokes, terminal logs, subprocess output) can barge in mid-stream without destroying active computation or corrupting KV memory.

```
[ Incoming Sensory Streams: Chat, Subshell, Terminals ]
                       │
                       ▼ (512-Byte Pull Blocks)
┌─────────────────────────────────────────────────────────────┐
│ Channel Pull-Stream Ingestion Loop                          │
│ - Addressable file and pipe cursors                         │
│ - Non-blocking, incremental KV prefill                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Autoregressive Decode with Elastic Syntactic Yielding       │
│ - SyntaxTracker monitors punctuation and code fences        │
│ - Yields on punctuation (., ?, !) and line breaks           │
│ - Emits OP_STATUS (STOP_ELASTIC_YIELD)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   [ Unserviced Interrupts? ]  [ Clear Channel? ]
   ├── YES: Stage Barge-in     └── NO: Host sends OP_RESUME
   │   and Close Turn Envelopes    and generation continues
```

---

## 2. Elastic Syntactic Unit Gating (`src/server/syntax_tracker.zig`)

Arbitrary token yielding (e.g. slicing generation every 32 tokens) produces catastrophic cognitive degradation: models yield in the middle of variable names, arithmetic operators, or open string quotes.

Channel implements an **Elastic Syntactic Tracker** that identifies true semantic resting points:
1. **Punctuation Boundaries:** Yields on sentence-terminating punctuation (`.` = token 236761, `?` = 236881, `!` = 236888) and newline boundaries.
2. **Code Fence Protection:** Tracks open markdown backtick code blocks (```` ``` ````). Yielding is strictly prohibited inside unclosed code blocks to prevent syntax corruption.
3. **Delimiter Nesting:** Tracks unclosed parentheses, brackets, and quotes (`"`, `'`, `(`, `[`), ensuring the model yields only when its syntactic expression is balanced.

When a resting point is reached, the engine pauses generation and sends `OP_STATUS` with reason `STOP_ELASTIC_YIELD` (`0x03`), waiting for an `OP_RESUME` frame from the host.

---

## 3. In-Flight User Barge-In & Turn Envelope Recovery

In real-time interactive systems, user barge-ins and high-priority interrupts occur mid-stream. 

Traditional engines either discard in-flight output or interleave user text directly into the assistant's output channel, causing severe hallucination and token stutter.

Channel handles barge-ins cleanly:
1. **Server Gate Enforcement (`turn_open`):**
   - The engine tracks whether an assistant turn is actively open.
   - If the host delivers an interrupt turn (`OP_STREAM_INPUT`) while the assistant was mid-generation or yielding, the engine automatically commits token 106 (`<turn|>`), closing the active model turn cleanly.
   - The user barge-in is then ingested as a canonical `<|turn>user` turn, followed by `<|turn>model\n`, preserving valid Gemma 4 template grammar in the KV cache.
2. **Atomic Interjection Staging:**
   - In the host TUI and stream store, user barge-ins are staged in `pendingInterjection`.
   - When the active response yields, the assistant's partial thought and content are flushed first, followed immediately by the user interjection.
   - This guarantees strict chronological and causal monotonicity across UI cards and persistent `.stream.jsonl` logs.

---

## 4. Multi-Channel Addressable Pull-Streams

Instead of pushing arbitrary unstructured text into a single prompt pipe, Channel manages inputs through addressable **Channel Streams** (`tui/lib/channels/`):

* **`chat/user`:** Primary conversational interaction.
* **`cmd/stdout`:** Asynchronous subshell execution stdout/stderr streams.
* **`trm/<id>`:** Interactive terminal session buffers.

Each channel maintains an independent cursor pointing to an addressable VFS file. The engine pulls raw 512-byte slices incrementally, eliminating memory spikes and allowing the host A-FSM to dynamically arbitrate focus between background telemetry and user conversation.
