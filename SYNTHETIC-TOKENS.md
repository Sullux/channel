# Synthetic Token Architecture & Temporal Grounding

## 1. Executive Summary & Problem Space

In traditional Large Language Model (LLM) agent frameworks, contextual metadata—such as turn boundaries, system instructions, tool execution envelopes, and temporal memory markers—is injected into the model using natural language text or raw XML/Jinja pseudo-tokens:

```xml
<|turn>system
<memory timestamp="2026-10-14T08:30:00Z" age="days">
User stated that Natasha manages back office bookkeeping.
</memory><turn|>
```

This textual approach exhibits severe engineering and cognitive defects:
1. **Context Window Waste:** Serializing structural framing into multi-token ASCII text consumes dozens of valuable context slots per turn. In a strictly bounded 4,096-slot geometry, textual scaffolding creates unacceptable context pressure.
2. **Syntactic Fragility & Delimiter Hallucination:** Small and medium models frequently emit pseudo-delimiters (`<|turn>`, `<memory>`, `</memory>`) as raw content, causing token stutter, runaway formatting loops, and parse failures.
3. **Temporal Incomprehension:** Standard foundation models do not possess an intuitive sense of physical time. Formatting time as an ISO-8601 string or integer epoch timestamp requires the model to perform multi-step arithmetic over character tokens to deduce whether an event happened 5 minutes ago or 3 years ago.

**Channel** solves this by introducing **Empirically Calibrated Synthetic Tokens**:
- Structural delimiters and temporal horizons are represented as **dense mathematical embedding vectors** injected directly into Layer 0 via soft prompt embedding ingress (`OP_STREAM_INPUT`, `data_type: 0x02`).
- Temporal scales are grounded in **logarithmic orders of magnitude** reflecting human neurophysiological decay envelopes.
- Synthetic vectors are generated through the binary wire protocol (`OP_SYNTHESIZE_TOKEN`), calculated natively by the engine, and persisted as binary assets in the host Virtual File Subsystem (VFS).
- The syntax is **unidirectional and delimiter-free**: the appearance of a new typed start-token implicitly marks the closure of the preceding semantic span, completely eliminating closing tags.

---

## 2. Temporal Grounding: Logarithmic Horizon Scale

Human episodic memory does not index events by discrete microsecond timestamps. Instead, mammalian recall operates across logarithmic temporal decay envelopes.

Channel defines a discrete, exhaustive set of **Temporal Grounding Tokens**:

| Temporal Horizon | Semantic Scope | Cognitive Utility |
| :--- | :--- | :--- |
| `SYNTH_TIME_IMMEDIATE` | $0 \dots 10$ seconds | Immediate conversational echoic trace / in-flight barge-in. |
| `SYNTH_TIME_JUST` | $10 \dots 60$ seconds | The immediate prior conversational turn. |
| `SYNTH_TIME_SHORTLY` | $1 \dots 10$ minutes | The active task session or recent terminal command. |
| `SYNTH_TIME_RECENT` | $1 \dots 4$ hours | Events occurring earlier in the current work session ("this morning / afternoon"). |
| `SYNTH_TIME_TODAY` | $4 \dots 24$ hours | Consolidated events from the current day. |
| `SYNTH_TIME_THIS_WEEK` | $1 \dots 7$ days | Working memory of current sprint or weekly active projects. |
| `SYNTH_TIME_THIS_MONTH`| $1 \dots 4$ weeks | Medium-term historical records and milestones. |
| `SYNTH_TIME_THIS_YEAR` | $1 \dots 12$ months | Long-term episodic memory archive. |
| `SYNTH_TIME_YEARS_AGO` | $> 1$ year | Foundational identity, historical facts, and ancient priors. |

When an episodic memory is recalled from `.episodic.mem`, the host calculates $\Delta t = \text{Date.now}() - \text{episode.timestamp}$, maps $\Delta t$ to its corresponding logarithmic bucket, and prefixes the injected memory slab with the single pre-calibrated synthetic horizon vector.

---

## 3. The Empirical Synthesis Lifecycle (`OP_SYNTHESIZE_TOKEN`)

Synthetic tokens are **not** hand-tuned arbitrary numbers; they are **empirically distilled activation vectors** produced by the model itself during a clean calibration pass.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE MIND (Host)                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Identifies missing token asset in `.agent/tokens/`                   │
│ 2. Formulates calibration scenario prompt:                             │
│    "Context: A memory from this morning. Content: {ANCHOR}"           │
│ 3. Issues OP_SYNTHESIZE_TOKEN over binary socket                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Binary Wire Protocol
                                    │ (OP_SYNTHESIZE_TOKEN)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     CHANNEL BRAINSTEM (Engine)                         │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Prefills calibration prompt on isolated scratch context             │
│ 2. Evaluates forward pass through Layer 47                             │
│ 3. Extracts unit-normalized activation vector at {ANCHOR}              │
│ 4. Returns vector payload via OP_SYNTHESIZED_TOKEN_RESULT              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Binary Wire Protocol
                                    │ (OP_SYNTHESIZED_TOKEN_RESULT)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Host writes 8KB binary blob to `.agent/tokens/time_morning.bin`     │
│ 5. Token is cached permanently for the lifetime of the agent instance  │
└────────────────────────────────────────────────────────────────────────┘
```

### Protocol Specification: `OP_SYNTHESIZE_TOKEN` (`0x0013`)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Magic (0x53554C58)                       │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Version (1)          |           Msg ID              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       OP_SYNTHESIZE_TOKEN     |          Reserved (0)         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Payload Length                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  target_layer | extract_stage |          reserved             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        anchor_token_idx       |          prompt_len           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Prompt UTF-8 ...                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

* **`target_layer` (`u8`):** Layer to extract activation from (typically `0` for input soft embedding, or `N-1` for high-level semantic representation).
* **`extract_stage` (`u8`):** Extraction stage (`0x00` = pre-RMSNorm hidden state, `0x01` = post-attention residual, `0x02` = final unembedding).
* **`anchor_token_idx` (`u16`):** Token offset within the prompt representing the target semantic concept.
* **`prompt` (`[]u8`):** Full calibration scenario string.

### Outbound Response: `OP_SYNTHESIZED_TOKEN_RESULT` (`0x010D`)
* Payload returns:
  - `vector_dim: u32` (e.g. `3840` or `4096`)
  - `data_format: u8` (`0x01` = FP16, `0x02` = BF16, `0x03` = FP32)
  - `vector_data: [vector_dim * sizeof(format)]u8`

---

## 4. Ingress Mechanics: Layer 0 Soft Prompt Embedding

When injecting synthetic tokens into active cognition (such as during memory rehydration or task-switching), the client delivers them via `OP_STREAM_INPUT` with `data_type: 0x02` (`Soft Prompt Embeddings`).

### Why Layer 0 Soft Embeddings Outperform Multi-Layer Slices
1. **Mathematical Cleanliness:** Injecting a raw multi-layer KV slab requires adjusting 48 separate layers of keys and values, risking cross-layer representational drift. A soft embedding vector at Layer 0 enters the model through the front door: it naturally propagates up through all 48 transformer layers, allowing every attention head and MLP to condition on it natively.
2. **Zero Positional Phase Decoherence:** Because the vector enters at Layer 0 alongside incoming tokens, the engine's RoPE rotary embedding matrices ($\mathcal{R}_{\Theta}$) are calculated in continuous lockstep with the active token clock.
3. **Sub-Microsecond Ingress:** The soft vector is copied directly into the input staging buffer in a single contiguous memory copy (`memcpy`), bypassing tokenizer token-to-id dictionaries entirely.

---

## 5. Unidirectional Framing (Elimination of End Tokens)

Traditional systems require explicit opening and closing tags:
```xml
<memory>...</memory><user>...</user>
```
If the model or stream truncates before emitting `</memory>`, the entire subsequent context is corrupted because the parser believes it is still inside the memory container.

Channel implements **Implicit Span Closure via Unidirectional Start Tokens**:

```
[SYNTH_START_SYS]   ==> (Anchors: System identity & tool contracts)
[SYNTH_START_MEM]   ==> (Temporal Marker) ==> (Recalled Episode Slabs)
[SYNTH_START_USER]  ==> (Active conversational input from chat/user)
[SYNTH_START_CMD]   ==> (stdout stream slice from subshell)
[SYNTH_START_MODEL] ==> (Autoregressive generation stream)
```

**System Invariant:** The arrival of any typed synthetic start-token **atomically and unambiguously terminates the preceding span**. 
- No closing tags (`</turn>`, `</memory>`, `<turn|>`) are ever required.
- The GPU ring buffer marks a new `SemanticSpan` record in `src/ring_buffer.zig` at the exact slot offset of the start-token, locking boundary salience deterministically.

---

## 6. Model Family Configuration (`config/<model>.yaml`)

While an individual agent instance (e.g. `~/agents/hopper`) is permanently bound to an immutable model weight signature, Channel maintains architectural agility across different foundation model families through declarative configuration manifests:

```yaml
model_family: "gemma-4"
variant: "12B-Unified"
architecture:
  num_layers: 48
  hidden_dim: 3840
  kv_dim: 1024
  num_heads: 32
  num_kv_heads: 8
  head_dim: 128
  sliding_window: 1024
  sliding_layers: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, ... ] # 40 layers
  global_layers: [8, 16, 24, 32, 40, 44, 46, 47]       # 8 layers
synthetic_tokens:
  embedding_dim: 3840
  format: "bf16"
  required_assets:
    - name: "time_immediate"
      anchor: "{NOW}"
      scenario: "Sensory event occurring right now: {NOW}."
    - name: "time_shortly"
      anchor: "{PAST}"
      scenario: "Context: A task executed a few minutes ago: {PAST}."
    - name: "time_today"
      anchor: "{TODAY}"
      scenario: "Context: Historical interaction from earlier today: {TODAY}."
    - name: "time_years_ago"
      anchor: "{ANCIENT}"
      scenario: "Foundational memory established years ago: {ANCIENT}."
```

On cold boot, the host reads `config/<model>.yaml`, checks whether `.agent/tokens/*.bin` exist, generates any missing synthetic tokens via `OP_SYNTHESIZE_TOKEN`, and caches them permanently. Subsequent runs warm-boot instantaneously from disk.
