# Physical KV Ring Buffer Geometry

## 1. Unified 4,096-Slot Architecture

A fundamental failure mode of traditional LLM inference engines is dynamic KV memory reallocation. As sequences grow, the engine continuously resizes and fragments VRAM allocations, culminating in out-of-memory crashes or catastrophic turn snapping.

The **Channel Inference Engine** enforces a strictly bounded, uniform **4,096 physical slot geometry** across all 48 transformer layers. 

```
┌────────────────────────────────────────────────────────────────────────┐
│ Total Fixed Allocation: 4,096 Slots per Layer                          │
├────────────────────────┬────────────────────────────────┬──────────────┤
│ Tier 1: Anchors        │ Tier 2: Sliding FIFO Ring      │ Tier 3:      │
│ (System Prompt & Tools)│ (Continuous Working Dialogue)  │ Recall Slots │
│ [Slots 0 .. N_sys-1]   │ [Slots N_sys .. 3967]          │ [3968..4095] │
└────────────────────────┴────────────────────────────────┴──────────────┘
```

---

## 2. Deterministic 3-Tier Partitioning

### Tier 1: Dynamic System Anchors (`0 .. N_sys - 1`)
* On cold boot, the system prompt and YAML frontmatter tool definitions are prefilled into slots $0 \dots N_{\text{sys}}-1$.
* These slots are marked immutable and permanently locked.
* Even when working context wraps millions of tokens, the model never forgets its core instructions, persona, or tool contracts.

### Tier 2: Dynamic FIFO Working Window (`N_sys .. 3967`)
* Working dialogue and sensory stream tokens advance sequentially within this sliding window.
* When the write pointer approaches slot 3968, it wraps circularly back to $N_{\text{sys}}$:
  $$\text{Physical Slot} = N_{\text{sys}} + \left((\text{clock} - N_{\text{sys}}) \pmod{3968 - N_{\text{sys}}}\right)$$
* The write pointer never overwrites Tier 1 anchors or intrudes into Tier 3 recall slots.

### Tier 3: Associative Recall Partition (`3968 .. 4095`)
* Exactly 128 slots reserved exclusively for episodic memory injection (`OP_MEM_QUERY`) and subconscious priming.
* Isolating recall to dedicated high slots prevents physical slot collisions between prefill prompts and recalled memories, completely eliminating KV cache clobbering.

---

## 3. GPU Indirection Table (`Active_slots`)

Rather than performing expensive memory shifts (`memmove`) when the ring buffer slides or wraps, the GPU attention shader accesses KV cache vectors through an indirect slot mapping table:

```zig
pub const ActiveSlots = struct {
    slots: [4096]u32,
    count: u32,
};
```

1. **Zero-Branch Attention:** The causal attention shader (`shaders/batch_causal_attn.wgsl` and `decode_attn.wgsl`) loops over `active_slots[i]` rather than linear indices:
   ```wgsl
   for (var i = 0u; i < num_active_slots; i++) {
       let physical_slot = active_slots[i];
       let k = kv_cache[layer][0][physical_slot];
       let v = kv_cache[layer][1][physical_slot];
       // dot product accumulation
   }
   ```
2. **$O(1)$ Ring Sliding:** Evicting old tokens or activating recalled episodes is a purely metadata operation on host integers, taking sub-microsecond CPU time with zero GPU VRAM writes.

---

## 4. Micro-Turn Semantic Boundaries (`SpanType`)

To prevent arbitrary token pruning that cuts words or sentences in half, Channel tracks semantic spans within the ring buffer:

```zig
pub const SpanType = enum(u8) {
    system = 0,
    user = 1,
    thought = 2,
    tool_call = 3,
    tool_result = 4,
    response_sentence = 5,
    turn_end = 6,
};

pub const SemanticSpan = struct {
    span_type: SpanType,
    start_clock: u64,
    end_clock: u64,
    start_slot: u32,
    slot_count: u32,
};
```

When memory pressure requires pruning old context, Channel evicts at **semantic boundaries** (e.g. an entire tool execution result or an entire thought sentence) rather than shearing arbitrary token offsets.

---

## 5. Non-Destructive Clock Rollback (`rollbackClock`)

For autonomic logit probes (`OP_PROBE_AUTONOMIC`), Channel temporarily advances the token clock and fills transient slots in the ring buffer. 

Upon probe completion:
```zig
pub fn rollbackClock(self: *DynamicRingBuffer, saved_clock: u64) void {
    while (self.clock > saved_clock) {
        self.clock -= 1;
        self.deactivateSlot(self.clockToSlot(self.clock));
    }
}
```
* **$O(1)$ Complexity:** Deactivates transient slots and restores `self.clock` instantaneously.
* **Context Isolation:** Leaves zero residual tokens, zero altered KV states, and zero garbage in the conversational context.
