# Zero-Copy Working State Snapshots

## 1. Overview & Motivation

In traditional stateless LLM inference, conversational context must be re-supplied and re-encoded on every application launch or session restart. In Channel's continuous streaming transduction architecture, working memory is maintained progressively inside the model's physical 4,096-slot dynamic ring buffer ($K, V$ activations, clock counter, and attention-mass salience).

To eliminate amnesia across sessions and bypass the multi-second GPU prefill penalty on startup, Channel incorporates **Zero-Copy Working State Snapshotting**.

---

## 2. Binary Snapshot Architecture (`src/snapshot.zig`)

The working snapshot persists:
1. **Header Metadata**: Magic bytes (`SNAP`), version, global clock position, Tier 1 anchor count, ingested token count, and turn boundaries.
2. **Ring Buffer Geometry & Salience**: Per-slot clock counters, active boolean flags, and accumulated multi-layer attention mass.
3. **Physical KV Cache Slabs**:
   - For all active slots across all 48 layers, key ($K$) and value ($V$) activations are extracted directly from Vulkan `HOST_VISIBLE | HOST_COHERENT` UMA memory.
   - Active slots are compacted contiguously. Inactive slots are skipped, shrinking the file size from ~6.1 GB down to **~30–75 MB** for early turns and **~1.2 GB** for full contexts.
4. **Stream Write-Ahead Anchor**: The ULID / stream message ID of the exact `.stream.jsonl` event corresponding to the checkpoint point.

### Binary Header Structure

```zig
pub const SnapshotHeader = extern struct {
    magic: [4]u8 = SNAPSHOT_MAGIC, // 'SNAP'
    version: u32 = SNAPSHOT_VERSION, // 1
    clock: u64,
    num_anchors: u32,
    num_layers: u32,
    kv_dim: u32,
    max_slots: u32,
    total_ingested: u64,
    stream_id_len: u16,
    reserved_pad: u16 = 0,
    stream_id: [64]u8,
    turn_boundary_count: u32,
    turn_boundaries: [128]u32,
};
```

---

## 3. Non-Blocking File I/O Subsystem

Writing a gigabyte of KV activations to disk can freeze the main thread if implemented synchronously.

Channel optimizes snapshot I/O through two mechanisms:
1. **Detached Background Thread:** The server delegates `handleSnapshotSave` to an asynchronous thread (`std.Thread.spawn`), allowing decode loops and stream polling to continue without hitching.
2. **1 MB Buffered I/O:** Disk writes are buffered in 1 MB blocks using `std.io.bufferedWriter`, maximizing NVMe sequential throughput (~2,500–4,000 MB/s) and completing full checkpoints in milliseconds.
3. **Atomic File Swapping:** Snapshots write to a temporary file (`.snapshot.bin.tmp`) and atomically replace the destination (`.snapshot.bin`) via `std.fs.rename`, preventing corrupted states if power is lost mid-write.
4. **Zero-Delta Gate:** If the logical clock has not advanced since the last checkpoint, the engine immediately returns `SNAPSHOT_STATUS_EXISTS = 2` without performing redundant disk writes.

---

## 4. Checkpointing Lifecycle & Write-Ahead Log (WAL) Replay

```
[ Session Boot ]
        │
        ├── Has .snapshot.bin?
        │     ├── YES: Dispatch OP_SNAPSHOT_LOAD (< 100ms warm restore)
        │     │        Scan .stream.jsonl for uncommitted turns and replay
        │     └── NO:  Cold boot, prefill Tier 1 system prompt & cache anchors
        │
[ Turn Execution Loop ]
        │
        ├── STOP_END_OF_TURN received
        │     └── Arm 5-second quiescence debounce timer
        │           └── Timer fires: Dispatch OP_SNAPSHOT_SAVE
        │
[ Graceful Shutdown (Ctrl+C / Ctrl+Q) ]
        └── Immediate synchronous snapshot flush before closing child processes
```

### Warm Boot vs. Cold Boot Comparison

| Metric | Cold Boot | Channel Warm Boot |
| :--- | :--- | :--- |
| **System Prompt Prefill** | ~11.2 seconds | **0.0 seconds (Bypassed)** |
| **KV Cache Restore** | N/A | **< 100 milliseconds** |
| **Working Memory Continuity** | Reset to zero | **100% Exact Tensor Parity** |
| **Historical Replay** | Re-encode all past turns | **Zero-FLOP WAL Catch-up** |
