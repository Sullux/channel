# Channel Inference Engine

A high-performance continuous streaming inference engine with autonomic reflexes, written in pure Zig for Google's Gemma 4 models (`gemma-4-E2B` and `gemma-4-12B-it`).

**Channel** is hardware-tailored for the **AMD Ryzen AI Max+ 395 (Radeon 890M/800-series / RDNA 3.5)** integrated GPU on unified memory architecture (UMA) via Vulkan 1.3 compute. It models LLM inference as a continuous, interruptible sensory stream with sub-millisecond autonomic reflexes, zero-copy working state snapshots, and high-speed episodic associative recall.

---

## Documentation

Full architectural specifications, foundations, and API references are organized in the [`docs/`](docs/) directory:

* **[Quick Start Guide](docs/quick-start.md)**: Fast track installation, prerequisites, and running the reference TUI or CLI in minutes.

### Foundations
* **[The Problem Space](docs/foundations/problem.md)**: Four fundamental bottlenecks of modern LLMs (streaming, learning, memory, and the multi-second tool-use tax).
* **[Model Selection](docs/foundations/model-selection.md)**: Comparative analysis of Gemma 4 variants and why 12B Unified was selected.
* **[Approach & Philosophy](docs/foundations/approach.md)**: Core design principles—fixed 4,096-slot geometry, autonomic reflexes, and continuous transduction.

### Architecture
* **[Architecture Hub](docs/architecture/README.md)**: Executive overview of the Dual-Plane system (Tensor Brainstem vs Cognitive Mind).
* **[Autonomic Reflex Plane](docs/architecture/autonomic.md)**: 1-token discrete logit probes, Shannon entropy, Thinking Gate, and A-FSM focus arbitration.
* **[Episodic Memory & Recall](docs/architecture/memory.md)**: Dual-process cognitive memory, UMA zero-copy slabs, and 2D Givens RoPE delta rotation.
* **[Physical KV Ring Buffer](docs/architecture/ring-buffer.md)**: Fixed 4,096-slot geometry, Tier 1 anchors, sliding FIFO ring, and micro-turn semantic boundaries.
* **[Streaming Transduction](docs/architecture/streaming.md)**: Continuous 512-byte pull-streams, elastic syntactic unit gating (`STOP_ELASTIC_YIELD`), and in-flight barge-in recovery.
* **[Zero-Copy Snapshots](docs/architecture/snapshots.md)**: Compacted working state snapshots, non-blocking 1MB buffered I/O, and warm boot recovery.
* **[Reference TUI Architecture](docs/architecture/tui.md)**: Production-grade reference agent implementation showcasing zero-markup YAML controllers, Markdown AST rendering, VFS storage, and LIFO interrupts.

### API & Client Integration
* **[Command Line Interface (CLI)](docs/api/cli.md)**: Comprehensive CLI reference, operational modes, and runtime flags.
* **[Binary Wire Protocol](docs/api/binary-protocol.md)**: 16-byte fixed framing, opcodes `0x0001`–`0x010C`, UMA sync, and full-duplex communication over STDIN/STDOUT.
* **[Client Implementation Guide](docs/api/client-guide.md)**: Architectural guide for building custom clients (robotics, cloud microservices, voice agents) on Channel.

### Reference

---

## CLI Reference

### Synopsis

```bash
./zig-out/bin/infer [OPTIONS] [PROMPT...]
```

If no prompt arguments are supplied on the command line, Channel enters an interactive, continuous prompt session where conversation context is maintained across multiple turns.

### Options & Switches

| Switch | Arguments | Default | Description |
| :--- | :--- | :--- | :--- |
| `-m`, `--model` | `<path>` | `../gemma-4-E2B` | Path to the model directory containing `config.json`, `tokenizer.json`, and `model.safetensors`. |
| `-n`, `--max-tokens` | `<N>` | `128` | Maximum number of new tokens to generate for the response (runaway safety ceiling). |
| `--gpu` | — | Disabled | Enables Vulkan 1.3 GPU compute dispatch in unquantized 16-bit bfloat16 (`BF16`). |
| `--q4`, `--mixed` | — | Disabled | Enables GPU compute acceleration with pure symmetric zero-centered **Q4_0** linear projections and **Q8_0** output classification. |
| `--q8` | — | Disabled | Enables GPU compute acceleration with on-the-fly **Q8_0** (8-bit signed integer) weight dequantization across all layers. |
| `--quant` | `<q4\|q8\|none>` | `none` | Explicitly selects the GPU weight quantization mode. |
| `--serve` | — | Disabled | Runs engine as a headless full-duplex binary wire-protocol server over STDIN/STDOUT for TUI or driver clients. |
| `--bench` | — | Disabled | Executes single-batch GPU compute throughput and latency benchmarking. |
| `--anchors` | `<N>` | `32` | Number of immutable early context anchor slots reserved at the start of the ring buffer. |
| `--window` | `<N>` | `512` | Number of rolling active sliding-window slots in the dynamic ring buffer. |
| `--recall` | `<N>` | `96` | Number of dynamic recall slots injected by the associative long-term memory subsystem (up to 128 max). |
| `--memory`, `--storage` | `[<path>]` | `.episodic.mem` (if flag present) | Enables persistent memory-mapped episodic store (`.episodic.mem`). Cross-session latent memory persistence. |
| `--mem-capacity` | `<N>` | `64` | Maximum capacity of stored episodic entries in `.episodic.mem` (e.g. 64 episodes = 4,096 tokens). |
| `--no-memory` | — | Enabled | Disables associative long-term memory ingestion and recall injection. |
| `--quiescence` | — | Disabled | Enables hierarchical multi-scale temporal quiescence gating to skip upper transformer layers during low activation velocity. |
| `--quiescence-threshold` | `<float>` | `0.001` | Sets the cosine similarity activation velocity threshold for quiescence skipping. |
| `-h`, `--help` | — | — | Prints command-line usage instructions and available switches, then exits. |

---

## Usage Examples

### 1. Interactive 12B GPU Chat (Q4_0 Acceleration)
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --q4
```

### 2. Single-Prompt 12B Execution with Gemma 4 Chat Formatting
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --q4 --prompt "<|turn>user
What is the capital of France?<turn|>
<|turn>model
" --max-tokens 50
```

### 3. Persistent Long-Term Associative Memory Session
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --gpu --q4 --memory .memory/.episodic.mem --mem-capacity 64
```

### 4. Full-Duplex Binary Wire Protocol Server for TUI / Clients
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --gpu --q4 --serve --memory .memory/.episodic.mem
```

### 5. Running CPU Inference on Gemma 4 E2B
```bash
./zig-out/bin/infer --model ../gemma-4-E2B --max-tokens 30 "The capital of France is"
```

---

## Development & Testing

### Building the Binary
```bash
# Debug build
zig build

# ReleaseFast build for maximum GPU throughput
zig build -Doptimize=ReleaseFast
```

### Running the Test Suites

```bash
# Run native Zig engine unit tests (35/35 passing)
zig test src/main.zig

# Run TUI client test suite (98/98 passing)
yarn --cwd tui test
```
