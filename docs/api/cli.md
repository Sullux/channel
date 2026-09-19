# Command Line Interface (CLI)

The **Channel Inference Engine** compiles to a single, standalone binary (`infer` or `channel`) with zero dynamic runtime dependencies beyond the system's Vulkan driver and C runtime.

---

## 1. Synopsis

```bash
channel [OPTIONS] [PROMPT...]
```

* **Interactive Mode:** If no prompt argument is provided and `--serve` is not specified, Channel enters an interactive terminal chat session where context is maintained sequentially.
* **Single-Prompt Mode:** If a prompt string is supplied (or `--prompt` is used), Channel processes the prompt, streams the output to STDOUT, and exits.
* **Server Mode:** If `--serve` is specified, Channel runs as a headless binary wire-protocol daemon over STDIN/STDOUT.

---

## 2. Command-Line Options Reference

### Model & Hardware Precision

| Option | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `-m`, `--model` | `<path>` | `../gemma-4-E2B` | Path to the model directory containing `config.json`, `tokenizer.json`, and `model.safetensors`. |
| `--gpu` | — | Disabled | Enables Vulkan 1.3 GPU compute dispatch in unquantized 16-bit bfloat16 (`BF16`). |
| `--q4`, `--mixed` | — | Disabled | Enables GPU acceleration with pure symmetric zero-centered **Q4_0** linear projections and **Q8_0** output classification. Matches official Google QAT weights. |
| `--q8` | — | Disabled | Enables GPU acceleration with on-the-fly **Q8_0** (8-bit signed integer) weight dequantization across all layers. |
| `--quant` | `<q4\|q8\|none>` | `none` | Explicitly selects the GPU weight quantization mode. |

### Generation & Token Constraints

| Option | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p`, `--prompt` | `<string>` | `null` | Initial prompt text to evaluate. |
| `-n`, `--max-tokens` | `<N>` | `128` | Maximum number of new tokens to generate. Serves as a safety runaway circuit-breaker. |
| `--anchors` | `<N>` | `32` | Number of immutable early context anchor slots reserved at the start of the physical ring buffer. |
| `--window` | `<N>` | `512` | Number of rolling active sliding-window slots in the dynamic ring buffer. |
| `--recall` | `<N>` | `96` | Number of dynamic recall slots reserved for episodic memory injection. |

### Memory & Persistence Subsystem

| Option | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `--memory`, `--storage` | `[<path>]` | `.episodic.mem` | Enables the persistent memory-mapped episodic store. Cross-session latent memory persistence. |
| `--mem-capacity` | `<N>` | `64` | Maximum capacity of stored episodic entries in `.episodic.mem` (e.g. 64 episodes $\approx$ 4,096 tokens). |
| `--no-memory` | — | Enabled | Disables associative long-term memory ingestion and recall injection. |

### Execution Modes

| Option | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `--serve` | — | Disabled | Runs the engine as a headless full-duplex binary wire-protocol server over STDIN/STDOUT for TUI or driver clients. |
| `--bench` | — | Disabled | Executes single-batch GPU compute throughput and latency benchmarking without entering interactive mode. |
| `-h`, `--help` | — | — | Prints command-line usage instructions and available switches, then exits cleanly. |

### Quiescence Gating (Experimental)

| Option | Argument | Default | Description |
| :--- | :--- | :--- | :--- |
| `--quiescence` | — | Disabled | Enables hierarchical multi-scale temporal quiescence gating to skip upper transformer layers during low activation velocity. |
| `--quiescence-threshold` | `<float>` | `0.001` | Sets the cosine similarity activation velocity threshold for layer skipping. |

---

## 3. Operational Modes & Examples

### 3.1 Interactive Console Chat
Run Gemma 4 12B Unified directly in the terminal with hardware Q4_0 acceleration:

```bash
./zig-out/bin/infer --model ../gemma-4-12B-it-qat-q4_0-unquantized --gpu --q4
```

### 3.2 Single-Turn Evaluation with Gemma 4 Template Formatting
```bash
./zig-out/bin/infer \
  --model ../gemma-4-12B-it-qat-q4_0-unquantized \
  --gpu --q4 \
  --prompt "<|turn>user
Summarize the difference between synchronous and asynchronous compute queues.<turn|>
<|turn>model
" \
  --max-tokens 200
```

### 3.3 Headless Wire Protocol Server
Spawn Channel as an asynchronous coprocessor for custom clients, harnesses, or the reference TUI:

```bash
./zig-out/bin/infer \
  --model ../gemma-4-12B-it-qat-q4_0-unquantized \
  --gpu --q4 \
  --serve \
  --memory .memory/.episodic.mem
```

* In server mode, Channel reads 16-byte binary frames from STDIN and streams binary frames to STDOUT.
* For the wire frame specification, see the [Binary Wire Protocol](binary-protocol.md).

### 3.4 Hardware Throughput Benchmarking
Measure raw matrix compute performance (TFLOPS) and decode latency on the host hardware:

```bash
./zig-out/bin/infer --model ../gemma-4-12B-it-qat-q4_0-unquantized --gpu --q4 --bench
```

---

## 4. Exit Codes

| Code | Meaning |
| :--- | :--- |
| `0` | Successful execution / clean shutdown. |
| `1` | General error (invalid arguments, missing model file, or initialization failure). |
| `2` | Vulkan device selection or shader compilation failure. |
| `130` | Terminated via interrupt signal (`SIGINT` / `Ctrl+C`). |
