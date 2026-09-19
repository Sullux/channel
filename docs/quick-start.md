# Quick Start Guide

Get up and running with the **Channel Inference Engine** and its interactive reference TUI in minutes.

---

## 1. Fast Track: Install & Run

For systems meeting the hardware requirements, clone the repositories, compile the engine, and launch the interactive agent:

```bash
git clone https://github.com/sullux/channel
git clone https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized
cd channel
zig build -Doptimize=ReleaseFast
cd tui
yarn install
yarn start
```

---

## 2. System Requirements & Prerequisites

* **Operating System:** Linux x86_64 (Ubuntu 22.04+ or modern distributions).
* **Target Hardware:** AMD Ryzen AI Max+ 395 (Radeon 890M / RDNA 3.5 architecture) or modern AMD GPUs with Vulkan 1.3 compute support and Unified Memory Architecture (UMA).
* **Dependencies:**
  - **Zig Compiler:** Version `0.14.0` or later.
  - **Node.js & Yarn:** Node.js `v18.0.0` or later, with `yarn`.
  - **Vulkan Driver & SDK:** `vulkan-tools`, `libvulkan-dev`, and modern AMDGPU proprietary or open-source drivers supporting Vulkan 1.3 compute.

To verify your Vulkan compute environment:
```bash
vulkaninfo --summary
```

---

## 3. Step-by-Step Walkthrough

### Step 1: Clone the Channel Repository
```bash
git clone https://github.com/sullux/channel
cd channel
```

### Step 2: Download Model Weights
Channel is optimized out-of-the-box for Google's official **Gemma 4 12B Unified** Quantization-Aware Trained (QAT) Q4_0 weights:

```bash
git clone https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized ../gemma-4-12B-it-qat-q4_0-unquantized
```

*Note: You can place model directories anywhere on your filesystem; you simply need to point `modelPath` in `tui/config.json` or `--model` on the CLI to the directory containing `config.json`, `tokenizer.json`, and `model.safetensors`.*

### Step 3: Build the Optimized Engine Binary
Compile the single-binary engine using Zig's `ReleaseFast` optimization mode:

```bash
zig build -Doptimize=ReleaseFast
```

This compiles the binary to `./zig-out/bin/infer`.

### Step 4: Configure the Reference TUI
The TUI's configuration lives in `tui/config.json`. By default, it expects the model to be located at `../../gemma-4-12B-it-qat-q4_0-unquantized`:

```json
{
  "modelPath": "../../gemma-4-12B-it-qat-q4_0-unquantized",
  "memoryDir": "./.memory",
  "filesystemRoot": "./.agent",
  "promptPath": "./PROMPT.md",
  "extraArgs": [
    "--gpu",
    "--q4"
  ],
  "runtime": {
    "thinkingBudget": 256,
    "maxTokens": 4096,
    "thinkingGate": true
  }
}
```

* **Using a Different Model:** To run a smaller test model such as **Gemma 4 E2B**, simply change `"modelPath"` to point to your E2B directory (e.g. `"../../gemma-4-E2B"`).
* For detailed coverage of every configuration parameter, see the [Reference TUI Architecture](architecture/tui.md).

### Step 5: Launch the Reference TUI
```bash
cd tui
yarn install
yarn start
```

* The first launch performs a cold boot, pre-caching the system prompt and immutable tool contracts.
* Subsequent launches utilize Channel's **Warm Boot** capability, restoring working memory and conversation history from `.memory/.snapshot.bin` in **< 100 milliseconds**.

---

## 4. Direct CLI Execution (Without the TUI)

If you prefer to interact with Channel directly from your terminal shell without the full Node.js TUI harness:

### Interactive GPU Chat Session
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it-qat-q4_0-unquantized --gpu --q4
```

### Single Prompt Evaluation
```bash
./zig-out/bin/infer \
  --model ../gemma-4-12B-it-qat-q4_0-unquantized \
  --gpu --q4 \
  --prompt "<|turn>user\nExplain quantum entanglement in two sentences.<turn|>\n<|turn>model\n" \
  --max-tokens 60
```

### Benchmark GPU Compute Throughput
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it-qat-q4_0-unquantized --gpu --q4 --bench
```

For the complete reference of all command-line arguments, switches, and operational modes, see the [CLI Documentation](api/cli.md).
