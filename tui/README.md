# AI Research TUI Client

An interactive 3-panel terminal user interface for real-time streaming inference, autonomous thought tracing, and hierarchical task planning with Gemma 4 models on integrated AMD Vulkan GPUs and CPUs.

---

## Quick Start

```bash
# Launch with default configuration (reads tui/config.json)
node tui/index.js

# Launch with an explicit configuration file
node tui/index.js --config ./custom-config.json

# Pass custom engine arguments directly
node tui/index.js --gpu --q4 --memory ./.my-workspace/.episodic.mem
```

---

## Command Line Arguments

| Argument | Description |
|---|---|
| `--config <path>`, `-c <path>` | Path to a custom `config.json` configuration file (defaults to `tui/config.json`). |
| `[extraArgs...]` | Any additional flags (e.g. `--gpu`, `--q4`, `--cpu`, `--threads 16`) are forwarded directly to the backend engine binary (`../zig-out/bin/infer`). |

---

## Configuration Reference (`config.json`)

The TUI is configured via a JSON file (by default `tui/config.json`). Below is an example configuration with all supported properties:

```json
{
  "modelPath": "../../gemma-4-12B-it-qat-q4_0-unquantized",
  "memoryDir": "./.memory",
  "promptPath": "./PROMPT.md",
  "extraArgs": [
    "--gpu",
    "--q4"
  ],
  "runtime": {
    "thinkingBudget": 512,
    "maxTokens": 512,
    "temp": 0.7,
    "topP": 0.95,
    "minP": 0.05,
    "repeatPenalty": 1.1,
    "repeatLastN": 64,
    "qThresh": 0.0
  }
}
```

### Configuration Properties

| Property | Type | Default | Description |
|---|---|---|---|
| `modelPath` | `string` | `../../gemma-4-12B-it` | Path to the directory containing model weights and `tokenizer.json`. |
| `memoryDir` | `string \| null` | `null` | Workspace directory for persistence. When set, persists episodic memory to `<memoryDir>/.episodic.mem` and telemetry trace to `<memoryDir>/.stream.jsonl`. When omitted or `null`, runs ephemerally in RAM. |
| `promptPath` | `string` | `./PROMPT.md` | Path to the user/mission prompt file. Concatenated with `./PROMPT_KERNEL.md` on session initialization. |
| `extraArgs` | `string[]` | `["--gpu", "--q4"]` | Default command-line flags forwarded to the engine server binary on startup. |
| `runtime` | `object` | `{ ... }` | Autoregressive sampling and engine execution settings. |

### `runtime` Sub-properties

| Setting | Type | Default | Description |
|---|---|---|---|
| `thinkingBudget` | `number` | `512` | Token limit allocated for reasoning inside the `<|think|>` channel before closing the channel. |
| `maxTokens` | `number` | `512` | Token limit for the final user-facing response content. |
| `temp` | `number` | `0.7` | Softmax temperature for token sampling. |
| `topP` | `number` | `0.95` | Nucleus sampling probability mass cutoff. |
| `minP` | `number` | `0.05` | Minimum relative probability threshold for candidate tokens. |
| `repeatPenalty` | `number` | `1.1` | Multiplicative penalty applied to recently emitted tokens to prevent repetition loops. |
| `repeatLastN` | `number` | `64` | Number of previous tokens tracked for repetition penalty calculation. |
| `qThresh` | `number` | `0.0` | Quiescence threshold for dynamic layer skipping (0.0 evaluates all 48 layers). |

---

## User Interface & Navigation

The TUI interface consists of three responsive columns:
1. **Left Column (40%)**: Scrollable Conversation thread and pinned bottom Input Box.
2. **Middle Column (40%)**: Live Stream displaying reasoning thoughts (`💭 THOUGHT`), tool executions (`🛠️ TOOL`), tool outputs (`📦 RESULT`), and system events.
3. **Right Column (20%)**: Hierarchical Plan Stack showing active goals, steps, deferred timers, and completion statuses.

### Keyboard Shortcuts

| Shortcut | Description |
|---|---|
| `Enter` | Focus input box (when blurred) / Submit prompt (when editing). |
| `Shift+Enter` | Insert a newline into multi-line prompt input. |
| `Esc` | Unfocus/blur input box and return to panel navigation mode. |
| `a` | Select and focus the Conversation panel. |
| `s` | Select and focus the Live Stream panel. |
| `d` | Select and focus the Plan Stack panel. |
| `j` / `k` (or `↓` / `↑`) | Scroll down / up through messages and stream cards. |
| `x` | Expand or collapse the selected stream item. |
| `Ctrl+Q` / `Ctrl+C` | Gracefully terminate the TUI and shutdown the inference backend. |

---

## Architecture & Engineering

For deep details on the autonomous orchestrator, tool schemas, streaming protocol, and memory persistence mechanisms, see [ENGINEERING.md](./ENGINEERING.md).
