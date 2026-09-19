# Reference TUI Architecture

## 1. Overview & Architectural Role

The **Channel Reference Terminal User Interface (TUI)** (located in `tui/`) is a production-grade, interactive agent application built on top of the Channel Inference Engine.

> **Important Architectural Boundary:** The TUI is **not** part of the core tensor engine. It is an **illustrative reference implementation** demonstrating how an autonomous cognitive agent, terminal multiplexer, and interactive chat interface can be constructed on top of Channel's binary wire protocol and autonomic reflex plane.

```
┌────────────────────────────────────────────────────────────────────────┐
│ CHANNEL REFERENCE TUI RUNTIME (Node.js)                                │
├────────────────────────────────────────────────────────────────────────┤
│ • Responsive 3-Panel Layout (@sullux/tui, Tier 1 / 2 / 3 adaptation)   │
│ • Zero-Markup Controllers (pure data models & semantic class names)    │
│ • Abstract Markdown Rendering (@sullux/markdown-compiler)              │
│ • Unix-Style Virtual File Subsystem (VFS) & Bounded Reading            │
│ • Dual-Mode Command Execution (Fast inline vs Detached Subshell)       │
│ • Persistent Terminal Sessions (trm 24x80 virtual screen grids)        │
│ • LIFO Notification Interrupt Stack (ack, snooze, turn context)        │
└────────────────────────────────────────────────────────────────────────┘
                                   ▲
               16-Byte Binary Wire Protocol (STDIN / STDOUT)
               OP_STREAM_INPUT, OP_PROBE_AUTONOMIC, OP_RESUME, ...
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ CHANNEL INFERENCE ENGINE (Pure Zig & Vulkan 1.3 Compute)               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Declarative Zero-Markup UI Pattern

The TUI strictly enforces a **Zero-Markup Controller Principle**:
* **Presentation Independence:** JavaScript controllers in `tui/lib/ui/controller/` (`conversation.js`, `stream.js`, `plan.js`, `layout.js`) contain **zero UI markup, zero layout trees, and zero hardcoded colors**. They compute domain data structures and attach semantic CSS-style class names (`userCard`, `streamCard`, `planDone`).
* **Declarative Templates (`view.yaml`):** The entire visual hierarchy, panel layouts, borders, and conditional components are defined in declarative YAML using `@sullux/tui`.
* **Centralized Design System (`theme.yaml`):** Semantic color palettes, typography, borders, and Markdown syntax highlighting are managed centrally.

### Abstract Markdown Control (`tui/lib/ui/markdown.js`)
To display rich streaming text without terminal clutter, the TUI integrates `@sullux/markdown-compiler`:
* Parses streaming responses into structured AST block and inline nodes.
* Compiles AST nodes into styled `@sullux/tui/lib/controls/rich` spans.
* Supports headers, paragraphs, syntax-highlighted code blocks, blockquotes, and nested lists with relative indentation depth preservation.

---

## 3. The Virtual File Subsystem (VFS)

To shield the model's 4,096-slot dynamic ring buffer from context shockwaves, user messages and tool outputs are never pushed into the prompt window en masse. Instead, the TUI implements a real Linux **Virtual File Subsystem (VFS)** rooted at `filesystemRoot`:

```
.agent/
├── msg/
│   ├── user/
│   │   ├── 1001.txt              <-- chmod 0444 (Read-Only user turn)
│   │   └── 1002.txt              <-- chmod 0444 (Read-Only user turn)
│   └── assistant/
│       └── 1002_reply.txt        <-- Generated assistant response
├── trm/
│   └── trm_server/
│       ├── screen.txt            <-- Live 24x80 rendered text grid
│       ├── stdout.log            <-- Raw append-only historical output stream
│       ├── stdin                 <-- Named FIFO for piping interactive input
│       └── ctrl                  <-- Virtual device for control keys (ctrl+c, esc)
├── tmp/
│   ├── cmd_101.stdout.log        <-- Ephemeral log for background subshell
│   └── cmd_102.stdout.log
└── notify/
    ├── pending/
    │   └── not47                 <-- Active interrupt descriptor
    └── snoozed/
        └── not48                 <-- Suppressed until timer wake
```

### Key VFS Invariants
* **Read-Only Inbound Message Isolation:** User turns written to `msg/user/` are sealed with `chmod 0444`. Any attempt by subshells or model tools to alter user input fails with `EACCES` (`Permission denied`).
* **Root-Relative Addressing:** Paths exposed to the model (`msg/user/1001.txt`, `tmp/cmd_101.stdout.log`) are root-relative to `filesystemRoot`.
* **Bounded Reading (`read`):** The `read` tool is strictly capped at **512 characters** (~128 tokens) per invocation with an offset cursor, enforcing incremental ingestion.

---

## 4. Subshell & Persistent Terminal Management

The TUI bifurcates command execution into two distinct patterns:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PROCESS EXECUTION PATTERNS                        │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. SUBSHELL EXECUTION (`cmd`)     │ 2. PERSISTENT TERMINALS (`trm`)   │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Fast, one-off commands          │ • Interactive, stateful CLI tools  │
│ • Fast path: ≤ 250ms & ≤ 128 chars│ • 24x80 rendered `screen.txt` grid │
│   returns inline immediately      │ • Background PTY child process     │
│ • Detached path: > 250ms notifies │ • Real-time ANSI escape parsing    │
│   via interrupt when finished     │ • Control keys (`ctrl+c`, `enter`) │
└───────────────────────────────────┴────────────────────────────────────┘
```

1. **Fast-Path Inline (`cmd`):** Short-lived commands (e.g. `ls`, `git status`) that execute in $\le 250\text{ ms}$ with $\le 128\text{ chars}$ return immediately inline, avoiding multi-turn latency.
2. **Spillover & Detached (`cmd`):** Commands exceeding 128 characters write to `tmp/cmd_<seq>.stdout.log` and return preview slices. Commands taking $> 250\text{ ms}$ detach to the background and issue an interrupt notification when complete.
3. **Interactive Terminals (`trm`):** Stateful processes (e.g. `vim`, `htop`, REPLs) run in dedicated pseudo-terminals (PTYs), parsing ANSI escapes into a live 24-row by 80-column `screen.txt` text grid with `key` control dispatch.

---

## 5. The LIFO Interrupt Controller

All user barge-ins, background subshell completions, and timers enter the TUI through `NotificationManager`:
* **Uniform Typed IDs:** Notifications use IDs formatted as `not<seq>` without underscores (`not101`, `not102`) to prevent pseudo-token delimiter confusion in smaller models.
* **LIFO Priority Stack:** Interrupts are prioritized in Last-In, First-Out order, ensuring the user or the most recent environmental event can immediately steer active cognition.
* **Explicit Resolution Primitives:** 
  - `ack({ id })`: Permanently resolves and dismisses the notification.
  - `snooze({ id })`: Temporarily defers the notification until a timer fires or a subsequent turn completes.
* **Automatic Turn Context Cleanup:** Notifications created to announce inbound user turns (`isTurnContext`) are automatically marked `ACKNOWLEDGED` when the assistant completes its turn (`STOP_END_OF_TURN`).

---

## 6. Configuration Reference (`tui/config.json`)

The TUI's runtime behavior is configured via `tui/config.json`:

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
    "temp": 0.7,
    "topP": 0.95,
    "minP": 0.05,
    "repeatPenalty": 1.1,
    "repeatLastN": 64,
    "frequencyPenalty": 0.1,
    "presencePenalty": 0.1,
    "qThresh": 0.0,
    "thinkingGate": true
  }
}
```

### Configuration Options

| Option | Type | Description |
| :--- | :--- | :--- |
| `modelPath` | `string` | Relative or absolute path to the Gemma 4 model directory containing `config.json`, `tokenizer.json`, and `model.safetensors`. |
| `memoryDir` | `string` | Directory where `.episodic.mem` and write-ahead log files (`.stream.jsonl`, `.tasks.json`) are stored. |
| `filesystemRoot` | `string` | Directory acting as the root of the Unix-style Virtual File Subsystem (`.agent/`). |
| `promptPath` | `string` | Path to the mutable system identity and mission prompt markdown file (`PROMPT.md`). |
| `extraArgs` | `string[]` | Command-line switches passed directly to the spawned engine binary (e.g. `["--gpu", "--q4"]`). |
| `runtime.thinkingBudget` | `number` | Maximum allowed tokens inside the thought channel before forced channel closure. |
| `runtime.maxTokens` | `number` | Runaway circuit-breaker safety ceiling for generation turns (default: `4096`). |
| `runtime.temp` | `number` | Generation sampling temperature (default: `0.7`). |
| `runtime.topP` | `number` | Nucleus sampling threshold (default: `0.95`). |
| `runtime.minP` | `number` | Minimum probability threshold relative to the top candidate (default: `0.05`). |
| `runtime.repeatPenalty` | `number` | Multiplicative penalty applied to previously generated tokens (default: `1.10`). |
| `runtime.repeatLastN` | `number` | Window of recent tokens considered for repetition penalties (default: `64`). |
| `runtime.thinkingGate` | `boolean` | Enables the 1-token autonomic thinking gate probe before response decoding (default: `true`). |
