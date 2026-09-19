# TUI Autonomous Orchestration Architecture

This document records the design, engineering principles, state machine mechanics, and cognitive lifecycle of the interactive TUI client and autonomous streaming orchestrator.

---

## 1. Executive Summary & Core Philosophy

Standard autoregressive language models (including the Gemma 4 family) are fundamentally trained on **discrete, synchronous conversational turns**:
1. Host feeds user prompt: `<|turn>user ... <turn|>`.
2. Model reasons and completes: `<|turn>model <|think|> ... <channel|> ... <turn|>`.
3. Model halts on End-Of-Sequence (`EOS`), idling the GPU until the next user prompt.

This paradigm fails in continuous agentic workflows where a model must:
- Page through long documents (10k+ tokens) without blowing context windows.
- Perform multi-step terminal tasks across dozens of clock ticks.
- Accept live mid-flight interruptions without losing working memory or aborting progress.
- Resume suspended tasks cleanly once an interruption is addressed.

Our architecture breaks out of the turn-based cage using a **"Dumb" Task-Agnostic Orchestrator** paired with **Autoregressive Thought Injection (Thought Prefixing)**, **LIFO Task Stacking**, and the **Persistent Episodic Memory Subsystem**.

---

## 2. Architectural Pillars

```
                     ┌─────────────────────────────────────────┐
                     │          USER INPUT / TIMERS            │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
       ┌────────────►│         AUTONOMOUS ORCHESTRATOR         │
       │             │  - Priority LIFO Task Stack             │
       │             │  - Thought Prefix Synthesizer           │
       │             │  - Timer & Interruption Manager         │
       │             └────────────────────┬────────────────────┘
       │                                  │ Injected Thought Frame
       │                                  ▼
       │             ┌─────────────────────────────────────────┐
       │             │           GEMMA 4 12B ENGINE            │
       │             │  - Dynamic Sliding Ring Buffer (1024)   │
       │             │  - Hippocampus Staging & Debounce       │
       │             │  - GPU-Side Top-64 Softcapping Sampler  │
       │             └────────────────────┬────────────────────┘
       │                                  │
       │                 ┌────────────────┴────────────────┐
       │                 ▼                                 ▼
       │        [Tool Call Generated]            [Direct Public Output]
       │        - `plan(brief, steps)`           - Output printed to user
       │        - `done(summary)`                - If stack empty -> IDLE
       │        - `defer(duration)`              - If stack active -> Next Tick
       │        - `terminal_write(...)`
       │                 │
       └─────────────────┴─────────────────────────────────┘
```

### Principle 1: Task-Agnostic "Dumb" Orchestrator
The orchestrator possesses zero domain knowledge about specific jobs (e.g., "reading files", "writing code", "monitoring builds"). It manages only a lightweight stack of atomic plan steps, timer queues, and execution ticks.

### Principle 2: Low Cognitive Load on 12B-Class Models
Small-to-medium parameter models (12B) frequently hallucinate or loop when required to maintain and mutate large, deeply nested JSON state trees. We reduce the model's tool burden to atomic, zero-overhead primitives (`plan`, `done`, `defer`, `ask_user`) while the orchestrator handles hierarchical ID generation, stack nesting, and state transitions.

### Principle 3: Thought Injection (Thought Prefixing)
Rather than appending verbose system instructions on every turn, the orchestrator feeds an open `<|think|>` channel prefix directly into the engine's autoregressive sampler. By setting the model's self-attention attractor into an immediate cognitive groove, the model requires zero ramp-up tokens to re-orient itself.

### Principle 4: Terminal Paging as the Streaming Substrate
Long files and documents are not ingested through bespoke chunking pipelines. The model reads files through an internal terminal session using standard CLI utilities (`cat`, `head`, `less`). The sliding ring buffer retains the active screen, while the `Hippocampus` automatically captures and indexes evicted context into `.episodic.mem`.

---

## 3. The Task Stack & Interruption Lifecycle

The orchestrator maintains an in-memory **LIFO (Last-In, First-Out) Priority Stack**:

```
[Stack Top (Active)]  ---> Step 1140: Respond to user interruption ("What's the status?")
[Suspended Step]      ---> Step 1138.2: Scan filesystem for one-off package installations
[Parent Plan]         ---> Plan 1138: Write automated update script
```

### Lifecycle Rules:
1. **New Plan Creation**: When the model calls `plan(brief, steps)`, the orchestrator converts the plan into a tree of steps, assigns hierarchical IDs (`1138.1`, `1138.2`, `...`), and pushes Step 1 to the stack top.
2. **Step Completion**: When the active step completes, the model calls `done()`. The orchestrator pops the step, advances to the next sibling step, and ticks inference.
3. **Task Suspension & Deferral**: If a step is blocked (e.g. waiting for a long compilation or background job), the model calls `defer(duration="30s", reason="awaiting build")`. The orchestrator moves the step to the `TimerManager` queue and ticks the next pending item or sleeps.
4. **Blocking on User Input (`ask_user`)**:
   - When a step requires human intervention (e.g. `sudo` installation, confirmation, credentials), the model calls `ask_user({ brief: "...", message: "..." })`.
   - The orchestrator displays `message` to the user, transitions the task status to `WAITING_FOR_USER` with `brief` tagged, and immediately puts the engine to sleep (0% CPU/GPU).
5. **Live User Interruption & Automatic Resumption**:
   - When user input arrives while a task is executing, the orchestrator pauses the active step, pushes a transient user interaction task to the stack top, and prompts the model.
   - Once the user interaction is fulfilled, the orchestrator pops the interruption task and **automatically resumes** the next suspended task on the stack without confirmation prompts.
6. **Idle State**: When the stack is empty (or all tasks are sleeping on timers / `WAITING_FOR_USER`) and the model emits an end-of-turn delimiter (`<turn|>`), the orchestrator enters zero-CPU sleep until the next user prompt or timer expiration.

---

## 4. User Request Routing: The "Answer vs. Plan" Fork

To avoid unnatural latency on simple questions while ensuring complex action requests are never lost, user turns follow a dual-lane strategy with an automatic safety net:

### A. The Injected Decision Thought
When user input arrives, the orchestrator opens the model turn with a contextual thought frame.

If there are **no tasks waiting for user intervention**:
```
<|turn>model
<|think|>
User request received.
Decision:
- If this is a simple query I can answer directly in one turn, respond immediately.
- If this requires actions, multiple steps, or investigation, call `plan` with a brief summary.
```

If there are **tasks in `WAITING_FOR_USER` state**:
```
<|turn>model
<|think|>
User message received: "[User Message Text]"

Tasks currently awaiting user intervention:
- Step 1138.2: [Brief summary of requested user action]

Decision:
1. If the user's message fulfills an awaiting task, resume that task (or call `done()` if completed).
2. If the user provided a new unrelated instruction, prioritize answering/planning the new instruction (suspended tasks will remain in `WAITING_FOR_USER`).
3. If the user cancelled the task, abort the task.
Next action:
```

### B. Execution Lanes:
1. **Direct Answer (Zero-Task Path):**
   - If the model determines it can answer immediately, it emits text in the public channel (`<|channel>response ... <channel|>`) and closes the turn (`<turn|>`).
   - Output is printed directly to the TUI. No tasks are created. If a prior task was suspended underneath, the orchestrator **automatically resumes** the suspended task on the next tick.
2. **Structured Multi-Step Plan:**
   - If the model calls `plan(brief, steps)`, the orchestrator creates the plan stack and enters the autonomous tick loop.
3. **Auto-Wrap Safety Net (The "Impulsive Model" Fallback):**
   - If a 12B model bypasses `plan` and immediately invokes a working tool (e.g., `terminal_write`, `recall`), the orchestrator catches the event and automatically synthesizes an implicit task entry: `Task: "<Tool Name>: <Command snippet>"`.
   - **Result:** No action or side-effect is ever executed untracked.

---

## 5. Tool Primitives

| Tool | Parameters | Description |
|---|---|---|
| `plan` | `brief: string`, `steps: string[]` | Creates a new structured plan or child sub-plan. Pushes Step 1 to stack. |
| `done` | `summary?: string` | Marks the current active step as complete and advances stack. |
| `defer` | `duration: string`, `reason?: string` | Suspends active step onto timer queue (`"10s"`, `"2m"`). |
| `ask_user` | `brief: string`, `message: string` | Presents `message` to user, marks task `WAITING_FOR_USER` with `brief`, and sleeps. |
| `terminal_write` | `input: string` | Sends raw keystrokes/commands to the active pty/terminal session. |
| `recall` | `query: string`, `top_k?: number` | Performs explicit associative memory search and latent KV slab rehydration. |

---

## 6. Thought Injection Templates

On every tick, the orchestrator builds a contextual thought prefix tailored to the current stack state:

### Initial Step Tick:
```
<|turn>model
<|think|>
Focus: Plan [ID] - [Plan Brief]
Active Step [ID.Step]: [Step Brief]
Status: In progress.
Next action:
```

### Post-Interruption Automatic Resume Tick:
```
<|turn>model
<|think|>
Interruption handled. Automatically resuming Plan [ID] at Step [ID.Step]: [Step Brief].
Previous context remains active in episodic memory.
Next action:
```

### User Input Fulfilled Resume Tick:
```
<|turn>model
<|think|>
User fulfilled Step [ID.Step] ([Brief]). Resuming execution.
Next action:
```

### Timer Expiration Tick:
```
<|turn>model
<|think|>
Timer expired for Step [ID.Step] ([Reason]). Checking terminal/state for updates.
Next action:
```

---

## 7. Cognitive Memory Integration

The orchestrator interfaces seamlessly with the engine's three-tier memory hierarchy:

1. **Tier 1 (Static Anchors):** First 32 tokens (system prompt kernel and tool definitions) permanently pinned.
2. **Tier 2 (Sliding Context):** 1,024 active sliding-window tokens. As terminal outputs and step notes scroll off, they are continuously staged in the `Hippocampus`.
3. **Tier 3 (Upper Reasoning Recall):** Top-128 implicit resonant memories injected into layers 16–47 prior to generation.
4. **Episodic Persistence (`.episodic.mem`):** 
   - Every completed step or debounce flush writes a 64-byte aligned episode header, unit-normalized centroid vector, and full-fidelity multi-layer FP16 $(K, V)$ tensor slab.
   - When a suspended plan is resumed after thousands of conversational tokens, calling `recall()` rehydrates the original task context directly into GPU UMA memory in $<0.2\text{ ms}$.

---

## 8. TUI Layout, Responsive Panels & Modal Ergonomics

The frontend client leverages `@sullux/tui` to provide a real-time, responsive multi-pane terminal interface designed specifically for autonomous agent monitoring and debugging.

### A. Three-Panel Visual Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1 (≥ 160 cols): Full Tri-Pane View (40% Conversation / 40% Stream / 20% Plan)     │
├──────────────────────────────┬──────────────────────────────┬──────────────────────────┤
│ 💬 CONVERSATION (40%)        │ ⚡ LIVE STREAM (40%)         │ 📋 PLAN STACK (20%)      │
│                              │                              │                          │
│ User: Update all packages    │ <|think|>                    │ [1138] Update Packages   │
│ Assistant: Scanning system...│ User requested update.       │  ├─ [1138.1] Scan (Done) │
│                              │ <|tool>terminal_write(...)   │  └─ [1138.2] Write script│
│                              │ ↳ dpkg -l ... (exit 0)       │     ⏳ WAITING_FOR_USER  │
├──────────────────────────────┤                              │                          │
│ [Input Box (Caret/History)]  │                              │                          │
├──────────────────────────────┴──────────────────────────────┴──────────────────────────┤
│ [Enter] Type  [a] Chat  [s] Stream  [d] Plan  [c] Copy  [Esc] Pause  [Ctrl+Q] Quit     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Note on Input Box Placement:** The input control is confined **strictly to the bottom of the Conversation column**. This maximizes vertical context for the Live Stream and Plan Stack columns so long telemetry logs and deep plan trees remain visible.

---

### B. Turn Routing & Panel Separation

To eliminate clutter in the conversational view while retaining 100% observability, inbound turns are routed deterministically:

| Message / Event Type | Wire / Token Boundary | Conversation Panel | Live Stream Panel | Plan Stack Panel |
|---|---|:---:|:---:|:---:|
| **User Prompt** | `<|turn>user ... <turn|>` | **YES** (User Bubble) | **YES** (Chronological sequence) | No |
| **Model Thought** | `<|think|> ... <channel|>` | **NO** | **YES** (Dim collapsible block) | No |
| **Action Tools** (`plan`, `done`, `defer`, `terminal_write`, `recall`) | `<|tool> ... <tool|>` | **NO** | **YES** (Color-coded call + result) | **YES** (If plan/done/defer) |
| **User Request Tool** (`ask_user`) | `<|tool> ask_user({ message: "..." }) <tool|>` | **YES** (Rendered as Assistant message) | **YES** (Tool call + text) | **YES** (Tagged `⏳ WAITING_FOR_USER`) |
| **Public Assistant Response** | `<|channel> ... <channel|>` | **YES** (Assistant Bubble) | **YES** (Public stream text) | No |

---

### C. Responsive Real Estate & Overlay Modes

The TUI dynamically recalculates its column layout on terminal resize events (`SIGWINCH`) based on minimum widths of **60 cols** Conversation, **60 cols** Live Stream, and **40 cols** Plan Stack:

1. **Tier 1 (≥ 160 cols):** Full Tri-Pane View (`40%` Conversation / `40%` Live Stream / `20%` Plan Stack).
2. **Tier 2 (120–159 cols):** Dual-Pane View (`50%` Conversation / `50%` Live Stream).
   - The Plan Stack is hidden by default. Pressing **`d`** surfaces the Plan Stack (`40%`) alongside Conversation (`60%`) while hiding Stream. Pressing **`a`** or **`s`** returns to Conversation + Stream.
3. **Tier 3 (< 120 cols):** Single-Pane Focused View (`100%` width).
   - Focused panel takes full width (`100%`). Default is Conversation; pressing **`s`** surfaces Stream; pressing **`d`** surfaces Plan; pressing **`a`** or **`Enter`** returns to Conversation.
4. **Hard Minima & Clipping HUD:**
   - Minimum supported terminal dimensions: **60 columns $\times$ 30 rows**.
   - If terminal width $< 60$ or height $< 30$, a high-contrast status banner is rendered at the bottom:
     `⚠️ TERMINAL TOO SMALL (Minimum 60x30 required) — CONTENT CLIPPED`.

---

### D. Modal Keyboard Ergonomics (Vim-Style Navigation)

The TUI avoids cycle-heavy `Tab` navigation in favor of direct modal hotkeys:

- **`Enter` (from Normal mode):** Enters **Edit Mode** (focuses native multiline text input at the bottom of the Conversation column).
  - `Enter`: Submits message (no-op if empty).
  - `Esc`: Exits edit mode back to Normal mode (preserves unsubmitted draft).
  - `Up` / `Down`: Navigates prompt history (when cursor is on top/bottom line).
  - `Ctrl+C`: Clears active input buffer.
- **`a`:** Focuses **Conversation Panel** in Normal/Browse mode.
  - `j` / `k`: Scroll down / up by item (or page if item is taller than viewport).
  - `c`: Copies selected message bubble to clipboard.
- **`s`:** Focuses **Live Stream Panel** (or opens it as an overlay modal if hidden).
  - `j` / `k`: Scroll down / up.
  - `x`: Toggles expanded/collapsed view for the focused thought or tool block.
  - `c`: Copies selected telemetry entry to clipboard.
- **`d`:** Focuses **Plan Stack Panel** (or opens it as an overlay modal if hidden).
  - `j` / `k`: Scroll up / down task tree.
  - `c`: Copies active plan brief and step list.
- **`Esc` (in Normal mode):** Pauses model inference engine. Displays centered HUD:
  `⏸️ INFERENCE PAUSED — Press 'r' to resume`.
  - `r`: Resumes inference execution.
- **`Ctrl+Q`:** Gracefully shuts down TUI, sends `OP_ABORT` if active, and disconnects socket.

---

### E. Auto-Scroll vs. Sticky Scroll Semantics

- In both Conversation and Live Stream viewports, when scrolled to the very bottom, new streaming tokens and tool logs automatically pin and scroll with new output.
- Pressing `k` (scrolling up) immediately disengages auto-scroll to allow calm scrollback reading.
- Scrolling back to the bottom (`j` or `G`) automatically re-engages sticky auto-scroll.

---

### F. Dynamic Bottom Status Bar

The bottom bar updates continuously to reflect the active focus and available hotkeys:

- **In Edit Mode:** `[Enter] Send  [Esc] Normal Mode  [Ctrl+C] Clear  [Up/Down] History`
- **In Conversation Mode (`a`):** `[j/k] Scroll  [Enter] Type  [s] Stream  [d] Plan  [c] Copy  [Esc] Pause  [Ctrl+Q] Quit`
- **In Stream Mode (`s`):** `[j/k] Scroll  [x] Expand/Collapse  [a] Chat  [d] Plan  [c] Copy  [Esc] Close/Pause`
- **In Plan Mode (`d`):** `[j/k] Navigate Tasks  [a] Chat  [s] Stream  [Esc] Close/Pause`
- **When Inference Paused:** `[r] Resume Inference  [Ctrl+Q] Quit`

