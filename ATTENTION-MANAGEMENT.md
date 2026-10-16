# Attention Management & Sensory Quiescence Architecture

## 1. Executive Summary & Biological Analogy

Mammalian brains process a torrential flood of continuous multi-modal inputs—retinal image streams at 60 Hz, cochlear audio waveforms at 20 kHz, somatosensory touch, and internal autonomic monitoring—without saturating the working memory of the prefrontal cortex. 

This is made possible by the **Thalamocortical Filter** (the thalamus and reticular activating system). The thalamus acts as a dynamic, neuromodulated sensory gatekeeper. It evaluates incoming signals against active expectations:
* 99.9% of sensory noise (the hum of an air conditioner, peripheral visual motion, ambient background chatter) is processed only in lower sensory nuclei and **quiesces** before reaching the neocortex.
* Working memory is never allocated to mundane signals.
* When a high-delta or unexpected stimulus breaks through (e.g. someone whispering your name in a noisy room), the thalamus routes executive attention to that sensory channel, allowing the brain to query its short-term echoic buffer (*"Wait, what was just said?"*).

In artificial intelligence, contemporary models (like `llama.cpp` or vLLM) suffer from the opposite pathology: **they treat all incoming tokens with identical, full-depth compute across all layers**. Feeding an open microphone or a continuous 2 fps desktop screen capture into a standard 4,096-slot transformer saturates the working context and exhausts GPU memory bandwidth within seconds.

**Channel** solves this through **Dynamic Sensory Quiescence & Attention Budgeting**:
1. **Layer Penetration Depth ($L / 48$) as Continuous SNR:** Quiescence is not a binary on/off switch; the exact layer at which activation delta drops below threshold provides a continuous mathematical metric ($0.0 \dots 1.0$) of signal novelty.
2. **Zero-Sum Attention Budget:** The host orchestrator allocates a finite focus budget (e.g. 10 points) across active channels. Elevating terminal or chat focus automatically throttles audio and visual background expectations.
3. **Speculative Latent Staging (The Thalamic Buffer):** Semi-novel signals that penetrate into intermediate layers (e.g. Layers 13–32) are cached as partial multi-layer KV slabs in server-side unified memory (UMA). If executive focus shifts to that channel, compute resumes from the paused layer forward, achieving instant catch-up with zero redundant lower-layer FLOPs.

---

## 2. Layer Penetration Depth as a Continuous Signal-to-Noise Ratio

In standard deep transformers, all $N$ layers execute sequentially. By monitoring the activation velocity between layers:
$$\Delta x^{(l)} = \| x^{(l)} - x^{(l-1)} \|_2$$

Channel determines the exact layer $L_{\text{stop}} \in [0 \dots 48]$ where $\Delta x^{(l)} < \tau$. 

The ratio:
$$\text{SNR}_{\text{cognitive}} = \frac{L_{\text{stop}}}{N_{\text{layers}}}$$

is a continuous, real-time classifier of cognitive importance:

```
Layer 48 ──┐
   ▲       │ [Zone 3: Executive Cognition] (SNR: 0.67 .. 1.00)
Layer 32 ──┼───────────────────────────────────────────────────
   ▲       │ [Zone 2: Semantic Resonance]  (SNR: 0.25 .. 0.66)
Layer 12 ──┼───────────────────────────────────────────────────
   ▲       │ [Zone 1: Physical Noise]      (SNR: 0.00 .. 0.24)
Layer 0  ──┴───────────────────────────────────────────────────
```

### The Three Functional Zones:

#### Zone 1: Physical & Acoustic Noise ($0 \le L_{\text{stop}} \le 12$, $\text{SNR} < 0.25$)
* **Characteristics:** Microphone thermal noise, room hum, blinking terminal cursors, identical static video frames.
* **Engine Action:** The forward pass terminates immediately at $L_{\text{stop}}$. 
* **Context Impact:** **Zero.** The engine executes an instantaneous $O(1)$ clock rollback (`rollbackClock`), deactivating transient ring slots. The working context never perceives the event.

#### Zone 2: Sub-Conscious Semantic Resonance ($13 \le L_{\text{stop}} \le 32$, $0.25 \le \text{SNR} \le 0.66$)
* **Characteristics:** Coherent ambient dialogue not directed at the assistant, benign compiler warnings in a background terminal, normal desktop mouse movement.
* **Engine Action:** The signal had enough structure to be parsed phonetically or syntactically, but carried insufficient novelty or relevance to trigger conscious action.
* **Context Impact:** The working context clock is rolled back, but the **partial multi-layer KV activations** ($K^{(0..L)}, V^{(0..L)}$) are captured into the server's **Speculative Thalamic Staging Buffer** in unified memory.

#### Zone 3: Conscious Executive Breakthrough ($33 \le L_{\text{stop}} \le 48$, $\text{SNR} > 0.66$)
* **Characteristics:** Direct user questions, fatal compiler errors, sudden visual alarms, high-salience novelty.
* **Engine Action:** The signal penetrates through all upper attention and MLP layers.
* **Context Impact:** The event commits permanently into Tier 2 working context. The engine emits `OP_STATUS` with reason `SIGNAL_BREAKTHROUGH`, alerting the host A-FSM to prioritize the channel.

---

## 3. Dynamic Expectation & The Zero-Sum Attention Budget

Cognitive bandwidth is physically bounded by hardware FLOPs and memory bandwidth. To prevent multi-stream starvation, Channel implements a **Zero-Sum Attention Budget** managed by the host A-FSM:

$$\sum_{c \in \text{Channels}} \text{FocusLevel}(c) = \text{Budget}_{\text{total}} \quad (\text{Default: } 10)$$

```
                       TOTAL ATTENTION BUDGET = 10
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   Active Chat (Focus: 8)     Terminal (Focus: 1)       Audio Mic (Focus: 1)
   - High Sensitivity         - Heavy Quiescence        - Heavy Quiescence
   - τ = 0.005                - τ = 0.080               - τ = 0.120
   - Penetrates to L48        - Dies at Layer 12        - Dies at Layer 6
```

### Dynamic Threshold Mapping:
The host maps integer focus levels ($0 \dots 9$) to physical layer quiescence thresholds ($\tau$):

$$\tau(c) = \tau_{\text{base}} \cdot \kappa^{(9 - \text{FocusLevel}(c))}$$

* When **Focus = 9**: $\tau \approx 0.001$. The filter is wide open. Even subtle nuances propagate through all 48 layers.
* When **Focus = 1**: $\tau \approx 0.150$. The filter is aggressively closed. Only extreme shockwaves can penetrate past Layer 12.
* When **Focus = 0**: The channel is completely muted; incoming frames are dropped at Layer 0 without dispatching Vulkan compute shaders.

### Autonomic Budget Rebalancing
When an event on a background channel breaks through into Zone 3 ($L_{\text{stop}} \ge 33$), the host A-FSM triggers a sub-2ms autonomic probe:
```
"High-delta event detected on channel [terminal]. Shift focus? Options: 0: Ignore, 1: Shift"
```
If `1` wins with confidence $\ge 0.80$, the host rebalances the budget:
* Terminal focus elevates from `1` to `7`.
* Conversational chat focus scales down from `8` to `2`.
* Audio mic remains at `1`.

---

## 4. Speculative Latent Staging (The Server-Side Thalamic Buffer)

A primary bottleneck of naive sensory gating is the **Temporal Discontinuity Paradox**: if an ambient background event was ignored at $T=100$, but at $T=110$ becomes critically relevant, standard engines must either:
1. Re-read the raw text/audio from disk and re-compute all 48 layers from scratch (causing a multi-second pipeline bubble), or
2. Forfeit the background context entirely, causing situational amnesia.

Channel resolves this by utilizing AMD Unified Memory Architecture (UMA) for **Speculative Latent Staging**:

```
[ Inbound Background Stream (e.g. Terminal stdout) ]
                        │
                        ▼
         Vulkan Compute (Layers 0 .. 28)
                        │
             [ Quiesces at Layer 28 ]
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
  [ Rollback Active Clock ]   [ Copy KV Slices (0..28) to UMA Buffer ]
  (Working ring stays clean)  - Size: ~3.5 MB for 32 tokens
                              - Transfer cost: ZERO (Unified RAM)
                              - Stored in server-side circular staging FIFO
                                     │
                 [ Host Elevates Focus to Terminal ]
                                     │
                                     ▼
                      [ WARM-START RESUMPTION ]
                      - Slices (0..28) already resident in VRAM
                      - Dispatch Vulkan compute ONLY for Layers 29..48!
                      - 60% compute savings vs cold prefill
```

### Storage Topology:
* The server maintains a dedicated circular staging pool (default: 256 physical slots in unified memory).
* When a Zone 2 event occurs, the partial KV states ($K^{(0..L)}, V^{(0..L)}$) and the intermediate hidden state $x^{(L)}$ are written into the staging pool.
* If the channel is elevated to focus within a temporal window (e.g. 30 seconds), the engine resumes the forward pass starting directly at layer $L+1$.
* If no focus elevation occurs, the staging pool wraps circularly, overwriting old latent activations with zero CPU/GPU synchronization locks.

---

## 5. Binary Wire Protocol Integration

To support seamless attention management, the Channel Binary Wire Protocol exposes explicit telemetry and control opcodes:

### 1. Inbound Channel Focus Control (`OP_SET_CHANNEL_FOCUS` — `0x0014`)
Sent by the host A-FSM to dynamically set channel expectation levels:
* `channel_id: u8` (`0` = Chat, `1` = Subshell, `2` = Terminal, `3` = Audio, `4` = Vision)
* `focus_level: u8` ($0 \dots 9$)
* `quiescence_override: f32` (Optional manual threshold, or `0.0` for automatic formula)

### 2. Outbound Telemetry Enhancement (`OP_STATUS` — `0x0106`)
Augmented to deliver real-time perceptual depth telemetry on every processed block:
* `active_channel: u8`: Channel that produced the current status.
* `penetration_depth: u8`: Exact layer ($0 \dots 48$) reached by the most recent input chunk.
* `activation_velocity: f32`: Peak $\| \Delta x \|$ observed across the layer stack.
* `is_quiesced: u8`: `1` if execution halted before Layer 48, `0` if it broke through completely.

This allows client applications to build rich, real-time visualizers showing the model's subconscious sensory processing activity in real time.
