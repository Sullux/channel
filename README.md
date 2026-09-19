<p align="center">
  <img src="docs/logo.svg" alt="Channel Logo" width="160" height="160" />
</p>

# Channel Inference Engine

A high-performance continuous streaming inference engine with autonomic reflexes, written in pure Zig for Google's Gemma 4 models (`gemma-4-E2B` and `gemma-4-12B-it`).

* **[Official Website](https://sullux.com/projects/channel)**
* **[Quick Start Guide](https://sullux.com/projects/channel/quick-start.html)**

Feel free to jump to our [quick start guide](https://sullux.com/projects/channel/quick-start.html) to start chatting with your local streaming agent as soon as possible!

---

## Documentation

Full architectural specifications, foundations, and API references are available on the [official documentation website](https://sullux.com/projects/channel) and in the [`docs/`](docs/) directory:

* **[The Problem Space](https://sullux.com/projects/channel/foundations/problem.html)**: Four fundamental bottlenecks of modern LLMs (streaming, learning, memory, and the multi-second tool-use tax).
* **[Model Selection](https://sullux.com/projects/channel/foundations/model-selection.html)**: Comparative analysis of Gemma 4 variants and why 12B Unified was selected.
* **[Approach & Philosophy](https://sullux.com/projects/channel/foundations/approach.html)**: Fixed 4,096-slot geometry, autonomic reflexes, and continuous transduction.
* **[Architecture Hub](https://sullux.com/projects/channel/architecture/README.html)**: Overview of the Dual-Plane system (Tensor Brainstem vs Cognitive Mind).
* **[Reference TUI Architecture](https://sullux.com/projects/channel/architecture/tui.html)**: Declarative YAML controllers, Markdown AST rendering, VFS storage, and LIFO interrupts.
* **[Command Line Interface (CLI)](https://sullux.com/projects/channel/api/cli.html)**: Standalone CLI reference and runtime options.
* **[Binary Wire Protocol](https://sullux.com/projects/channel/api/binary-protocol.html)**: 16-byte fixed framing, opcodes, and full-duplex communication.
* **[Client Implementation Guide](https://sullux.com/projects/channel/api/client-guide.html)**: Architectural manual for building custom harnesses (robotics, cloud microservices, voice agents).

---

## License

Copyright © 2026 Sullux LLC. All rights reserved.

Channel is released under a dual-licensing model:

* **Personal & Non-Commercial Use:** Free for personal, non-commercial, evaluation, and educational use. You may inspect, build, run, and modify Channel on your own hardware for personal experimentation.
* **Commercial Use:** A commercial license is required for any commercial deployment, commercial product integration, SaaS hosting, or revenue-generating use. For commercial licensing inquiries and enterprise support, please contact Sullux LLC at [licensing@sullux.com](mailto:licensing@sullux.com) or visit [sullux.com](https://sullux.com).

---

## Warranty & Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL SULLUX LLC, THE AUTHORS, OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
