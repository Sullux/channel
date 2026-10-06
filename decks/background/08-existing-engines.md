---
template: Header/Columns/Footer
steps: true
notes: |
  - there are a few inference engines, but the most common is llama.cpp
  - like most existing engines, its job is simple:
    - takes a block of plain text
      - no bold or italics
      - no headers or footers
      - no concept of back-and-forth conversation
      - just raw text
    - needs to encode the text into numbers
      - this is where we get the idea of "tokens"
        - each bit of text--word, punctuation, prefix/suffix, etc.--gets encoded into a list of numbers
        - the word "learn" is 1 token while "prelearned" is 3 ("pre" "learn" "ed")
        - each list of numbers we call a "vector"
        - a token is one piece of text, and a vector is the list of numbers that can be used with the model
    - applies the vectors to the model
      - works just like the human neocortex
      - applying the vectors (lists of numbers) to the model, the model weights will produce new vectors as output
      - those output vectors are answer
    - decodes the output token vectors back into text, and that text is the answer
---

# Existing inference engines

::: Canvas {#engines-row height="300px"}
---
layout:
  llama: [25%, 50%]
  vllm:  [75%, 50%]
---

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg){#llama width="380px"}
![vLLM](https://vllm.ai/vLLM-Full-Dark-Mode-Logo.svg){#vllm width="240px"}
:::

### What do they do?

---

<br/>

- Accept a single block of text (the `Context`)
- Break the text into pieces (`Tokens`)
- Encode each token as a list of numbers (a `Vector`)
- This list of vectors is the `Hidden State Tensor`

---

<br/>

- Pass the Hidden State to each layer (`Hidden State Passing`)
- Produce a new list of vectors (`Inference`)
- Decode vectors back into text tokens (`Decode`)
- The decoded text is the `Response`

---
