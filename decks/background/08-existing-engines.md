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

<!-- TODO: evenly space these from each other across full width and center vertically within their row -->
![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg){:width="400px"} ![vLLM](https://vllm.ai/vLLM-Full-Dark-Mode-Logo.svg){:width="400px"}

### What do they do?

---

- Accept a single block of text
- Break the text into tokens
- Encode each token into a list of numbers called a **Vector**

---

- Apply the vectors to the model
- Get a list of token vectors back from the model
- Decode the answer tokens back into text

---
