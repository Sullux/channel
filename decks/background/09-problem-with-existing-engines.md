---
template: Title/Content
steps: true
notes: |
  - what's the problem with llama.cpp and other modern inference engines?
  - first, the one-shot problem
    - we build a huge block of text and pull the lever
    - what happens next is inside a black box
    - no back-and-forth, no switching between reading and writing, no alternating between speaking and listening
  - second, no memory
    - why does the AI seem to remember you?
    - every time we talk to it we bundle up all your previous conversation into the same giant blob of text
    - it has to re-read all of your past conversations _every time_
  - third, no learning
    - learning is different from memory in a very important way
    - memory is episodic
    - learning is instinctual
    - if I ask you about the first time you visited Paris, you have episodic memory
    - if I ask you the capital of France, you remember reflexively
    - you don't think about the classroom you were sitting in when you learned that fact: it is just instinctive knowledge
    - if it looks like a model is learning, that is a trick: it is just re-reading all of your past conversations and using those episodes to inform its next answer
    - it has not actually learned (or remembered!) anything
  - fourth, text only
    - this is the most pernicious of all
    - how would you get through the day if you had no episodic memory at all and you had to rely on your own notes
    - heard about an event you wanted to attend? read your notes later and tell me why the event interested you? you're not allowed to remember how you felt when you heard about it: you're only allowed to read your notes
    - and what about tool use?
    - you probably know that models can use tools
    - that's how things like OpenClaw can let an agent respond to a message
      - you send a Telegram message to your agent
      - OpenClaw receives it and adds your message to a giant wall of text
      - OpenClaw gives the text to an inference engine
      - the inference engine pushes the text through a model
      - the model's answer includes instructions for how to answer back on Telegram
      - those "answer on telegram" instructions are the "tool call"
      - OpenClaw sees the tool call instructions inside the model's answer
      - OpenClaw follows the instructions to send the answer on Telegram
    - sounds complicated? it is!!!
---

# So what's the problem?

The classic inference engine has a number of drawbacks:

- Everything in one shot. No interruptions, no back-and-forth.
- No memory. The model is read-only. It only knows what it knew when it was put on disk.
- No learning. The model only has the skills and understanding it had when it was put on disk.
- Text only.
  - A _memory_ is like reading written notes about a concert instead of actually being there
  - _Tool use_ is like reading instructions to a blindfolded person driving a car
