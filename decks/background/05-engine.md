---
template: Title/Content
transitions:
  - "#engine-desc"
notes: |
  - now imagine an app
  - we can put the disk in the computer and the app can access the weights in your model
  - the app gives us a chat box on the computer screen
  - we can type questions into the chat box and the app will:
    - use the weights on the disk
    - do the maths to come up with the answer
    - translate the answer weights back into text on the screen for us
  - the app lets us have a conversation with "you"
  - that app is an Inference Engine
---

# The Engine

<br/>

Now imagine an app that can use the disk (model) to let you have a conversation with yourself.

::: Canvas {#engines-row height="400px"}
---
layout:
  disk: [25%, 50%]
  arrow: [50%, 50%]
  computer:  [75%, 50%]
---

![Disk](images/disk.svg){#disk}
![Arrow](images/r-arrow.svg){#arrow}
![Computer](images/computer.svg){#computer}
:::

That app is the _Inference Engine_.
