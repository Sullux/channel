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

Now imagine an app that can use the disk (model) to let you have a conversation with yourself.

<!-- TODO: eliminate HTML -->

<div style="display: flex; align-items: center; justify-content: center; gap: 2rem; margin: 2.5rem 0;">
  <img src="images/disk.svg" alt="Disk" width="80" height="80" />
  <img src="images/r-arrow.svg" alt="Arrow" width="48" height="48" />
  <img src="images/computer.svg" alt="Computer" width="80" height="80" />
</div>

<p id="engine-desc" style="font-size: 1.25rem;">
  That app is the <em>Inference Engine</em>.
</p>
