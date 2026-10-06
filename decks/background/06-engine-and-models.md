---
template: Title/Content
notes: |
  - the engine doesn't care which disk is in the computer
  - the engine just takes input, applies it to the model weights, and returns the output
  - you could take the disk out and put in a different disk and then you'd be speaking with a different model
---

# One Engine, Many Models

Ideally, one engine can run any model.*

<!-- TODO: eliminate HTML -->

<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1.5rem; margin: 2rem 0;">
  <div>
    <img src="images/computer.svg" alt="Computer" width="80" height="80" />
  </div>
  <div style="display: flex; gap: 1.5rem;">
    <img src="images/disk.svg" alt="Disk" width="64" height="64" />
    <img src="images/disk.svg" alt="Disk" width="64" height="64" />
    <img src="images/disk.svg" alt="Disk" width="64" height="64" />
  </div>
</div>

_*But different models have different quirks, so sometimes newer models don't work on older inference engines._
