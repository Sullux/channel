---
template: Title/Content
notes: |
  - the engine doesn't care which disk is in the computer
  - the engine just takes input, applies it to the model weights, and returns the output
  - you could take the disk out and put in a different disk and then you'd be speaking with a different model
---

# One Engine, Many Models

<br/>

Ideally, one engine can run any model.*

::: Canvas {#engines-row height="600px"}
---
layout:
  computer:  [50%, 25%]
  disk1: [25%, 75%]
  disk2: [50%, 75%]
  disk3: [75%, 75%]
---

![Computer](images/computer.svg){#computer}
![Disk](images/disk.svg){#disk1}
![Disk](images/disk.svg){#disk2}
![Disk](images/disk.svg){#disk3}
:::

_*But different models have different data formats, so sometimes newer models don't work on existing inference engines right away._
