const test = require('node:test')
const assert = require('node:assert')
const { Afsm } = require('../lib/afsm')

test('Afsm evaluates discrete probe and transitions state', async () => {
  const definition = {
    initial: 'idle',
    states: {
      idle: {
        probe: {
          prompt: 'Choose next action: 1 (read), 2 (wait)',
          candidates: [
            { key: '1', op: 'start_reading', target: 'reading' },
            { key: '2', op: 'wait', target: 'idle' },
          ],
        },
      },
      reading: {
        probe: {
          prompt: 'Reading: 1 (continue), 0 (done)',
          candidates: [
            { key: '1', op: 'continue_reading', target: 'reading' },
            { key: '0', op: 'finish', target: 'idle' },
          ],
        },
      },
    },
  }

  const mockClient = {
    probe: async (prompt, candidates) => {
      if (prompt.includes('Choose next action')) {
        return { winningIdx: 0, confidence: 0.95, entropy: 0.05, text: '' } // selects '1'
      }
      return { winningIdx: 1, confidence: 0.9, entropy: 0.1, text: '' } // selects '0'
    },
  }

  let started = false
  let finished = false
  const ops = {
    start_reading: (ctx) => {
      started = true
      return { offset: 0 }
    },
    finish: (ctx) => {
      finished = true
      return { offset: 1024 }
    },
  }

  const fsm = Afsm({ definition, client: mockClient, initialContext: {}, ops })
  assert.strictEqual(fsm.getState().currentState, 'idle')

  const res1 = await fsm.step()
  assert.strictEqual(res1.from, 'idle')
  assert.strictEqual(res1.to, 'reading')
  assert.strictEqual(res1.op, 'start_reading')
  assert.strictEqual(started, true)
  assert.strictEqual(fsm.getState().context.offset, 0)

  const res2 = await fsm.step()
  assert.strictEqual(res2.from, 'reading')
  assert.strictEqual(res2.to, 'idle')
  assert.strictEqual(res2.op, 'finish')
  assert.strictEqual(finished, true)
  assert.strictEqual(fsm.getState().context.offset, 1024)
})

test('Afsm supports generative micro-decode for notes', async () => {
  const definition = {
    initial: 'reflecting',
    states: {
      reflecting: {
        probe: {
          prompt: (ctx) => `Summarize finding at offset ${ctx.offset}:`,
          maxTokens: 16,
          op: 'save_note',
          target: 'idle',
        },
      },
      idle: {},
    },
  }

  const mockClient = {
    probe: async (prompt, candidates, maxTokens) => {
      assert.strictEqual(maxTokens, 16)
      assert.ok(prompt.includes('offset 512'))
      return { winningIdx: 0, confidence: 0, entropy: 0, text: 'File contains Vulkan compute headers.' }
    },
  }

  const ops = {
    save_note: (ctx, probeRes) => ({
      notes: [...(ctx.notes || []), probeRes.text],
    }),
  }

  const fsm = Afsm({
    definition,
    client: mockClient,
    initialContext: { offset: 512, notes: [] },
    ops,
  })

  const res = await fsm.step()
  assert.strictEqual(res.to, 'idle')
  assert.strictEqual(res.op, 'save_note')
  assert.strictEqual(fsm.getState().context.notes.length, 1)
  assert.strictEqual(
    fsm.getState().context.notes[0],
    'File contains Vulkan compute headers.',
  )
})

test('ChatMachine evaluates thinking gate with confidence threshold', async () => {
  const { ChatMachine } = require('../lib/afsm')

  // Case A: winningIdx 0 with high confidence (>= 0.80) -> direct_response
  const mockClientDirect = {
    probe: async () => ({
      winningIdx: 0,
      confidence: 0.95,
      entropy: 0.05,
      costMs: 1.5,
      text: '',
    }),
  }

  const fsmDirect = Afsm({
    machine: ChatMachine,
    client: mockClientDirect,
    initialContext: { val: 'Good morning, Hopper.' },
  })

  const resDirect = await fsmDirect.step()
  assert.strictEqual(resDirect.op, 'direct_response')
  assert.strictEqual(resDirect.transition?.meta?.isDirect, true)
  assert.strictEqual(fsmDirect.getState().currentState, 'idle')

  // Case B: winningIdx 0 but lower confidence (< 0.80) -> fall through to think
  const mockClientThinkFallback = {
    probe: async () => ({
      winningIdx: 0,
      confidence: 0.72,
      entropy: 0.35,
      costMs: 1.8,
      text: '',
    }),
  }

  const fsmThink = Afsm({
    machine: ChatMachine,
    client: mockClientThinkFallback,
    initialContext: { val: 'How would you summarize a 10GB file?' },
  })

  const resThink = await fsmThink.step()
  assert.strictEqual(resThink.op, 'think')
  assert.strictEqual(resThink.transition?.meta?.isDirect, false)
  assert.strictEqual(fsmThink.getState().currentState, 'idle')
})

test('FocusMachine drives focus arbitration between channels', async () => {
  const { FocusMachine } = require('../lib/afsm')
  const { ChannelManager } = require('../lib/channels')

  const cm = ChannelManager()
  cm.registerChannel({ id: 'trm/build', isFocused: false })

  const mockClient = {
    probe: async () => ({
      winningIdx: 1, // selects '1': switch_focus
      confidence: 0.92,
      entropy: 0.08,
      costMs: 1.2,
      text: '',
    }),
  }

  const ops = {
    switch_focus: (ctx) => {
      cm.setFocus(ctx.incomingChannel)
      return { focusedChannel: ctx.incomingChannel }
    },
    bookmark_interrupt: () => ({ bookmarked: true }),
  }

  const fsm = Afsm({
    machine: FocusMachine,
    client: mockClient,
    initialContext: {
      incomingChannel: 'trm/build',
      focusedChannel: cm.getFocused().id,
      preview: 'build error in shader',
    },
    ops,
  })

  assert.strictEqual(cm.getFocused().id, 'chat/user')
  const res = await fsm.step()
  assert.strictEqual(res.op, 'switch_focus')
  assert.strictEqual(cm.getFocused().id, 'trm/build')
  assert.strictEqual(fsm.getState().context.focusedChannel, 'trm/build')
})
