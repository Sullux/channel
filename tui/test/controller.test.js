const { describe, it } = require('node:test')
const assert = require('node:assert')
const controller = require('../lib/ui/controller')
const { StateStore } = require('../lib/ui/state')
const { Orchestrator } = require('../lib/orchestrator')

describe('UI Controller & Layout Tiering', () => {
  it('computes layout tier based on terminal width', () => {
    assert.strictEqual(controller.getLayoutTier(180), 1)
    assert.strictEqual(controller.getLayoutTier(140), 2)
    assert.strictEqual(controller.getLayoutTier(100), 3)
  })

  it('detects clipped dimensions below minimum 60x30', () => {
    assert.strictEqual(controller.isClipped(50, 40), true)
    assert.strictEqual(controller.isClipped(100, 20), true)
    assert.strictEqual(controller.isClipped(80, 35), false)
  })

  it('generates conversation, stream, and plan nodes', () => {
    const store = StateStore()
    const orch = Orchestrator()
    orch.createPlan('Test Plan', ['Step A', 'Step B'])
    controller.init(store, null, null, orch, null)

    const convNodes = controller.getConversationNodes()
    assert(convNodes.length >= 1)

    const streamNodes = controller.getStreamNodes()
    assert(streamNodes.length >= 1)

    const planNodes = controller.getPlanNodes()
    assert(planNodes.length >= 2)
  })

  it('toggles mode on a, s, d hotkeys', () => {
    const store = StateStore()
    controller.init(store, null, null, null, null)

    let redrawn = false
    const mockCtx = {
      redraw: () => { redrawn = true },
    }

    controller.onGlobalKey(mockCtx, { key: 's', stopPropagation: () => {} })
    assert.strictEqual(store.state.mode, 'stream')

    controller.onGlobalKey(mockCtx, { key: 'd', stopPropagation: () => {} })
    assert.strictEqual(store.state.mode, 'plan')

    controller.onGlobalKey(mockCtx, { key: 'a', stopPropagation: () => {} })
    assert.strictEqual(store.state.mode, 'chat')
  })

  it('calculates responsive panel visibility and widths across tiers', () => {
    const store = StateStore()
    controller.init(store, null, null, null, null)

    // Tier 1 (>= 160 cols): All 3 panels visible
    store.setDimensions(180, 35)
    store.setMode('chat')
    assert.strictEqual(controller.isConversationVisible(), true)
    assert.strictEqual(controller.getConversationWidth(), '40%')
    assert.strictEqual(controller.isStreamVisible(), true)
    assert.strictEqual(controller.getStreamWidth(), '40%')
    assert.strictEqual(controller.isPlanVisible(), true)
    assert.strictEqual(controller.getPlanWidth(), '20%')

    // Tier 2 (120-159 cols) in chat mode: Chat + Stream visible, Plan hidden
    store.setDimensions(140, 35)
    store.setMode('chat')
    assert.strictEqual(controller.isConversationVisible(), true)
    assert.strictEqual(controller.getConversationWidth(), '50%')
    assert.strictEqual(controller.isStreamVisible(), true)
    assert.strictEqual(controller.getStreamWidth(), '50%')
    assert.strictEqual(controller.isPlanVisible(), false)

    // Tier 2 in plan mode (surfaced on d): Chat + Plan visible, Stream hidden
    store.setMode('plan')
    assert.strictEqual(controller.isConversationVisible(), true)
    assert.strictEqual(controller.getConversationWidth(), '60%')
    assert.strictEqual(controller.isStreamVisible(), false)
    assert.strictEqual(controller.isPlanVisible(), true)
    assert.strictEqual(controller.getPlanWidth(), '40%')

    // Tier 3 (< 120 cols) in chat mode: Chat 100%, others hidden
    store.setDimensions(100, 35)
    store.setMode('chat')
    assert.strictEqual(controller.isConversationVisible(), true)
    assert.strictEqual(controller.getConversationWidth(), '100%')
    assert.strictEqual(controller.isStreamVisible(), false)
    assert.strictEqual(controller.isPlanVisible(), false)

    // Tier 3 in stream mode: Stream 100%, others hidden
    store.setMode('stream')
    assert.strictEqual(controller.isConversationVisible(), false)
    assert.strictEqual(controller.isStreamVisible(), true)
    assert.strictEqual(controller.getStreamWidth(), '100%')
    assert.strictEqual(controller.isPlanVisible(), false)

    // Tier 3 in plan mode: Plan 100%, others hidden
    store.setMode('plan')
    assert.strictEqual(controller.isConversationVisible(), false)
    assert.strictEqual(controller.isStreamVisible(), false)
    assert.strictEqual(controller.isPlanVisible(), true)
    assert.strictEqual(controller.getPlanWidth(), '100%')
  })

  it('submits user input as clean model-agnostic text without Jinja tokens', async () => {
    const store = StateStore()
    const orch = Orchestrator()
    let sentInput = ''
    const mockClient = {
      sendInput: (payload) => { sentInput = payload },
    }

    controller.init(store, mockClient, null, orch, null, 'Test System Prompt')

    const mockCtx = {
      redraw: () => {},
      setFocus: () => {},
    }

    controller.onSubmitInput(mockCtx, { value: 'Hello world', node: { value: 'Hello world', cursor: 11 } })
    // Allow async afsm.step() resolution
    await new Promise((resolve) => setImmediate(resolve))
    assert.ok(sentInput.includes('Hello world'))
    assert.ok(!sentInput.includes('<|turn>'))

    // Second turn should NOT contain system prompt or model tokens
    controller.onSubmitInput(mockCtx, { value: 'Follow up', node: { value: 'Follow up', cursor: 9 } })
    await new Promise((resolve) => setImmediate(resolve))
    assert.ok(!sentInput.includes('Test System Prompt'))
    assert.ok(sentInput.includes('Follow up'))
    assert.ok(!sentInput.includes('<|turn>'))
  })

  it('renders smart 4-line live stream cards with conditional expand hint', () => {
    const store = StateStore()
    store.state.stream = []
    // Short item
    store.addStreamEntry({ type: 'thought', title: '💭 THOUGHT', content: 'Short thought' })
    // Long item (more than 3 lines)
    const longContent = 'Line 1 word word word\nLine 2 word word word\nLine 3 word word word\nLine 4 word word word\nLine 5 word word word'
    store.addStreamEntry({ type: 'response', title: '🤖 ASSISTANT', content: longContent })

    controller.init(store, null, null, null, null)

    // Unselected state (mode = chat)
    store.setMode('chat')
    let nodes = controller.getStreamNodes()
    assert.strictEqual(nodes.length, 2)
    assert.ok(nodes[1].hint.includes('(↑ 2 more)'))
    assert.ok(!nodes[1].hintAction.includes('(x to expand)'))

    // Selected state (mode = stream, selectedIdx = 1)
    store.setMode('stream')
    store.state.selectedIdx.stream = 1
    nodes = controller.getStreamNodes()
    assert.ok(nodes[1].hintAction.includes('(x to expand)'))

    // Expanded state
    store.toggleExpandStreamItem(1)
    nodes = controller.getStreamNodes()
    assert.ok(nodes[1].hintAction.includes('(x to collapse)'))
  })

  it('copies selected conversation or stream item on c key', () => {
    const { getSelectedText, copyToClipboard } = require('../lib/ui/controller/clipboard')
    const store = StateStore()
    controller.init(store, null, null, null, null)

    // Chat mode copy
    store.state.mode = 'chat'
    store.state.conversation = [
      { id: '1', sender: 'User', text: 'First user prompt', time: 1000 },
      { id: '2', sender: 'Assistant', text: 'First assistant reply', time: 2000 },
    ]
    store.state.selectedIdx.chat = 1

    let written = ''
    const mockOut = {
      write: (data) => { written += data },
    }

    assert.strictEqual(getSelectedText(store.state), 'First assistant reply')
    copyToClipboard(getSelectedText(store.state), mockOut)
    assert.ok(written.startsWith('\x1b]52;c;'))
    assert.ok(written.endsWith('\x07'))
    const b64 = written.slice(7, -1)
    assert.strictEqual(Buffer.from(b64, 'base64').toString('utf-8'), 'First assistant reply')

    // Stream mode copy
    store.state.mode = 'stream'
    store.state.stream = [
      { id: 's1', type: 'thought', title: '💭 THOUGHT', content: 'Deep reasoning steps', time: 3000 },
    ]
    store.state.selectedIdx.stream = 0
    assert.strictEqual(getSelectedText(store.state), 'Deep reasoning steps')

    // Test key routing
    let keyHandled = false
    controller.onGlobalKey({ redraw: () => {} }, { key: 'c', stopPropagation: () => { keyHandled = true } })
    assert.strictEqual(keyHandled, true)
  })
})
