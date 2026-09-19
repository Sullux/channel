const { describe, it } = require('node:test')
const assert = require('node:assert')
const { SyntaxTracker } = require('../lib/stream/syntax')
const { stateStoreFactory } = require('../lib/ui/state')
const controller = require('../lib/ui/controller')

describe('Streaming Transduction Flow & In-Flight Barge-In', () => {
  it('detects natural resting points across multiline streamed tokens', () => {
    const tracker = SyntaxTracker()

    // 1. Initial code block starts
    tracker.ingestChunk('```typescript\n')
    assert.strictEqual(tracker.isAtRest(), false)
    assert.strictEqual(tracker.isNaturalBoundary('```typescript\n'), false)

    // 2. Code inside block
    tracker.ingestChunk('const x = (10 + 20);\n')
    assert.strictEqual(tracker.isAtRest(), false)

    // 3. Code block closes
    tracker.ingestChunk('```\n\n')
    assert.strictEqual(tracker.isAtRest(), true)
    assert.strictEqual(tracker.isNaturalBoundary('```typescript\nconst x = (10 + 20);\n```\n\n'), true)
  })

  it('handles in-flight user barge-in cleanly without losing active thoughts or responses', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    let sentPayloads = []
    const mockClient = {
      sendInput: (payload) => { sentPayloads.push(payload) },
    }
    const mockSession = {}
    const mockOrchestrator = {
      getWaitingForUserTasks: () => [],
      getActiveStep: () => null,
    }
    const mockTimers = {}

    controller.init(store, mockClient, mockSession, mockOrchestrator, mockTimers, 'You are Gemma.')

    // Simulate model actively generating a thought and partial response
    store.appendActiveThought('Analyzing performance data...')
    store.appendActiveResponse('Based on the initial benchmarks, ')
    store.setGenerating(true)

    // User barges in with new input
    const mockCtx = {
      setFocus: () => {},
      redraw: () => {},
    }
    controller.onSubmitInput(mockCtx, {
      value: 'Wait, focus on memory bandwidth instead.',
    })

    // Verify in-flight thought was flushed
    const thoughts = store.state.stream.filter(s => s.type === 'thought')
    assert.strictEqual(thoughts.length, 1)
    assert.strictEqual(thoughts[0].content, 'Analyzing performance data...')

    // Verify user message staged as pendingInterjection (or appended) and present in conversation
    const userMsg = store.state.pendingInterjection || conv[conv.length - 1]
    assert.strictEqual(userMsg.sender === 'User', true)
    assert.strictEqual(userMsg.text, 'Wait, focus on memory bandwidth instead.')

    // When the in-flight response finishes/flushes, pendingInterjection moves to conversation
    store.flushActiveResponse()
    const convAfter = store.state.conversation
    assert.strictEqual(convAfter.length >= 3, true) // init + assistant + user
    const partialAssistant = convAfter.find(c => c.sender === 'Assistant' && c.text.includes('Based on the initial'))
    assert.ok(partialAssistant)
    const finalUserMsg = convAfter[convAfter.length - 1]
    assert.strictEqual(finalUserMsg.sender === 'User', true)
    assert.strictEqual(finalUserMsg.text, 'Wait, focus on memory bandwidth instead.')

    // Verify payload sent to backend
    assert.strictEqual(sentPayloads.length, 1)
    assert.strictEqual(sentPayloads[0].includes('Wait, focus on memory bandwidth instead.'), true)
  })

  it('handles in-flight user barge-in during thinking phase before any response starts', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    let sentPayloads = []
    const mockClient = {
      sendInput: (payload) => { sentPayloads.push(payload) },
    }
    const mockSession = {}
    const mockOrchestrator = {
      getWaitingForUserTasks: () => [],
      getActiveStep: () => null,
    }
    const mockTimers = {}

    controller.init(store, mockClient, mockSession, mockOrchestrator, mockTimers, 'You are Gemma.')

    // Simulate model actively generating thoughts ONLY (thinking channel)
    store.appendActiveThought('Step 1: Reading cache lines.\nStep 2: Checking bandwidth.')
    store.setGenerating(true)

    const mockCtx = {
      setFocus: () => {},
      redraw: () => {},
    }
    // User interrupts mid-thought!
    controller.onSubmitInput(mockCtx, {
      value: 'Hold on, cancel that and run a quick probe instead.',
    })

    // Verify thoughts were flushed and preserved in stream
    const thoughts = store.state.stream.filter(s => s.type === 'thought')
    assert.strictEqual(thoughts.length, 1)
    assert.strictEqual(thoughts[0].content.includes('Step 1: Reading cache lines'), true)

    // Verify conversation has user message
    const conv = store.state.conversation
    const userMsg = conv[conv.length - 1]
    assert.strictEqual(userMsg.sender, 'User')
    assert.strictEqual(sentPayloads[0].includes('cancel that and run a quick probe'), true)
  })

  it('queues in-flight user barge-in into NotificationManager as pending interrupt without premature active assignment', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    let sentPayloads = []
    const mockClient = {
      sendInput: (payload) => { sentPayloads.push(payload) },
    }
    const mockSession = {}
    const mockOrchestrator = {
      getWaitingForUserTasks: () => [],
      getActiveStep: () => null,
    }
    const mockTimers = {}
    const { NotificationManager } = require('../lib/notify')
    const notManager = NotificationManager()

    const fs = require('fs')
    const os = require('os')
    const path = require('path')
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vfs-bargein-test-'))
    const { Vfs } = require('../lib/vfs')
    const vfs = Vfs(tmpDir)

    controller.init(store, mockClient, mockSession, mockOrchestrator, mockTimers, 'You are Gemma.', vfs, notManager)

    // Simulate an active turn in-flight
    const initialTurn = notManager.notify('msg/user/1001.txt', 'Initial task', 'not101')
    notManager.markServicing(initialTurn.id)
    controller.refs.activeTurnNotificationId = initialTurn.id

    store.appendActiveResponse('Starting initial response...')
    store.setGenerating(true)

    const mockCtx = {
      setFocus: () => {},
      redraw: () => {},
    }

    // User barges in mid-generation
    controller.onSubmitInput(mockCtx, {
      value: 'No python please',
    })

    // Previous active turn should be suspended
    const suspended = notManager.getSuspended()
    assert.strictEqual(suspended.length, 1)
    assert.strictEqual(suspended[0].id, 'not101')

    // Active turn notification ID must NOT be prematurely assigned to new item
    assert.strictEqual(controller.refs.activeTurnNotificationId, null)

    // Unserviced queue must contain the new barge-in notification
    const unserviced = notManager.getUnserviced().filter(a => a.id !== controller.refs.activeTurnNotificationId)
    assert.strictEqual(unserviced.length, 1)
    assert.strictEqual(unserviced[0].preview, 'No python please')
    assert.strictEqual(unserviced[0].status, 'PENDING')

    // Input was queued for elastic yield, not dispatched immediately
    assert.strictEqual(sentPayloads.length, 0)

    // Clean up tmpDir
    fs.rmSync(tmpDir, { recursive: true, force: true })
  })
})
