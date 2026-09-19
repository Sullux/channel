const { describe, it } = require('node:test')
const assert = require('node:assert')
const { stateStoreFactory } = require('../lib/ui/state')

describe('UI StateStore', () => {
  it('initializes with default conversation and stream items', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    assert.strictEqual(store.state.conversation.length, 1)
    assert.strictEqual(store.state.stream.length, 1)
    assert.strictEqual(store.state.mode, 'chat')
    assert.strictEqual(store.state.isEditMode, false)
  })

  it('accumulates and flushes active thoughts into stream', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    store.appendActiveThought('Checking filesystem')
    store.appendActiveThought(' for package managers')
    assert.strictEqual(store.state.activeThought, 'Checking filesystem for package managers')

    store.flushActiveThought()
    assert.strictEqual(store.state.activeThought, '')
    assert.strictEqual(store.state.stream.length, 2)
    assert.strictEqual(store.state.stream[1].type, 'thought')
    assert.strictEqual(store.state.stream[1].content, 'Checking filesystem for package managers')
  })

  it('accumulates and flushes active response into conversation and stream', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    store.appendActiveResponse('System update completed successfully.')
    store.flushActiveResponse()

    assert.strictEqual(store.state.conversation.length, 2)
    assert.strictEqual(store.state.conversation[1].text, 'System update completed successfully.')
    assert.strictEqual(store.state.stream.length, 2)
    assert.strictEqual(store.state.stream[1].type, 'response')
  })

  it('tracks prompt history accurately', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    store.pushHistory('First prompt')
    store.pushHistory('Second prompt')

    assert.strictEqual(store.state.history.length, 2)
    assert.strictEqual(store.state.history[0], 'First prompt')
    assert.strictEqual(store.state.history[1], 'Second prompt')
  })

  it('hydrates conversation and stream from historical stream items', () => {
    const persisted = []
    const StateStore = stateStoreFactory()
    const store = StateStore((item) => persisted.push(item))

    const historyItems = [
      { id: '1', type: 'user', content: 'What is the date?', time: 1000 },
      { id: '2', type: 'thought', content: 'Checking date format', time: 1001 },
      { id: '3', type: 'response', content: 'It is March 3rd.', time: 1002 },
      { id: '4', type: 'ask_user', content: 'Do you want to proceed?', time: 1003 },
    ]

    store.hydrateFromStream(historyItems)

    assert.strictEqual(store.state.stream.length, 4)
    assert.strictEqual(store.state.conversation.length, 3) // user, response, ask_user
    assert.strictEqual(store.state.conversation[0].sender, 'User')
    assert.strictEqual(store.state.conversation[0].text, 'What is the date?')
    assert.strictEqual(store.state.conversation[1].sender, 'Assistant')
    assert.strictEqual(store.state.conversation[1].text, 'It is March 3rd.')
    assert.strictEqual(store.state.conversation[2].waitingUser, true)
    assert.strictEqual(store.state.conversation[2].text, 'Do you want to proceed?')
    assert.strictEqual(persisted.length, 0) // hydration does not re-emit persistence
  })

  it('hydrates user messages cleanly with rawText or strips event headers', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    const historyItems = [
      {
        id: '1',
        type: 'user',
        content: '[Event: not101 | Source: msg/user/1001.txt]\nHello world',
        time: 1000,
      },
      {
        id: '2',
        type: 'user',
        content: '[Event: not102 | Source: msg/user/1002.txt]\nIgnored stream text',
        rawText: 'Explicit clean text',
        time: 1001,
      },
    ]

    store.hydrateFromStream(historyItems)

    assert.strictEqual(store.state.conversation.length, 2)
    assert.strictEqual(store.state.conversation[0].text, 'Hello world')
    assert.strictEqual(store.state.conversation[1].text, 'Explicit clean text')
  })

  it('handles multi-phase thought and response channel transitions', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    // Phase 1: Thought
    store.appendActiveThought('Initial thought')
    assert.strictEqual(store.state.activeThought, 'Initial thought')

    // Transition to Response: flush thought, start response
    store.flushActiveThought()
    store.appendActiveResponse('Part 1 of answer')
    assert.strictEqual(store.state.stream.length, 2) // init + thought
    assert.strictEqual(store.state.stream[1].type, 'thought')
    assert.strictEqual(store.state.activeResponse, 'Part 1 of answer')

    // Transition back to Thought: flush response, start thought
    store.flushActiveResponse()
    store.appendActiveThought('Second mid-turn thought')
    assert.strictEqual(store.state.stream.length, 3) // init + thought + response
    assert.strictEqual(store.state.stream[2].type, 'response')
    assert.strictEqual(store.state.activeThought, 'Second mid-turn thought')

    // Finalize
    store.flushActiveThought()
    assert.strictEqual(store.state.stream.length, 4) // init + thought + response + thought
    assert.strictEqual(store.state.stream[3].type, 'thought')
    assert.strictEqual(store.state.stream[3].content, 'Second mid-turn thought')
  })

  it('selects and toggles expansion on in-progress thought and response', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()

    store.appendActiveThought('Step 1 thinking...')
    assert.strictEqual(store.state.selectedIdx.stream, 1) // Points to live thought
    assert.strictEqual(store.state.activeThoughtExpanded, false)

    store.toggleExpandStreamItem(1)
    assert.strictEqual(store.state.activeThoughtExpanded, true)

    store.toggleExpandStreamItem(1)
    assert.strictEqual(store.state.activeThoughtExpanded, false)

    store.flushActiveThought()
    assert.strictEqual(store.state.activeThoughtExpanded, false)

    store.appendActiveResponse('Streaming answer...')
    assert.strictEqual(store.state.selectedIdx.chat, 1) // Points to live response
    assert.strictEqual(store.state.activeResponseExpanded, false)

    store.toggleExpandStreamItem(2)
    assert.strictEqual(store.state.activeResponseExpanded, true)
  })

  it('inserts stream and conversation entries in strict chronological order', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    const baseTime = Date.now()

    // Add entry at baseTime + 100
    store.addStreamEntry({ type: 'notice', content: 'Notice at 100', time: baseTime + 100 })
    // Add entry at baseTime + 300
    store.addStreamEntry({ type: 'user', content: 'User at 300', time: baseTime + 300 })
    // Add delayed entry with earlier timestamp baseTime + 200 (e.g. flushed thought)
    store.addStreamEntry({ type: 'thought', content: 'Thought at 200', time: baseTime + 200 })

    const contents = store.state.stream.slice(1).map(s => s.content)
    assert.deepStrictEqual(contents, [
      'Notice at 100',
      'Thought at 200',
      'User at 300',
    ])
  })

  it('stages stream entry in pendingInterjection and flushes in causal order', () => {
    const StateStore = stateStoreFactory()
    const store = StateStore()
    const baseTime = Date.now()

    store.appendActiveResponse('Assistant answering first...')
    store.setPendingInterjection(
      { sender: 'User', text: 'User interjecting', time: baseTime + 100 },
      { type: 'user', title: '👤 USER', content: 'User interjecting', time: baseTime + 100 },
    )

    // User message is staged, not yet in conversation or stream
    assert.strictEqual(store.state.conversation.length, 1)
    assert.strictEqual(store.state.stream.length, 1)

    // Flush active response -> assistant flushed first, then user
    store.flushActiveResponse()

    assert.strictEqual(store.state.conversation.length, 3)
    assert.strictEqual(store.state.conversation[1].sender, 'Assistant')
    assert.strictEqual(store.state.conversation[2].sender, 'User')

    assert.strictEqual(store.state.stream.length, 3)
    assert.strictEqual(store.state.stream[1].type, 'response')
    assert.strictEqual(store.state.stream[2].type, 'user')
  })
})
