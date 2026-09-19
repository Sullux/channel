const { createInitialState, cleanThought } = require('./model')
const { createConversationAppender, createStreamAppender, createHydrator } = require('./items')
const { createStreamingOps } = require('./streaming')

const stateStoreFactory = (now = Date.now) => (onStreamItem) => {
  const state = createInitialState(now)
  const addConversationMessage = createConversationAppender(state)
  const addStreamEntry = createStreamAppender(state, onStreamItem)
  const hydrateFromStream = createHydrator(state, addConversationMessage, addStreamEntry)
  const streamingOps = createStreamingOps(state, addConversationMessage, addStreamEntry)

  const toggleExpandStreamItem = (idx) => {
    const thIdx = cleanThought(state.activeThought) ? state.stream.length : -1
    const respIdx = state.activeResponse ? (state.stream.length + (thIdx !== -1 ? 1 : 0)) : -1
    if (idx === thIdx && thIdx !== -1) {
      state.activeThoughtExpanded = !state.activeThoughtExpanded
      return
    }
    if (idx === respIdx && respIdx !== -1) {
      state.activeResponseExpanded = !state.activeResponseExpanded
      return
    }
    const item = state.stream[idx]
    if (item) item.expanded = !item.expanded
  }

  const pushHistory = (txt) => {
    const trimmed = typeof txt === 'string' ? txt.trim() : ''
    if (!trimmed) return
    state.history.push(trimmed)
    state.historyIdx = state.history.length
    state.cachedDraft = ''
  }

  return {
    state,
    addConversationMessage,
    addStreamEntry,
    hydrateFromStream,
    ...streamingOps,
    setStatus: (s) => { state.status = s },
    setGenerating: (g) => { state.isGenerating = g },
    setPaused: (p) => { state.isPaused = p },
    setMode: (m) => { state.mode = m },
    setEditMode: (e) => { state.isEditMode = e },
    setOverlay: (o) => { state.overlay = o },
    setDimensions: (cols, rows) => { state.dimensions = { cols, rows } },
    toggleExpandStreamItem,
    pushHistory,
  }
}

module.exports = {
  stateStoreFactory,
  StateStore: stateStoreFactory(),
}
