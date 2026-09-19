const { cleanThought } = require('./model')

const createStreamingOps = (state, addConversationMessage, addStreamEntry) => {
  const appendActiveThought = (chunk) => {
    if (!state.activeThought) state.activeThoughtTime = Date.now()
    state.activeThought += chunk
    if (state.stickyScroll.stream) state.selectedIdx.stream = state.stream.length
  }

  const flushActiveThought = () => {
    const content = cleanThought(state.activeThought)
    const time = state.activeThoughtTime || Date.now()
    state.activeThought = ''
    state.activeThoughtTime = 0
    state.activeThoughtExpanded = false
    if (content) addStreamEntry({ type: 'thought', title: '💭 THOUGHT', content, time })
  }

  const appendActiveResponse = (chunk) => {
    if (!state.activeResponse) state.activeResponseTime = Date.now()
    state.activeResponse += chunk
    if (state.stickyScroll.chat) state.selectedIdx.chat = state.conversation.length
    if (state.stickyScroll.stream) {
      const th = cleanThought(state.activeThought)
      state.selectedIdx.stream = state.stream.length + (th ? 1 : 0)
    }
  }

  const setPendingInterjection = (msg, streamEntry) => {
    state.pendingInterjection = { ...msg, streamEntry }
  }

  const flushPendingInterjection = () => {
    if (!state.pendingInterjection) return
    const { streamEntry, ...msg } = state.pendingInterjection
    state.pendingInterjection = null
    addConversationMessage(msg)
    if (streamEntry) addStreamEntry(streamEntry)
  }

  const clearActiveResponse = () => {
    state.activeResponse = ''
    state.activeResponseTime = 0
    state.activeResponseExpanded = false
  }

  const flushActiveResponse = () => {
    if (state.activeResponse) {
      const text = state.activeResponse
      const time = state.activeResponseTime || Date.now()
      clearActiveResponse()
      addConversationMessage({ sender: 'Assistant', text: text.trim(), time })
      addStreamEntry({ type: 'response', title: '🤖 ASSISTANT', content: text.trim(), time })
    }
    flushPendingInterjection()
  }

  return {
    appendActiveThought,
    flushActiveThought,
    appendActiveResponse,
    setPendingInterjection,
    flushPendingInterjection,
    clearActiveResponse,
    flushActiveResponse,
  }
}

module.exports = { createStreamingOps }
