const { MAX_STREAM_ITEMS, MAX_CONVERSATION_ITEMS } = require('./model')

const insertMonotonic = (list, item) => {
  if (list.length === 0 || item.time >= (list[list.length - 1].time || 0)) {
    list.push(item)
    return
  }
  let idx = list.length - 1
  while (idx >= 0 && (list[idx].time || 0) > item.time) idx--
  list.splice(idx + 1, 0, item)
}

const createConversationAppender = (state) => (msg) => {
  const item = {
    id: msg.id || `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    time: msg.time || Date.now(),
    ...msg,
  }
  insertMonotonic(state.conversation, item)
  if (state.conversation.length > MAX_CONVERSATION_ITEMS) state.conversation.shift()
  if (state.stickyScroll.chat) state.selectedIdx.chat = state.conversation.length - 1
}

const createStreamAppender = (state, onStreamItem) => (entry, shouldPersist = true) => {
  const item = {
    id: entry.id || `str-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    time: entry.time || Date.now(),
    expanded: false,
    ...entry,
  }
  insertMonotonic(state.stream, item)
  if (state.stream.length > MAX_STREAM_ITEMS) state.stream.shift()
  if (state.stickyScroll.stream) state.selectedIdx.stream = state.stream.length - 1
  if (shouldPersist && onStreamItem) onStreamItem(item)
}

const createHydrator = (state, addConversationMessage, addStreamEntry) => (items) => {
  if (!items?.length) return
  state.conversation = []
  state.stream = []
  const sorted = [...items].sort((a, b) => (a.time || 0) - (b.time || 0))
  for (const it of sorted) addStreamEntry(it, false)

  const convItems = sorted.filter((it) => ['user', 'response', 'ask_user'].includes(it.type))
  for (const it of convItems) {
    if (it.type === 'user') {
      const userText = it.rawText || (it.content ? it.content.replace(/^\[Event: [^\]]+ \| Source: [^\]]+\]\r?\n/, '') : '')
      addConversationMessage({ sender: 'User', text: userText, time: it.time, id: it.id })
    } else if (it.type === 'response') {
      addConversationMessage({ sender: 'Assistant', text: it.content, time: it.time, id: it.id })
    } else if (it.type === 'ask_user') {
      addConversationMessage({ sender: 'Assistant', text: it.content, time: it.time, id: it.id, waitingUser: true })
    }
  }
  if (state.conversation.length === 0) {
    addConversationMessage({ sender: 'Assistant', text: 'Cognitive Engine Ready. Press Enter to type a prompt or task.' })
  }
}

module.exports = {
  createConversationAppender,
  createStreamAppender,
  createHydrator,
}
