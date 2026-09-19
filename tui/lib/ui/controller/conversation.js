const { refs, formatTimestamp } = require('./state')

const getConversationScroll = () => {
  if (!refs.store) return 0
  if (refs.store.state.stickyScroll.chat) return Infinity
  const selIdx = refs.store.state.selectedIdx.chat
  let offset = 0
  for (let i = 0; i < selIdx && i < refs.store.state.conversation.length; i++) {
    const msg = refs.store.state.conversation[i]
    const lineCount = (msg.text || '').split('\n').length
    offset += Math.max(1, lineCount) + 1
  }
  return offset
}

const getConversationNodes = () => {
  if (!refs.store) return []
  const { conversation, mode, isEditMode, selectedIdx, activeResponse, pendingInterjection } = refs.store.state
  const isChatFocused = mode === 'chat' && !isEditMode

  const nodes = conversation.map((msg, idx) => {
    const isUser = msg.sender === 'User'
    const isSelected = isChatFocused && selectedIdx.chat === idx
    const timeStr = formatTimestamp(msg.time)

    const itemClass = msg.waitingEngine ? 'waitingEngineCard'
      : msg.waitingUser ? 'waitingUserCard'
      : isUser ? 'userCard'
      : 'assistantCard'

    const headerClass = msg.waitingEngine || msg.waitingUser ? 'waitingHeader'
      : isUser ? 'userHeader'
      : 'assistantHeader'

    const headerPrefix = msg.waitingEngine || msg.waitingUser ? '⏳ '
      : isUser ? '👤 '
      : '🤖 '

    const senderText = msg.waitingEngine ? 'User (staging for engine ready)' : msg.sender

    return {
      type: 'conversationCard',
      itemClass,
      selectorClass: isSelected ? 'selectorActive' : 'selectorInactive',
      selector: isSelected ? '▶ ' : '  ',
      timestamp: `[${timeStr}] `,
      headerClass,
      header: `${headerPrefix}${senderText}: `,
      text: msg.text || '',
    }
  })

  if (activeResponse) {
    const isSelected = isChatFocused && selectedIdx.chat === conversation.length
    const timeStr = formatTimestamp(refs.store.state.activeResponseTime || Date.now())
    nodes.push({
      type: 'conversationCard',
      itemClass: 'assistantCard',
      selectorClass: isSelected ? 'selectorActive' : 'selectorInactive',
      selector: isSelected ? '▶ ' : '  ',
      timestamp: `[${timeStr}] `,
      headerClass: 'assistantHeader',
      header: '🤖 Assistant: ',
      text: `${activeResponse} ▍`,
    })
  }

  if (pendingInterjection) {
    const timeStr = formatTimestamp(pendingInterjection.time || Date.now())
    nodes.push({
      type: 'conversationCard',
      itemClass: 'inFlightCard',
      selectorClass: 'selectorInactive',
      selector: '  ',
      timestamp: `[${timeStr}] `,
      headerClass: 'inFlightHeader',
      header: '👤 User (in-flight): ',
      text: pendingInterjection.text || '',
    })
  }

  return nodes
}

module.exports = {
  getConversationScroll,
  getConversationNodes,
}
