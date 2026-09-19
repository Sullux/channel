const { refs, formatTimestamp } = require('./state')

const getItemClasses = (type) => {
  if (type === 'thought') return { itemClass: 'streamThought', headerClass: 'streamThoughtHeader' }
  if (type === 'tool_call') return { itemClass: 'streamToolCall', headerClass: 'streamToolCallHeader' }
  if (type === 'tool_result') return { itemClass: 'streamToolResult', headerClass: 'streamToolResultHeader' }
  if (type === 'user') return { itemClass: 'streamUser', headerClass: 'streamUserHeader' }
  if (type === 'notice') return { itemClass: 'streamNotice', headerClass: 'streamNoticeHeader' }
  return { itemClass: 'streamDefault', headerClass: 'streamDefaultHeader' }
}

const getCardHeight = (item) => {
  const lines = (item.content || '').split('\n')
  return (!item.expanded && lines.length > 3) ? 5 : Math.max(2, lines.length + 1)
}

const getStreamScroll = () => {
  if (!refs.store) return 0
  if (refs.store.state.stickyScroll.stream) return Infinity
  const selIdx = refs.store.state.selectedIdx.stream
  let offset = 0
  for (let i = 0; i < selIdx && i < refs.store.state.stream.length; i++) {
    offset += getCardHeight(refs.store.state.stream[i])
  }
  return offset
}

const buildCardItem = (item, isSel) => {
  const { itemClass, headerClass } = getItemClasses(item.type)
  const lines = (item.content || '').split('\n')
  const isMulti = lines.length > 3
  const isExpanded = Boolean(item.expanded)
  const hint = isMulti && !isExpanded ? `  (↑ ${lines.length - 3} more)` : ''
  const hintAction = isSel && isMulti ? (isExpanded ? '  (x to collapse)' : '  (x to expand)') : ''
  const content = isMulti && !isExpanded ? lines.slice(-3).join('\n') : (item.content || '')

  return {
    type: 'streamCard',
    itemClass,
    selectorClass: isSel ? 'selectorActive' : 'selectorInactive',
    selector: isSel ? '▶ ' : '  ',
    timestamp: `[${formatTimestamp(item.time)}] `,
    headerClass,
    title: item.title,
    hint,
    hintAction,
    content,
  }
}

const buildLiveCard = (title, type, content, isSel, isExp, time) => {
  const { itemClass, headerClass } = getItemClasses(type)
  return {
    type: 'streamCard',
    itemClass,
    selectorClass: isSel ? 'selectorActive' : 'selectorInactive',
    selector: isSel ? '▶ ' : '  ',
    timestamp: `[${formatTimestamp(time || Date.now())}] `,
    headerClass,
    title,
    hint: '',
    hintAction: isSel ? (isExp ? '  (x to collapse)' : '  (x to expand)') : '',
    content,
  }
}

const getStreamNodes = () => {
  if (!refs.store) return []
  const { stream, mode, isEditMode, selectedIdx, activeThought, activeResponse } = refs.store.state
  const isStreamFocused = mode === 'stream' && !isEditMode
  const selIdx = selectedIdx.stream

  const nodes = stream.map((item, idx) =>
    buildCardItem(item, isStreamFocused && selIdx === idx),
  )

  let curIdx = stream.length
  if (activeThought) {
    const isSel = isStreamFocused && curIdx === selIdx
    const { activeThoughtTime, activeThoughtExpanded } = refs.store.state
    nodes.push(buildLiveCard('💭 THOUGHT', 'thought', activeThought, isSel, activeThoughtExpanded, activeThoughtTime))
    curIdx++
  }

  if (activeResponse) {
    const isSel = isStreamFocused && curIdx === selIdx
    const { activeResponseTime, activeResponseExpanded } = refs.store.state
    nodes.push(buildLiveCard('🤖 ASSISTANT', 'response', activeResponse, isSel, activeResponseExpanded, activeResponseTime))
  }

  return nodes
}

module.exports = {
  getStreamScroll,
  getStreamNodes,
  getItemClasses,
}
