const { KeyHandler } = require('@sullux/tui')
const { refs } = require('./state')
const { getLayoutTier } = require('./layout')
const { copyToClipboard, getSelectedText } = require('./clipboard')

const getCount = (store) => {
  const s = store?.state
  if (s.mode === 'chat') return (s.conversation?.length || 0) + (s.activeResponse ? 1 : 0)
  if (s.mode === 'stream') return (s.stream?.length || 0) + (s.activeThought ? 1 : 0) + (s.activeResponse ? 1 : 0)
  return s.plan?.length || 0
}

const scrollList = (store, delta) => {
  const selKey = store?.state.mode === 'chat' ? 'chat' : 'stream'
  const total = getCount(store)
  const next = (store?.state.selectedIdx[selKey] || 0) + delta
  if (next >= 0 && next < total) {
    store.state.selectedIdx[selKey] = next
    store.state.stickyScroll[selKey] = delta >= 0 && next === total - 1
  }
}

const toggleMode = (ctx, e, target, fallback) => {
  e.stopPropagation()
  refs.store?.setMode(refs.store?.state.mode === target ? fallback : target)
  ctx.redraw()
}

const normalModeRouter = KeyHandler({
  'ctrl+q': () => process.exit(0),
  a: (ctx, e) => { e.stopPropagation(); refs.store?.setMode('chat'); ctx.redraw() },
  s: (ctx, e) => toggleMode(ctx, e, 'stream', getLayoutTier() === 3 ? 'chat' : 'stream'),
  d: (ctx, e) => toggleMode(ctx, e, 'plan', getLayoutTier() >= 2 ? 'chat' : 'plan'),
  enter: (ctx, e) => {
    e.stopPropagation()
    refs.store?.setEditMode(true)
    refs.store?.setMode('chat')
    ctx.setFocus?.('chatInput')
    ctx.redraw()
  },
  escape: (ctx, e) => {
    e.stopPropagation()
    if (getLayoutTier() >= 2 && refs.store?.state.mode !== 'chat') {
      refs.store?.setMode('chat')
    } else {
      const p = !refs.store?.state.isPaused
      refs.store?.setPaused(p)
      if (p) refs.client?.sendAbort?.()
    }
    ctx.redraw()
  },
  r: (ctx, e) => {
    if (refs.store?.state.isPaused) {
      e.stopPropagation()
      refs.store.setPaused(false)
      ctx.redraw()
    }
  },
  x: (ctx, e) => {
    if (refs.store?.state.mode === 'stream') {
      e.stopPropagation()
      refs.store.toggleExpandStreamItem(refs.store.state.selectedIdx.stream)
      ctx.redraw()
    }
  },
  c: (ctx, e) => {
    e.stopPropagation()
    const text = getSelectedText(refs.store?.state)
    if (text) copyToClipboard(text)
  },
  j: (ctx, e) => { e.stopPropagation(); scrollList(refs.store, 1); ctx.redraw() },
  k: (ctx, e) => { e.stopPropagation(); scrollList(refs.store, -1); ctx.redraw() },
  down: (ctx, e) => normalModeRouter(ctx, { ...e, key: 'j' }),
  up: (ctx, e) => normalModeRouter(ctx, { ...e, key: 'k' }),
})

const onGlobalKey = (ctx, event) => {
  if ((event.ctrl && event.key === 'q') || event.stroke === 'ctrl+q') process.exit(0)
  if (refs.store?.state.isEditMode) {
    if (event.key === 'escape') {
      event.stopPropagation()
      refs.store.setEditMode(false)
      ctx.setFocus?.(null)
      ctx.redraw()
    }
    return
  }
  if ((event.ctrl && event.key === 'c') || event.stroke === 'ctrl+c') process.exit(0)
  normalModeRouter(ctx, event)
  event.stopPropagation()
}

module.exports = { onGlobalKey }
