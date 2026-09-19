const { refs } = require('./state')

const getLayoutTier = (cols) => {
  const w = cols || refs.store?.state?.dimensions?.cols || process.stdout?.columns || 100
  return w >= 160 ? 1 : w >= 120 ? 2 : 3
}

const isClipped = (cols, rows) => {
  const c = cols || refs.store?.state?.dimensions?.cols || process.stdout?.columns || 100
  const r = rows || refs.store?.state?.dimensions?.rows || process.stdout?.rows || 30
  return c < 60 || r < 30
}

const getClippedBanner = () => ' ⚠️  TERMINAL TOO SMALL (Minimum 60x30 required) — CONTENT CLIPPED '

const isConversationVisible = () => getLayoutTier() <= 2 || (refs.store?.state?.mode || 'chat') === 'chat'

const getConversationWidth = () => {
  const tier = getLayoutTier()
  const mode = refs.store?.state?.mode || 'chat'
  return tier === 1 ? '40%' : tier === 2 ? (mode === 'plan' ? '60%' : '50%') : (mode === 'chat' ? '100%' : '0%')
}

const isStreamVisible = () => {
  const tier = getLayoutTier()
  const mode = refs.store?.state?.mode || 'chat'
  return tier === 1 ? true : tier === 2 ? mode !== 'plan' : mode === 'stream'
}

const getStreamWidth = () => {
  const tier = getLayoutTier()
  const mode = refs.store?.state?.mode || 'chat'
  return tier === 1 ? '40%' : tier === 2 ? (mode === 'plan' ? '0%' : '50%') : (mode === 'stream' ? '100%' : '0%')
}

const getStreamMargin = () => (!isStreamVisible() || !isConversationVisible() ? 0 : { left: 1 })

const isPlanVisible = () => getLayoutTier() === 1 || (refs.store?.state?.mode || 'chat') === 'plan'

const getPlanWidth = () => {
  const tier = getLayoutTier()
  const mode = refs.store?.state?.mode || 'chat'
  return tier === 1 ? '20%' : tier === 2 ? (mode === 'plan' ? '40%' : '0%') : (mode === 'plan' ? '100%' : '0%')
}

const getPlanMargin = () => (!isPlanVisible() || (!isConversationVisible() && !isStreamVisible()) ? 0 : { left: 1 })

const getStatusText = () => {
  if (!refs.store) return 'Status: Ready'
  const state = refs.store.state
  const paused = state.isPaused ? ' [⏸️ PAUSED]' : ''
  const pending = refs.notManager?.getUnserviced?.()?.length || 0
  return ` ${state.status}${paused}${pending > 0 ? ` | [🔔 ${pending} ALERTS]` : ''}`
}

const SHORTCUTS = {
  paused: ' [r] Resume Inference | [Ctrl+Q] Quit',
  edit: ' [Enter] Send | [Esc] Normal Mode | [Shift+Enter] Newline | [Ctrl+C] Clear',
  chat: ' [Enter] Type | [j/k] Scroll | [s] Stream | [d] Plan | [c] Copy | [Esc] Pause | [Ctrl+Q] Quit',
  stream: ' [j/k] Scroll | [x] Expand/Collapse | [a] Chat | [d] Plan | [c] Copy | [Esc] Chat',
  plan: ' [j/k] Navigate Tasks | [a] Chat | [s] Stream | [c] Copy | [Esc] Chat',
}

const getShortcutsText = () => {
  if (isClipped()) return getClippedBanner()
  if (!refs.store) return ' [Enter] Type  [a] Chat  [s] Stream  [d] Plan  [Ctrl+Q] Quit'
  const s = refs.store.state
  const key = s.isPaused ? 'paused' : s.isEditMode ? 'edit' : s.mode
  return SHORTCUTS[key] || ' [Enter] Type | [a] Chat | [s] Stream | [d] Plan | [Ctrl+Q] Quit'
}

module.exports = {
  getLayoutTier,
  isClipped,
  getClippedBanner,
  isConversationVisible,
  getConversationWidth,
  isStreamVisible,
  getStreamWidth,
  getStreamMargin,
  isPlanVisible,
  getPlanWidth,
  getPlanMargin,
  getStatusText,
  getShortcutsText,
}
