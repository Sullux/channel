const MAX_STREAM_ITEMS = 1000
const MAX_CONVERSATION_ITEMS = 500

const cleanThought = (raw) => {
  const t = (raw || '').trim()
  return t.startsWith('thought') ? t.slice(7).trim() : t
}

const createInitialState = (now = Date.now) => ({
  conversation: [
    {
      id: 'init',
      sender: 'Assistant',
      text: 'Cognitive Engine Ready. Press Enter to type a prompt or task.',
      time: now(),
    },
  ],
  stream: [
    {
      id: 's-init',
      type: 'system',
      title: '⚙ SYSTEM',
      content: 'Vulkan RDNA 3.5 Engine connected. Task stack active.',
      time: now(),
      expanded: false,
    },
  ],
  activeThought: '',
  activeThoughtTime: 0,
  activeThoughtExpanded: false,
  activeResponse: '',
  activeResponseTime: 0,
  activeResponseExpanded: false,
  pendingInterjection: null,
  status: '[Engine] Initializing GPU compute & pre-caching working state...',
  isGenerating: false,
  isPaused: false,
  mode: 'chat',
  isEditMode: false,
  overlay: null,
  dimensions: { cols: 100, rows: 30 },
  history: [],
  historyIdx: -1,
  cachedDraft: '',
  stickyScroll: { chat: true, stream: true },
  selectedIdx: { chat: 0, stream: 0, plan: 0 },
})

module.exports = {
  MAX_STREAM_ITEMS,
  MAX_CONVERSATION_ITEMS,
  cleanThought,
  createInitialState,
}
