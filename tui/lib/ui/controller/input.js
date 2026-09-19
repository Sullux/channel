const { refs } = require('./state')
const { formatUserTurn, formatUserDecisionTurn } = require('../../template')
const { sendTurnWithGate } = require('./dispatch')

const stageUserEvent = (val, isGenerating) => {
  if (!refs.vfs) return { turnHeader: '', turnBody: val }
  const saved = refs.vfs.saveUserMessage(val)
  refs.channelManager?.registerChannel({
    id: 'chat/user',
    path: saved.relPath,
    type: 'push',
    cursor: 0,
    isFocused: true,
  })
  const chunk = refs.channelManager?.readChunk('chat/user', refs.vfs, 512)
  const notItem = refs.notManager?.notify(
    saved.relPath,
    saved.preview,
    saved.id,
    { isTurnContext: true, payload: saved.payload },
  )
  if (isGenerating && refs.activeTurnNotificationId) {
    refs.notManager?.suspend(refs.activeTurnNotificationId)
    refs.activeTurnNotificationId = null
  } else if (!isGenerating && notItem) {
    refs.notManager?.markServicing(notItem.id)
    refs.activeTurnNotificationId = notItem.id
  }
  return {
    turnHeader: `[Event: ${notItem?.id || saved.id} | Source: ${saved.relPath}]\n`,
    turnBody: chunk?.content || saved.payload,
  }
}

const onSubmitInput = (ctx, payload) => {
  const val = payload.value?.trim()
  if (payload.node) { payload.node.value = ''; payload.node.cursor = 0 }
  if (!val || !refs.client) return

  if (refs.store?.state?.activeThought) refs.store.flushActiveThought()

  const isGenResp = Boolean(refs.store?.state?.activeResponse)
  const isGen = Boolean(refs.store?.state?.isGenerating || refs.store?.state?.activeResponse || refs.store?.state?.activeThought)

  const { turnHeader, turnBody } = stageUserEvent(val, isGen)
  const turnContent = `${turnHeader}${turnBody}`
  refs.store?.pushHistory(val)

  const entry = { type: 'user', title: '👤 USER', content: turnContent, rawText: val, time: Date.now() }
  if (isGenResp) {
    refs.store?.setPendingInterjection({ sender: 'User', text: val, time: Date.now() }, entry)
  } else {
    refs.store?.addConversationMessage({ sender: 'User', text: val, waitingEngine: !refs.isEngineReady || undefined })
    refs.store?.addStreamEntry(entry)
  }
  refs.store?.setEditMode(false)
  ctx.setFocus?.(null)

  const alerts = refs.notManager?.formatTurnAlerts?.() || ''
  const waiting = refs.orchestrator?.getWaitingForUserTasks?.() || []
  const text = waiting.length > 0
    ? formatUserDecisionTurn(`${alerts}${turnContent}`, waiting)
    : formatUserTurn(`${alerts}${turnContent}`)

  refs.store?.setGenerating(true)
  if (!refs.isEngineReady) {
    refs.pendingInputTurn = text
    ctx.redraw()
    return
  }

  let promise = null
  if (!isGen || !refs.notManager) {
    promise = sendTurnWithGate(refs, text, val, ctx)
  }
  ctx.redraw?.()
  return promise
}

module.exports = { onSubmitInput }
