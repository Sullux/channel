const { Afsm, chatMachineFactory } = require('../../afsm')

const sendTurnWithGate = (refs, payloadText, val, ctx) => {
  if (typeof refs.client?.probe !== 'function') {
    refs.client.sendInput(payloadText, { reason: true })
    return Promise.resolve()
  }

  const afsm = Afsm({
    machine: chatMachineFactory(),
    client: refs.client,
    initialContext: { val },
  })

  return afsm.step()
    .then((stepResult) => {
      const isDirect = stepResult?.op === 'direct_response'
      const meta = stepResult?.transition?.meta || {}
      const confPct = meta.confidence != null ? (meta.confidence * 100).toFixed(1) : '?'
      if (isDirect) {
        refs.store?.addStreamEntry({
          type: 'notice',
          title: '⚡ THINKING GATE',
          content: `Immediate response chosen (${confPct}% confidence >= 80%). Bypassing reasoning.`,
        })
        refs.client.sendInput(payloadText, { direct: true })
      } else {
        const reason = meta.winningIdx === 1 ? 'essential reasoning' : `confidence ${confPct}% < 80% threshold`
        refs.store?.addStreamEntry({
          type: 'notice',
          title: '🧠 THINKING GATE',
          content: `Deliberate reasoning engaged (${reason}).`,
        })
        refs.client.sendInput(payloadText, { reason: true })
      }
      ctx.redraw?.()
    })
    .catch(() => {
      refs.client.sendInput(payloadText, { reason: true })
      ctx.redraw?.()
    })
}

module.exports = {
  sendTurnWithGate,
}
