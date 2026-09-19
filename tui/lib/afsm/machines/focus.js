const focusMachineFactory = () => ({
  name: 'focus',
  initial: 'evaluate_interrupt',
  states: {
    evaluate_interrupt: async (ctx, { probe }) => {
      const prompt = [
        'Focus Evaluation:',
        `An event has arrived on background channel "${ctx.incomingChannel}":`,
        `"${ctx.preview}"`,
        '',
        `Active focused channel is: "${ctx.focusedChannel}"`,
        '',
        `Should focus immediately switch to "${ctx.incomingChannel}"?`,
        '0: No, remain on current focus (queue event in background alerts)',
        `1: Yes, switch focus immediately to "${ctx.incomingChannel}"`,
        'Decision:',
      ].join('\n')

      const res = await probe(prompt, ['0', '1'], 1)
      const shouldSwitch = res.winningIdx === 1 && res.confidence >= 0.70

      return {
        op: shouldSwitch ? 'switch_focus' : 'bookmark_interrupt',
        target: 'idle',
        meta: {
          winningIdx: res.winningIdx,
          confidence: res.confidence,
          cost: res.costMs,
          entropy: res.entropy,
          shouldSwitch,
        },
      }
    },
    idle: async () => ({
      op: 'yield',
      target: 'idle',
    }),
  },
})

module.exports = {
  focusMachineFactory,
  FocusMachine: focusMachineFactory(),
}
