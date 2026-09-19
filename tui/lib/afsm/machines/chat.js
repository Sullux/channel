const chatMachineFactory = () => ({
  name: 'chat',
  initial: 'responding',
  states: {
    responding: async (ctx, { probe }) => {
      const prompt = [
        "Regarding the user's message:",
        '0: I have an immediate answer',
        '1: I need to think',
        'Decision:',
      ].join('\n')

      const res = await probe(prompt, ['0', '1'], 1)
      const isDirect = res.winningIdx === 0 && res.confidence >= 0.80

      return {
        op: isDirect ? 'direct_response' : 'think',
        target: 'idle',
        meta: {
          winningIdx: res.winningIdx,
          confidence: res.confidence,
          cost: res.costMs,
          entropy: res.entropy,
          isDirect,
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
  chatMachineFactory,
  ChatMachine: chatMachineFactory(),
}
