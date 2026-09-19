const readerMachineFactory = () => ({
  name: 'reader',
  initial: 'reading',
  states: {
    reading: async (ctx, { probe }) => {
      const prompt = [
        'Reading Check-In for active channel:',
        '1: Continue reading next segment',
        '2: Pause to make a distilled note of a key finding',
        '3: Pause to ask the user a clarifying question',
        '0: Reading complete, proceed to synthesis and response',
        'Answer with only the index number:',
      ].join('\n')

      const res = await probe(prompt, ['1', '2', '3', '0'], 1)
      const ops = {
        0: { op: 'pull_next_chunk', target: 'reading' },
        1: { op: 'prepare_note', target: 'reflecting' },
        2: { op: 'ask_user', target: 'awaiting_user' },
        3: { op: 'finalize_reading', target: 'synthesizing' },
      }
      return ops[res.winningIdx] || ops[0]
    },
    reflecting: async (ctx, { probe }) => {
      const prompt = 'Distill the key takeaway from the recent reading segment in 1 to 2 sentences:'
      const res = await probe(prompt, [], 32)
      return { op: 'save_note', target: 'reading', note: res.text }
    },
    awaiting_user: async (ctx, { probe }) => {
      const prompt = [
        'Awaiting clarification from the user:',
        '1: Wait for user input',
        '2: Resume reading with best guess',
        'Answer with only the index number:',
      ].join('\n')
      const res = await probe(prompt, ['1', '2'], 1)
      return res.winningIdx === 0
        ? { op: 'yield', target: 'awaiting_user' }
        : { op: 'pull_next_chunk', target: 'reading' }
    },
    synthesizing: async (ctx, { probe }) => {
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
  readerMachineFactory,
  ReaderMachine: readerMachineFactory(),
}
