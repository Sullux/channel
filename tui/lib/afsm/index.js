const { chatMachineFactory, ChatMachine } = require('./machines/chat')
const { focusMachineFactory, FocusMachine } = require('./machines/focus')
const { readerMachineFactory, ReaderMachine } = require('./machines/reader')

const afsmFactory =
  () =>
  ({ machine, definition, client, initialContext = {}, ops = {}, onTransition }) => {
    const spec = machine || definition
    let currentState = spec.initial || Object.keys(spec.states)[0]
    let context = { ...initialContext }

    const getState = () => ({
      currentState,
      context: { ...context },
    })

    const setContext = (updates) => {
      context = { ...context, ...updates }
      return { ...context }
    }

    const probeHelper = (prompt, candidates = [], maxTokens = 1, minCertainty = 0.6) =>
      client.probe(prompt, candidates, maxTokens, minCertainty)

    const step = async (overridePrompt) => {
      const stateDef = spec.states?.[currentState]
      if (!stateDef) throw new Error(`Unknown A-FSM state: ${currentState}`)

      let selectedOp = null
      let nextState = currentState
      let probeRes = null
      let transition = null

      if (typeof stateDef === 'function') {
        transition = await stateDef(context, { probe: probeHelper, client, overridePrompt })
        selectedOp = transition?.op || null
        nextState = transition?.target || currentState
        if (transition && typeof transition === 'object') {
          const { op, target, meta, ...rest } = transition
          context = { ...context, ...rest }
        }
      } else if (stateDef.probe) {
        const probeDef = stateDef.probe
        const rawPrompt = overridePrompt || probeDef.prompt
        const prompt =
          typeof rawPrompt === 'function'
            ? rawPrompt(context)
            : typeof rawPrompt === 'string'
            ? rawPrompt.replace(/\{\{(\w+)\}\}/g, (_, k) => context[k] ?? '')
            : rawPrompt

        const resolvedCands =
          typeof probeDef.candidates === 'function'
            ? probeDef.candidates(context)
            : probeDef.candidates === 'context' || probeDef.candidates === true
            ? context.candidates || []
            : probeDef.candidates || []
        const candEntries = Array.isArray(resolvedCands)
          ? resolvedCands.map((c) => [c.key || c.id || String(c), c])
          : Object.entries(resolvedCands)
        const candidates = candEntries.map(([k]) => k)
        const maxTokens = probeDef.maxTokens || (candidates.length > 0 ? 1 : 14)
        const minCertainty = probeDef.minCertainty || 0.6

        probeRes = await probeHelper(prompt, candidates, maxTokens, minCertainty)
        selectedOp = probeDef.op
        nextState = probeDef.target || currentState

        if (candEntries.length > 0 && probeRes) {
          const winningEntry = candEntries[probeRes.winningIdx]
          const candConfig = winningEntry ? winningEntry[1] : null
          if (candConfig) {
            selectedOp = candConfig.op || selectedOp
            nextState = candConfig.target || nextState
          }
        }
      }

      if (selectedOp && typeof ops[selectedOp] === 'function') {
        const opResult = await ops[selectedOp](context, transition || probeRes)
        if (opResult && typeof opResult === 'object') {
          context = { ...context, ...opResult }
        }
      }

      const prev = currentState
      currentState = nextState

      if (typeof onTransition === 'function') {
        onTransition({ from: prev, to: currentState, op: selectedOp, probeRes, transition, context })
      }

      return {
        from: prev,
        to: currentState,
        op: selectedOp,
        probeRes,
        transition,
        context: { ...context },
      }
    }

    return {
      getState,
      setContext,
      step,
    }
  }

module.exports = {
  afsmFactory,
  Afsm: afsmFactory(),
  chatMachineFactory,
  ChatMachine,
  focusMachineFactory,
  FocusMachine,
  readerMachineFactory,
  ReaderMachine,
}
