const {
  formatUserDecisionTurn,
  formatStepTick,
  formatTimerWake,
  formatResumeAfterInterrupt,
} = require('../template')

const parseDurationMs = (str) => {
  if (!str || typeof str !== 'string') return 10000
  const match = str.trim().match(/^(\d+(?:\.\d+)?)\s*(ms|s|m|h)?$/i)
  if (!match) return 10000
  const val = parseFloat(match[1])
  const unit = (match[2] || 's').toLowerCase()
  if (unit === 'ms') return Math.round(val)
  if (unit === 's') return Math.round(val * 1000)
  if (unit === 'm') return Math.round(val * 60 * 1000)
  if (unit === 'h') return Math.round(val * 3600 * 1000)
  return 10000
}

const orchestratorFactory = (now) => (timers) => {
  let planCounter = 1000
  const plans = []
  const stack = []
  let saturated = false

  const setSaturated = (val) => {
    saturated = Boolean(val)
  }

  const isSaturated = () => saturated

  const createPlan = (brief, steps = []) => {
    saturated = false
    planCounter += 1
    const planId = `plan_${planCounter}`
    const stepList = Array.isArray(steps) ? steps : [steps]
    const normSteps = stepList.length > 0 ? stepList : [brief]

    const planObj = {
      id: planId,
      brief: typeof brief === 'string' ? brief : 'Untitled Plan',
      timestamp: now(),
      completed: null,
      status: 'IN_PROGRESS',
      steps: normSteps.map((s, idx) => ({
        id: `step_${planCounter}.${idx + 1}`,
        parentId: planId,
        brief: typeof s === 'string' ? s : (s?.brief || `Step ${idx + 1}`),
        timestamp: now(),
        completed: null,
        status: 'PENDING',
        waitingForUser: null,
      })),
    }

    plans.push(planObj)
    for (let i = planObj.steps.length - 1; i >= 0; i -= 1) {
      stack.push(planObj.steps[i])
    }
    if (stack.length > 0) {
      stack[stack.length - 1].status = 'IN_PROGRESS'
    }
    return planObj
  }

  const getActiveStep = () => {
    for (let i = stack.length - 1; i >= 0; i -= 1) {
      const item = stack[i]
      if (item.status === 'IN_PROGRESS' || item.status === 'WAITING_FOR_USER' || item.status === 'PENDING') {
        if (item.status === 'PENDING') item.status = 'IN_PROGRESS'
        return item
      }
    }
    return null
  }

  const getWaitingForUserTasks = () => {
    const list = []
    for (const p of plans) {
      for (const s of p.steps) {
        if (s.status === 'WAITING_FOR_USER') list.push(s)
      }
    }
    return list
  }

  const completeActiveStep = (summary) => {
    const active = getActiveStep()
    if (!active) return null

    active.status = 'DONE'
    active.completed = now()
    active.summary = summary || active.brief

    const idx = stack.indexOf(active)
    if (idx !== -1) stack.splice(idx, 1)

    const parent = plans.find((p) => p.id === active.parentId)
    if (parent) {
      const allDone = parent.steps.every((s) => s.status === 'DONE')
      if (allDone) {
        parent.status = 'DONE'
        parent.completed = now()
      }
    }

    const next = getActiveStep()
    return { completedStep: active, nextStep: next }
  }

  const deferActiveStep = (durationStr, reason, onWake) => {
    const active = getActiveStep()
    if (!active) return null

    active.status = 'DEFERRED'
    active.deferReason = reason || 'waiting on timer'
    const ms = parseDurationMs(durationStr)
    const sec = Math.max(1, Math.round(ms / 1000))

    if (timers) {
      timers.setTimer(sec, active.deferReason, () => {
        if (active.status === 'DEFERRED') {
          active.status = 'IN_PROGRESS'
          onWake?.(active)
        }
      })
    }
    return { deferredStep: active, durationMs: ms }
  }

  const askUserActiveStep = (brief, message) => {
    const active = getActiveStep()
    if (!active) {
      const plan = createPlan(brief, [brief])
      const step = plan.steps[0]
      step.status = 'WAITING_FOR_USER'
      step.waitingForUser = { brief, message }
      return { step, message }
    }

    active.status = 'WAITING_FOR_USER'
    active.waitingForUser = { brief, message }
    return { step: active, message }
  }

  const autoWrapToolCall = (toolName, snippet) => {
    const brief = `${toolName}: ${snippet || ''}`.trim()
    const plan = createPlan(brief, [brief])
    return plan.steps[0]
  }

  const buildThoughtPrefix = (type, meta = {}) => {
    const waiting = getWaitingForUserTasks()
    const satNotice = saturated
      ? '\n[Working memory capacity threshold reached. Consolidate current progress, state, and remaining milestones using the plan tool now.]\n'
      : ''

    if (type === 'USER_PROMPT') {
      if (waiting.length === 0) {
        return satNotice
      }
      return `${formatUserDecisionTurn(meta.message || '', waiting)}${satNotice}`
    }

    if (type === 'STEP_TICK') {
      const active = meta.step || getActiveStep()
      if (!active) return 'All tasks complete.\n'
      const parent = plans.find((p) => p.id === active.parentId)
      return `${formatStepTick(parent?.id, parent?.brief, active.id, active.brief)}${satNotice}`
    }

    if (type === 'RESUME_AFTER_INTERRUPT') {
      const active = meta.step || getActiveStep()
      if (!active) return 'No pending tasks to resume.\n'
      return formatResumeAfterInterrupt(active.id, active.brief)
    }

    if (type === 'TIMER_WAKE') {
      const active = meta.step
      return formatTimerWake(active?.id, active?.deferReason)
    }

    return ''
  }

  return {
    plans,
    stack,
    setSaturated,
    isSaturated,
    createPlan,
    getActiveStep,
    getWaitingForUserTasks,
    completeActiveStep,
    deferActiveStep,
    askUserActiveStep,
    autoWrapToolCall,
    buildThoughtPrefix,
  }
}

module.exports = {
  parseDurationMs,
  orchestratorFactory,
  Orchestrator: orchestratorFactory(Date.now),
}
