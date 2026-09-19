const toolRegistryFactory = () => (vfs, cmdRunner, trmManager, notManager, client, orchestrator) => {
  const tools = {
    plan: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const plan = orchestrator.createPlan(args.brief, args.steps)
      client?.sendMemCommit?.()
      return {
        status: 'plan_created',
        id: plan.id,
        brief: plan.brief,
        steps: plan.steps.map((s) => ({ id: s.id, brief: s.brief })),
        active_step: plan.steps[0]?.id || null,
      }
    },

    done: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const res = orchestrator.completeActiveStep(args.summary)
      if (!res) return { status: 'no_active_step' }
      return {
        status: 'step_completed',
        completed_step: res.completedStep.id,
        summary: res.completedStep.summary,
        next_step: res.nextStep?.id || null,
      }
    },

    ask_user: async (args) => {
      if (!orchestrator) return { error: 'No orchestrator available' }
      const brief = args.brief || 'user action required'
      const message = args.message || args.prompt || brief
      const res = orchestrator.askUserActiveStep(brief, message)
      return {
        status: 'waiting_for_user',
        step: res.step?.id || null,
        brief: res.step?.waitingForUser?.brief || brief,
        message: message,
      }
    },

    recall: async (args) => {
      const query = args.query || args.keywords || ''
      const topK = args.top_k || 5
      client?.sendMemQuery?.(query, topK)
      return { status: 'query_submitted', query, topK }
    },

    // Ephemeral Subshell Command
    cmd: async (args) => {
      if (!cmdRunner) return { error: 'No command runner available' }
      const command = args.command || args.cmd || ''
      const remind = args.remind !== undefined ? args.remind : '1m'
      return await cmdRunner.run(command, remind)
    },

    // Terminate Ephemeral Command
    cmd_kill: async (args) => {
      if (!cmdRunner) return { error: 'No command runner available' }
      const id = args.id || ''
      const signal = args.signal || 'SIGTERM'
      return cmdRunner.kill(id, signal)
    },

    // Open Persistent PTY Session
    trm_open: async (args) => {
      if (!trmManager) return { error: 'No terminal manager available' }
      const name = args.name || 'dev'
      const command = args.command || ''
      return trmManager.open(name, command)
    },

    // Close Persistent PTY Session
    trm_close: async (args) => {
      if (!trmManager) return { error: 'No terminal manager available' }
      const name = args.name || ''
      return trmManager.close(name)
    },

    // Send Key to Persistent PTY
    key: async (args) => {
      if (!trmManager) return { error: 'No terminal manager available' }
      const trm = args.trm || args.name || ''
      const name = args.name || args.key || ''
      return trmManager.key(trm, name)
    },

    // Permanent Interrupt Dismissal
    ack: async (args) => {
      if (!notManager) return { error: 'No notification manager available' }
      const id = args.id || ''
      return notManager.ack(id)
    },

    // Temporary Interrupt or Plan Step Suppression
    snooze: async (args) => {
      const id = args.id || ''
      const duration = args.duration || '1m'

      if (id.startsWith('step_')) {
        const res = orchestrator?.deferActiveStep(duration, 'snoozed by model', (step) => {
          const thought = orchestrator?.buildThoughtPrefix('TIMER_WAKE', { step })
          client?.sendInput?.(thought)
        })
        return { status: 'step_snoozed', id, duration, success: Boolean(res) }
      }

      if (notManager) {
        return notManager.snooze(id, duration)
      }
      return { status: 'not_found', id }
    },
  }

  const execute = async (name, args) => {
    let cleanName = (name || '').trim().replace(/^_+|_+$/g, '')
    if (cleanName.includes(':')) {
      cleanName = cleanName.split(':').pop()
    }
    const fn = tools[cleanName] || tools[name]
    if (!fn) {
      const knownTools = Object.keys(tools).join(', ')
      return {
        error: `Unknown tool \`${name}\`. Available tools: ${knownTools}.`,
      }
    }
    try {
      return await fn(args || {})
    } catch (err) {
      return { error: `Tool \`${cleanName}\` execution failed: ${err.message}` }
    }
  }

  return { tools, execute }
}

module.exports = {
  toolRegistryFactory,
  ToolRegistry: toolRegistryFactory(),
}
