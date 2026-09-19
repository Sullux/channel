const fs = require('fs')
const path = require('path')
const yaml = require('js-yaml')
const { Tui } = require('@sullux/tui')
const { Client } = require('./lib/client')
const { TerminalSession } = require('./lib/terminal/session')
const { TimerManager } = require('./lib/timers')
const { Orchestrator } = require('./lib/orchestrator')
const { Vfs } = require('./lib/vfs')
const { CommandRunner } = require('./lib/cmd')
const { TerminalManager } = require('./lib/terminal/manager')
const { NotificationManager } = require('./lib/notify')
const { ToolRegistry } = require('./lib/tools')
const { ToolParser } = require('./lib/tools/parser')
const { StreamLog } = require('./lib/storage')
const { stateStoreFactory } = require('./lib/ui/state')
const { TaskManager, taskManagerFactory } = require('./lib/tasks')
const { ChannelManager } = require('./lib/channels')
const {
  STOP_END_OF_TURN,
  STOP_ELASTIC_YIELD,
  STOP_TOOL_CALL,
  READ_STATUS_CHUNK,
  READ_STATUS_EOF,
  BACKLOG_ACTION_ACK,
  BACKLOG_ACTION_RESUME,
  BACKLOG_ACTION_SNOOZE,
} = require('./lib/protocol/constants')
const { formatNotificationInterrupt, formatBacklogResumeNudge } = require('./lib/template')
const controller = require('./lib/ui/controller')
const { MarkdownControl } = require('./lib/ui/markdown')

const STATUS_NAMES = [
  'Idle',
  'Encoding prompt...',
  'Generating...',
  'Searching memory...',
  'Consolidating diffs...',
]

const parseCliArgs = (argv) => {
  let configPath = null
  let debug = false
  const extraArgs = []
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--config' || arg === '-c') {
      configPath = argv[++i]
    } else if (arg.startsWith('--config=')) {
      configPath = arg.slice('--config='.length)
    } else if (arg === '--debug' || arg === '-d') {
      debug = true
    } else {
      extraArgs.push(arg)
    }
  }
  return { configPath, debug, extraArgs }
}

const resolveConfigPath = (explicitPath) => {
  if (explicitPath) {
    return path.resolve(process.cwd(), explicitPath)
  }
  const cwdConfig = path.resolve(process.cwd(), './config.json')
  if (fs.existsSync(cwdConfig)) {
    return cwdConfig
  }
  return path.resolve(__dirname, './config.json')
}

const loadConfig = (explicitPath) => {
  const cfgPath = resolveConfigPath(explicitPath)
  const baseDir = path.dirname(cfgPath)

  let raw = undefined
  if (fs.existsSync(cfgPath)) {
    try {
      raw = JSON.parse(fs.readFileSync(cfgPath, 'utf-8'))
    } catch (_) {}
  }
  const base = Object.assign({
    modelPath: '../../gemma-4-12B-it',
    memoryDir: './.memory',
    filesystemRoot: './.agent',
    promptPath: './PROMPT.md',
    extraArgs: ['--gpu', '--q4'],
    runtime: {
      thinkingBudget: 256,
      maxTokens: 1024,
      temp: 0.7,
      topP: 0.95,
      minP: 0.05,
      repeatPenalty: 1.1,
      repeatLastN: 64,
      frequencyPenalty: 0.1,
      presencePenalty: 0.1,
      qThresh: 0.001,
    }
  }, raw)

  return {
    ...base,
    modelPath: base.modelPath ? path.resolve(baseDir, base.modelPath) : undefined,
    memoryDir: base.memoryDir ? path.resolve(baseDir, base.memoryDir) : undefined,
    filesystemRoot: base.filesystemRoot ? path.resolve(baseDir, base.filesystemRoot) : undefined,
    promptPath: base.promptPath ? path.resolve(baseDir, base.promptPath) : undefined,
  }
}

const loadSystemConfig = (promptPath) => {
  const kernelPath = path.resolve(__dirname, './PROMPT_KERNEL.md')
  const userPromptPath = promptPath || path.resolve(__dirname, './PROMPT.md')

  const userPrompt = fs.existsSync(userPromptPath)
    ? fs.readFileSync(userPromptPath, 'utf-8').trim()
    : ''

  let directives = ''
  const tools = []

  if (fs.existsSync(kernelPath)) {
    const kernelRaw = fs.readFileSync(kernelPath, 'utf-8').trim()
    const match = kernelRaw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/)
    if (match) {
      const parsed = yaml.load(match[1]) || {}
      directives = match[2].trim()
      if (parsed.tools) {
        for (const [name, def] of Object.entries(parsed.tools)) {
          const props = {}
          const req = []
          for (const [pName, pDef] of Object.entries(def.parameters || {})) {
            props[pName] = {
              type: pDef.type || 'string',
              description: pDef.description || '',
            }
            if (pDef.required) req.push(pName)
          }
          tools.push({
            name,
            description: def.description || '',
            parameters: {
              type: 'object',
              properties: props,
              required: req,
            },
          })
        }
      }
    } else {
      directives = kernelRaw
    }
  }

  const instructions = [userPrompt, directives].filter(Boolean).join('\n\n')
  return { instructions, tools }
}

const main = () => {
  const { configPath, debug: cliDebug, extraArgs: cliExtraArgs } = parseCliArgs(process.argv.slice(2))
  const config = loadConfig(configPath)
  const isDebug = cliDebug || Boolean(config.debug || process.env.DEBUG_TUI)
  const extraArgs = [...(cliExtraArgs.length > 0 ? cliExtraArgs : config.extraArgs || [])]
  const modelPath = config.modelPath
  const systemConfig = loadSystemConfig(config.promptPath)
  const systemPrompt = systemConfig.instructions

  const debugLogPath = path.resolve(process.cwd(), 'debug.log')
  const logDebug = (msg) => {
    if (!isDebug) return
    try {
      fs.appendFileSync(debugLogPath, `[${new Date().toISOString()}] ${msg}\n`)
    } catch (_) {}
  }

  let streamLog = StreamLog(undefined)
  let snapshotPath = null
  if (config.memoryDir) {
    const memDir = config.memoryDir
    fs.mkdirSync(memDir, { recursive: true })
    const episodicMemPath = path.join(memDir, '.episodic.mem')
    if (!extraArgs.includes('--memory') && !extraArgs.includes('--storage')) {
      extraArgs.push('--memory', episodicMemPath)
    }
    snapshotPath = path.join(memDir, '.snapshot.bin')
    tasksPath = path.join(memDir, '.tasks.json')
    streamLog = StreamLog(memDir)
  }

  const StateStore = stateStoreFactory()
  const store = StateStore(streamLog.append)
  const updateDimensions = () => {
    const cols = process.stdout.columns || 100
    const rows = process.stdout.rows || 30
    store.setDimensions(cols, rows)
  }
  updateDimensions()
  process.stdout.on('resize', () => {
    updateDimensions()
    requestRedraw()
  })

  const tail = streamLog.loadTail()
  if (tail.items?.length > 0) {
    store.hydrateFromStream(tail.items)
  }

  const timers = TimerManager()
  const notManager = NotificationManager(timers)

  const vfs = Vfs(config.filesystemRoot)
  vfs.init()

  const cmdRunner = CommandRunner(vfs, timers, (n) => {
    notManager.notify(n.log || n.cmdId, n.preview, n.cmdId)
    requestRedraw()
  })

  const trmManager = TerminalManager(vfs, (n) => {
    notManager.notify(n.log, n.preview, n.name)
    requestRedraw()
  })

  const session = TerminalSession({ rows: 30, cols: 100 })
  const client = Client({
    binaryPath: path.resolve(__dirname, '../zig-out/bin/infer'),
    modelPath,
    extraArgs,
  })
  const orchestrator = Orchestrator(timers)
  let initialTasks = []
  if (tasksPath && fs.existsSync(tasksPath)) {
    try {
      initialTasks = JSON.parse(fs.readFileSync(tasksPath, 'utf-8'))
    } catch (_) {}
  } else if (streamLog.enabled) {
    const allItems = streamLog.loadAll()
    const foundTasks = new Map()
    for (const it of allItems) {
      if (it.type === 'system' && it.title === '🏷 TASK TITLE' && it.content) {
        const m = it.content.match(/^Task #(\d+) titled: "(.*)"/)
        if (m) {
          const tid = parseInt(m[1], 10)
          const title = m[2]
          foundTasks.set(tid, {
            id: tid,
            title,
            status: 'ACTIVE',
            created: it.time || Date.now(),
          })
        }
      }
    }
    initialTasks = Array.from(foundTasks.values())
  }

  const persistTasks = (tasksList) => {
    if (!tasksPath) return
    try {
      fs.writeFileSync(tasksPath, JSON.stringify(tasksList, null, 2))
    } catch (_) {}
  }

  const taskManager = taskManagerFactory(Date.now)(initialTasks, persistTasks)
  const channelManager = ChannelManager()
  const registry = ToolRegistry(vfs, cmdRunner, trmManager, notManager, client, orchestrator)
  const parser = ToolParser(registry, client, store)

  controller.init(
    store,
    client,
    session,
    orchestrator,
    timers,
    systemPrompt,
    vfs,
    notManager,
    taskManager,
    channelManager,
  )
  controller.refs.isEngineReady = false

  // Automatic Snapshot Checkpointing State
  let snapshotDebounceTimer = null
  let lastSnapshotAnchor = null
  let lastSnapshotClock = null
  let lastSavedSlots = 0
  let currentServerSlots = 0

  const scheduleSnapshot = () => {
    if (!snapshotPath) return
    if (snapshotDebounceTimer) {
      clearTimeout(snapshotDebounceTimer)
    }
    snapshotDebounceTimer = setTimeout(() => {
      snapshotDebounceTimer = null
      const latestStreamId = store.state.stream?.[store.state.stream.length - 1]?.id || ''
      // Suppress redundant zero-delta snapshots if neither stream anchor nor slot count changed
      if (latestStreamId && latestStreamId === lastSnapshotAnchor) {
        return
      }
      if (currentServerSlots > 0 && currentServerSlots <= lastSavedSlots) {
        return
      }
      lastSavedSlots = currentServerSlots
      client.sendSnapshotSave(snapshotPath, latestStreamId)
    }, 5000)
  }

  // Handle Snapshot Status Response from Backend
  client.on('snapshotStatus', ({ status, activeSlots, clock, streamId }) => {
    if (status === 0) {
      // Snapshot saved successfully
      lastSnapshotAnchor = streamId
      lastSnapshotClock = clock
      lastSavedSlots = activeSlots
      store.addStreamEntry({
        type: 'checkpoint',
        title: '💾 SNAPSHOT',
        content: `Working state checkpointed (clock: ${clock}, active slots: ${activeSlots}, anchor: ${streamId}).`,
      })
      requestRedraw()
    } else if (status === 2) {
      // Snapshot already exists at this clock (clean deduplicated no-op)
      lastSnapshotAnchor = streamId
      lastSnapshotClock = clock
      lastSavedSlots = activeSlots
    } else if (status === 1) {
      // Snapshot restored successfully
      lastSnapshotAnchor = streamId
      lastSnapshotClock = clock
      lastSavedSlots = activeSlots
      controller.refs.hasSentFirstTurn = true
      controller.refs.systemPrecacheLogged = true
      controller.refs.isEngineReady = true
      store.addStreamEntry({
        type: 'checkpoint',
        title: '⚡ RESTORE',
        content: `Cognitive state restored from snapshot (clock: ${clock}, active slots: ${activeSlots}, anchor: ${streamId || 'initial'}). 0 ms warm boot.`,
      })

      // If user typed an input during boot/restore, dispatch it now
      if (controller.refs.pendingInputTurn) {
        const turn = controller.refs.pendingInputTurn
        controller.refs.pendingInputTurn = null
        for (const m of store.state.conversation) {
          if (m.waitingEngine) delete m.waitingEngine
        }
        client.sendInput(turn)
      }

      // Catch-up replay: check if the stream log contains turns beyond the checkpoint
      if (streamLog.enabled && streamId) {
        const allItems = streamLog.loadAll()
        const anchorIdx = allItems.findIndex(it => it.id === streamId)
        if (anchorIdx !== -1 && anchorIdx < allItems.length - 1) {
          const missingItems = allItems.slice(anchorIdx + 1)
          // Find un-checkpointed user turns or notification events that need replaying
          const replayTurns = []
          for (const it of missingItems) {
            if (it.type === 'user' && it.content) {
              replayTurns.push(it.content)
            }
          }
          if (replayTurns.length > 0) {
            store.addStreamEntry({
              type: 'system',
              title: '🔄 REPLAY',
              content: `Replaying ${replayTurns.length} un-checkpointed turn(s) through cognitive pipeline...`,
            })
            for (const turn of replayTurns) {
              client.sendInput(formatUserTurn(turn))
            }
          }
        }
      }

      requestRedraw()
    }
  })

  // Send early session system prompt pre-caching or restore from snapshot the moment backend starts
  let systemPrecached = false
  client.on('status', ({ status, activeSlots, currentTok, totalTok, tokSec }) => {
    if (status === 0 && !systemPrecached) {
      systemPrecached = true
      controller.refs.hasSentFirstTurn = true
      if (snapshotPath && fs.existsSync(snapshotPath)) {
        // Instant warm boot: restore GPU KV state and clock from snapshot
        client.sendSnapshotLoad(snapshotPath)
      } else {
        // Cold start: dispatch system instructions for KV pre-caching
        client.sendSystem(systemConfig)
        store.addStreamEntry({
          type: 'system',
          title: '⚙ SYSTEM',
          content: 'Pre-caching system instructions and tool definitions into GPU KV cache...',
        })
      }
      requestRedraw()
    } else if (systemPrecached && status === 1 && !controller.refs.systemPrecacheLogged) {
      const rate = tokSec > 0 ? ` (${tokSec.toFixed(1)} tok/s)` : ''
      store.setStatus(`[System Pre-cache] Encoding ${currentTok}/${totalTok} tokens${rate}...`)
      requestRedraw()
    } else if (systemPrecached && status === 0 && !controller.refs.systemPrecacheLogged && activeSlots > 0) {
      controller.refs.systemPrecacheLogged = true
      controller.refs.isEngineReady = true
      controller.refs.hasSentFirstTurn = true
      store.addStreamEntry({
        type: 'system',
        title: '⚙ SYSTEM',
        content: `System instructions and abstract tools pre-cached into Tier 1 anchors (${activeSlots} slots). Cognitive engine ready.`,
      })
      // Save initial system prompt snapshot for future instant boots
      scheduleSnapshot()

      // If user typed an input during boot/pre-cache, dispatch it now
      if (controller.refs.pendingInputTurn) {
        const turn = controller.refs.pendingInputTurn
        controller.refs.pendingInputTurn = null
        for (const m of store.state.conversation) {
          if (m.waitingEngine) delete m.waitingEngine
        }
        client.sendInput(turn)
      }
      requestRedraw()
    }
  })

  const app = Tui({
    view: path.resolve(__dirname, './view.yaml'),
    modules: { controller, markdown: { MarkdownControl } },
    autoFocus: false,
    truecolor: true,
  })

  let isDirty = false
  let redrawPending = false
  const requestRedraw = () => {
    isDirty = true
    if (redrawPending) return
    redrawPending = true
    setImmediate(() => {
      redrawPending = false
      if (isDirty) {
        isDirty = false
        app.redraw()
      }
    })
  }

  let pendingInterjection = null

  client.on('thought', ({ text }) => {
    isThinking = true
    const clean = text.replaceAll('\u2581', ' ')
    if (store.state.activeResponse) {
      store.flushActiveResponse()
    }
    store.appendActiveThought(clean)
    requestRedraw()
  })

  client.on('content', ({ text }) => {
    isThinking = false
    const clean = text.replaceAll('\u2581', ' ')
    if (store.state.activeThought) {
      store.flushActiveThought()
    }
    store.appendActiveResponse(clean)
    parser.ingestChunk(clean)
    requestRedraw()
  })

  client.on('status', ({ status, isGpu, activeSlots, archivedDiffs, isSaturated, tokSec, currentTok, totalTok }) => {
    currentServerSlots = activeSlots
    orchestrator.setSaturated(isSaturated)
    const sName = STATUS_NAMES[status] || 'Active'
    const isMixed = extraArgs.includes('--mixed')
    const devTag = isGpu ? (isMixed ? 'GPU Mixed' : 'GPU Q4_0') : 'CPU BF16'
    const prog = status === 1
      ? ` (${currentTok}/${totalTok} tok)`
      : (status === 2 ? ` (${currentTok} tok)` : '')
    const rate = tokSec > 0 ? ` | ${tokSec.toFixed(1)} tok/s` : ''
    const satTag = isSaturated ? ' [SATURATED]' : ''
    store.setStatus(
      `[${devTag}] ${sName}${prog}${rate} | Slots: ${activeSlots}${satTag} | Memory: ${archivedDiffs} diffs`,
    )
    requestRedraw()
  })

  const processNextBacklog = (allowDeferred = false) => {
    const suspended = notManager.getSuspended()
    if (suspended.length > 0) {
      const nextSuspended = suspended[0]
      const probePrompt = `Autonomic Backlog Triage\nEvaluate whether this pending background task has been completed, should be executed now, or should be deferred:\nTask: "${nextSuspended.preview}"\n0: Satisfied/Completed\n1: Execute now\n2: Defer\nDecision: `
      client.probe(probePrompt, ['0', '1', '2'], 1).then((res) => {
        if (res.winningIdx === 0) {
          notManager.ack(nextSuspended.id)
          store.addStreamEntry({
            type: 'system',
            title: '🎯 AUTO-ACK',
            content: `Autonomic probe resolved Task ${nextSuspended.id}: satisfied in prior response.`,
          })
          processNextBacklog()
        } else if (res.winningIdx === 1) {
          notManager.markServicing(nextSuspended.id)
          if (controller.refs)
            controller.refs.activeTurnNotificationId = nextSuspended.id
          const resumeNudge = formatBacklogResumeNudge(nextSuspended)
          store.setGenerating(true)
          client.sendInput(resumeNudge)
        } else {
          notManager.snooze(nextSuspended.id)
          store.addStreamEntry({
            type: 'system',
            title: '💤 DEFERRED',
            content: `Autonomic probe deferred Task ${nextSuspended.id} to queue tail.`,
          })
          processNextBacklog(false)
        }
        requestRedraw()
      })
      return true
    }

    if (allowDeferred) {
      const deferred = notManager.getPending().filter((a) => a.isDeferred)
      if (deferred.length > 0) {
        const nextDeferred = deferred[0]
        nextDeferred.isDeferred = false
        notManager.markServicing(nextDeferred.id)
        if (controller.refs)
          controller.refs.activeTurnNotificationId = nextDeferred.id
        const interruptNudge = formatNotificationInterrupt(nextDeferred)
        store.setGenerating(true)
        client.sendInput(interruptNudge)
        return true
      }
    }

    scheduleSnapshot()
    return false
  }

  client.on('turnComplete', ({ tokSec, elapsedMs, totalTok, reason }) => {
    // On elastic yield without external interrupt, retain active thought card open
    // so continued chunks smoothly append into the same card.
    const activeTurnId = controller.refs?.activeTurnNotificationId
    const unserviced = notManager.getUnserviced().filter(a => a.id !== activeTurnId)

    if (reason !== STOP_ELASTIC_YIELD || unserviced.length > 0) {
      store.flushActiveThought()
      store.flushActiveResponse()
      isThinking = false
    } else if (!isThinking && store.state.pendingInterjection) {
      // An interjection is waiting and we reached an elastic rest point: flush active response
      // so pendingInterjection immediately moves into conversation order.
      store.flushActiveResponse()
    }

    if (reason !== STOP_TOOL_CALL && reason !== STOP_ELASTIC_YIELD) {
      store.setGenerating(false)
      store.setStatus(`Idle | ${tokSec.toFixed(1)} tok/s | ${totalTok} tok in ${elapsedMs}ms`)
    }

    // 1. Elastic Yield Handling (Syntactic micro-burst boundary reached)
    if (reason === STOP_ELASTIC_YIELD) {
      if (unserviced.length > 0) {
        // High-priority external interrupt arrived mid-stream!
        const topInterrupt = unserviced[0]
        notManager.markServicing(topInterrupt.id)
        if (controller.refs) controller.refs.activeTurnNotificationId = topInterrupt.id
        const interruptNudge = formatNotificationInterrupt(topInterrupt)
        store.setGenerating(true)
        client.sendInput(interruptNudge)
      } else if (isThinking && typeof client.probe === 'function') {
        const prompt = [
          'Autonomic Reasoning Assessment',
          'Evaluate whether the thoughts generated so far are sufficient to deliver the response to the user, or if further reasoning is needed:',
          '0: Reasoning sufficient, deliver response now',
          '1: Further reasoning needed',
          'Decision:',
        ].join('\n')
        client.probe(prompt, ['0', '1'], 1)
          .then((res) => {
            const isSufficient = res.winningIdx === 0 && res.confidence >= 0.80
            const confPct = res.confidence != null ? (res.confidence * 100).toFixed(1) : '?'
            if (isSufficient) {
              store.flushActiveThought()
              store.addStreamEntry({
                type: 'notice',
                title: '⚡ THINKING CAPPED',
                content: `Sufficient reasoning reached (${confPct}% confidence >= 80%). Transitioning to response.`,
              })
              isThinking = false
              store.setGenerating(true)
              client.sendResume({ closeThought: true })
            } else {
              store.setGenerating(true)
              client.sendResume()
            }
            requestRedraw()
          })
          .catch((_) => {
            store.setGenerating(true)
            client.sendResume()
            requestRedraw()
          })
      } else {
        // No new unserviced interruption: seamless autonomous continuation via binary protocol
        store.setGenerating(true)
        client.sendResume()
      }
      requestRedraw()
      return
    }

    // 2. Explicit Turn Completion (<turn|>)
    if (reason === STOP_END_OF_TURN) {
      // Auto-ACK the active turn notification if it completed cleanly
      if (activeTurnId) {
        notManager.ack(activeTurnId)
        if (controller.refs) controller.refs.activeTurnNotificationId = null
      }
      for (const item of notManager.getSuspended()) {
        if (item.extra?.isTurnContext) notManager.ack(item.id)
      }

      // Check for remaining unserviced interrupts in LIFO order
      const remainingUnserviced = notManager.getUnserviced().filter(a => a.id !== activeTurnId)
      if (remainingUnserviced.length > 0) {
        const nextAlert = remainingUnserviced[0]
        notManager.markServicing(nextAlert.id)
        if (controller.refs) controller.refs.activeTurnNotificationId = nextAlert.id
        const interruptNudge = formatNotificationInterrupt(nextAlert)
        store.setGenerating(true)
        client.sendInput(interruptNudge)
        requestRedraw()
        return
      }

      // Autonomically triage remaining suspended backlog via 1-token probe
      if (processNextBacklog()) {
        requestRedraw()
        return
      }

      // Arm 5-second quiescence snapshot checkpoint
      scheduleSnapshot()
    }

    // 3. Autonomous tick loop for active plan steps
    const activeStep = orchestrator.getActiveStep()
    if (activeStep && activeStep.status === 'IN_PROGRESS') {
      const nextThought = orchestrator.buildThoughtPrefix('STEP_TICK', { step: activeStep })
      store.setGenerating(true)
      client.sendInput(nextThought)
    }

    requestRedraw()
  })

  client.on('memResponse', ({ count, status }) => {
    store.setStatus(`Memory retrieved: ${count} episodes (status: ${status})`)
    requestRedraw()
  })

  client.on('drain', requestRedraw)

  logDebug('TUI main() started')

  client.on('exit', ({ code }) => {
    logDebug(`client on exit: code=${code}`)
    store.setGenerating(false)
    store.setStatus(`[Error] Inference backend stopped (exit code: ${code})`)
    requestRedraw()
  })

  client.on('error', ({ error }) => {
    logDebug(`client on error: ${error}`)
    store.setGenerating(false)
    store.setStatus(`[Error] Backend process error: ${error}`)
    requestRedraw()
  })

  const cleanup = (trigger) => {
    logDebug(`cleanup() invoked from trigger: ${trigger}`)
    if (snapshotPath && store.state.stream?.length > 0) {
      try {
        const latestStreamId = store.state.stream[store.state.stream.length - 1]?.id || ''
        client.sendSnapshotSave(snapshotPath, latestStreamId)
      } catch (_) {}
    }
    try { cmdRunner.killAll() } catch (_) {}
    try { trmManager.closeAll() } catch (_) {}
    try { streamLog.close() } catch (_) {}
    try { session.kill() } catch (_) {}
    try { client.shutdown() } catch (_) {}
    try { client.kill() } catch (_) {}
  }

  app.onExit(() => cleanup('app.onExit'))
  process.on('exit', (code) => {
    logDebug(`process exit event: code=${code}`)
    cleanup(`process.exit(${code})`)
  })
  process.on('SIGINT', () => { logDebug('SIGINT received'); cleanup('SIGINT'); process.exit(0) })
  process.on('SIGTERM', () => { logDebug('SIGTERM received'); cleanup('SIGTERM'); process.exit(0) })
  process.on('SIGHUP', () => { logDebug('SIGHUP received'); cleanup('SIGHUP'); process.exit(0) })
  process.on('uncaughtException', (err) => {
    logDebug(`uncaughtException: ${err?.stack || err}`)
    try {
      fs.appendFileSync(
        path.resolve(process.cwd(), 'crash.log'),
        `[${new Date().toISOString()}] Uncaught exception:\n${err.stack || err}\n\n`,
      )
    } catch (_) {}
    cleanup('uncaughtException')
    console.error(err)
    process.exit(1)
  })
  process.on('unhandledRejection', (err) => {
    logDebug(`unhandledRejection: ${err?.stack || err}`)
    try {
      fs.appendFileSync(
        path.resolve(process.cwd(), 'crash.log'),
        `[${new Date().toISOString()}] Unhandled rejection:\n${err?.stack || err}\n\n`,
      )
    } catch (_) {}
    cleanup('unhandledRejection')
    console.error(err)
    process.exit(1)
  })

  client.start()
  if (config.runtime) client.setConfig(config.runtime)
  app.start()

  setInterval(() => {
    app.redraw()
  }, 250)
}

if (require.main === module) {
  main()
}

module.exports = {
  loadConfig,
  loadSystemConfig,
  loadSystemPrompt: (promptPath) => loadSystemConfig(promptPath).instructions,
  parseCliArgs,
  main,
}
