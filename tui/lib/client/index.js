const { spawn } = require('child_process')
const { EventEmitter } = require('events')
const {
  OP_STREAM_CONTENT,
  OP_STREAM_THOUGHT,
  OP_TURN_COMPLETE,
  OP_TOOL_CALL,
  OP_MEM_RESPONSE,
  OP_STATUS,
  OP_SNAPSHOT_STATUS,
  OP_AUTONOMIC_RESULT,
  OP_PONG,
  OP_ERROR,
  STATUS_FLAG_SATURATED,
  INPUT_FLAG_NONE,
  INPUT_FLAG_DIRECT,
  INPUT_FLAG_REASON,
  RESUME_ACTION_CONTINUE,
  RESUME_ACTION_CLOSE_THOUGHT,
} = require('../protocol/constants')
const {
  streamInputFrame,
  resumeFrame,
  abortFrame,
  memQueryFrame,
  memCommitFrame,
  configFrame,
  shutdownFrame,
  setSystemFrame,
  snapshotSaveFrame,
  snapshotLoadFrame,
  toolReturnFrame,
  probeAutonomicFrame,
  parseAutonomicResult,
  parsedFrame,
} = require('../protocol/framing')

const clientFactory = (spawnProc, EmitterClass) => (opts) => {
  const { binaryPath, modelPath, extraArgs = [] } = opts
  const emitter = new EmitterClass()
  let proc = null
  let rxBuf = Buffer.alloc(0)
  let nextMsgId = 1
  const pendingProbes = new Map()

  const handleFrame = (h, p) => {
    const text = () => p.subarray(24).toString('utf-8').replace(/\u2581/g, ' ')
    if (h.opcode === OP_STREAM_CONTENT) emitter.emit('content', { text: text(), msgId: h.msgId })
    else if (h.opcode === OP_STREAM_THOUGHT) emitter.emit('thought', { text: text(), msgId: h.msgId })
    else if (h.opcode === OP_TOOL_CALL) {
      const callId = p.readUInt16LE(0)
      const nameLen = p.readUInt16LE(2)
      const toolName = p.subarray(4, 4 + nameLen).toString('utf-8')
      const argsJson = p.subarray(4 + nameLen).toString('utf-8')
      emitter.emit('toolCall', { callId, toolName, argsJson, msgId: h.msgId })
    }
    else if (h.opcode === OP_TURN_COMPLETE) emitter.emit('turnComplete', {
      totalTok: p.readUInt32LE(0), elapsedMs: p.readUInt32LE(4), tokSec: p.readFloatLE(8), reason: p.readUInt8(12), msgId: h.msgId,
    })
    else if (h.opcode === OP_STATUS) emitter.emit('status', {
      status: p.readUInt8(0), isGpu: p.readUInt8(1), activeSlots: p.readUInt16LE(2), archivedDiffs: p.readUInt16LE(4),
      flags: p.readUInt16LE(6), isSaturated: Boolean(p.readUInt16LE(6) & STATUS_FLAG_SATURATED),
      tokSec: p.readFloatLE(8), currentTok: p.readUInt32LE(12), totalTok: p.readUInt32LE(16), msgId: h.msgId,
    })
    else if (h.opcode === OP_SNAPSHOT_STATUS) {
      const status = p.readUInt8(0)
      const activeSlots = p.readUInt16LE(2)
      const clock = Number(p.readBigUInt64LE(4))
      const idLen = p.readUInt16LE(12)
      const streamId = p.subarray(14, 14 + idLen).toString('utf-8')
      emitter.emit('snapshotStatus', { status, activeSlots, clock, streamId, msgId: h.msgId })
    }
    else if (h.opcode === OP_AUTONOMIC_RESULT) {
      const autonomicResult = parseAutonomicResult(p)
      if (autonomicResult) {
        emitter.emit('autonomicResult', { ...autonomicResult, msgId: h.msgId })
        if (pendingProbes.has(h.msgId)) {
          const resolver = pendingProbes.get(h.msgId)
          pendingProbes.delete(h.msgId)
          resolver(autonomicResult)
        }
      }
    }
    else if (h.opcode === OP_MEM_RESPONSE) emitter.emit('memResponse', { count: p.readUInt16LE(0), status: p.readUInt8(2), msgId: h.msgId })
    else if (h.opcode === OP_PONG) emitter.emit('pong', { msgId: h.msgId })
    else if (h.opcode === OP_ERROR) emitter.emit('error', { error: p.toString('utf-8'), msgId: h.msgId })
  }

  const start = () => {
    proc = spawnProc(binaryPath, ['--model', modelPath, '--serve', ...extraArgs], { stdio: ['pipe', 'pipe', 'inherit'] })
    proc.stdout.on('data', (chunk) => {
      rxBuf = Buffer.concat([rxBuf, chunk])
      let parsed = parsedFrame(rxBuf)
      let count = 0
      while (parsed) {
        rxBuf = parsed.rest
        handleFrame(parsed.header, parsed.payload)
        count++
        parsed = parsedFrame(rxBuf)
      }
      if (count > 0) emitter.emit('drain')
    })
    proc.on('close', (code) => emitter.emit('exit', { code }))
    proc.on('error', (err) => emitter.emit('error', { error: err.message }))
  }

  const sendInput = (text, opts = {}) => {
    const id = nextMsgId++
    const flags = typeof opts === 'number'
      ? opts
      : opts.flags ?? (opts.direct ? INPUT_FLAG_DIRECT : (opts.reason ? INPUT_FLAG_REASON : INPUT_FLAG_NONE))
    if (proc?.stdin?.writable) proc.stdin.write(streamInputFrame(text, flags, id))
    return id
  }

  const sendSystem = (systemJson) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(setSystemFrame(systemJson, id))
    return id
  }

  const sendToolReturn = (toolName, result, callId = 1, status = 0) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(toolReturnFrame(toolName, result, callId, status, id))
    return id
  }

  const sendResume = (opts = {}) => {
    const id = nextMsgId++
    const action = typeof opts === 'number'
      ? opts
      : (opts?.closeThought ? RESUME_ACTION_CLOSE_THOUGHT : RESUME_ACTION_CONTINUE)
    if (proc?.stdin?.writable) proc.stdin.write(resumeFrame(id, action))
    return id
  }

  const sendProbeAutonomic = (prompt, candidates = [], maxDecodeTokens = 1, minCertainty = 0.6) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(probeAutonomicFrame(prompt, candidates, maxDecodeTokens, minCertainty, id))
    return id
  }

  const probe = (prompt, candidates = [], maxDecodeTokens = 1, minCertainty = 0.6) => {
    return new Promise((resolve) => {
      const id = sendProbeAutonomic(prompt, candidates, maxDecodeTokens, minCertainty)
      pendingProbes.set(id, resolve)
    })
  }

  const sendAbort = () => {
    if (proc?.stdin?.writable) proc.stdin.write(abortFrame(nextMsgId++))
  }

  const sendMemQuery = (query, topK = 5) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(memQueryFrame(query, id, topK))
    return id
  }

  const sendMemCommit = () => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(memCommitFrame(id))
    return id
  }

  const sendSnapshotSave = (snapPath, streamId = '') => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(snapshotSaveFrame(snapPath, streamId, id))
    return id
  }

  const sendSnapshotLoad = (snapPath) => {
    const id = nextMsgId++
    if (proc?.stdin?.writable) proc.stdin.write(snapshotLoadFrame(snapPath, id))
    return id
  }

  const setConfig = (o) => {
    if (proc?.stdin?.writable) {
      const budget = o.thinkingBudget ?? o.budget ?? 512
      proc.stdin.write(
        configFrame(
          budget,
          o.temp,
          o.topP,
          o.qThresh,
          o.maxTokens,
          o.minP,
          o.repeatPenalty,
          o.repeatLastN ?? 64,
          o.frequencyPenalty ?? 0.1,
          o.presencePenalty ?? 0.1,
          Boolean(o.thinkingGate),
          nextMsgId++,
        ),
      )
    }
  }

  const kill = () => {
    if (proc && !proc.killed) {
      try {
        proc.kill('SIGKILL')
      } catch (_) {}
    }
  }

  const shutdown = () => {
    if (proc?.stdin?.writable) {
      try {
        proc.stdin.write(shutdownFrame())
        proc.stdin.end()
      } catch (_) {}
    }
    setTimeout(kill, 500).unref?.()
  }

  return {
    start,
    sendInput,
    sendResume,
    sendProbeAutonomic,
    probe,
    sendSystem,
    sendToolReturn,
    sendSnapshotSave,
    sendSnapshotLoad,
    sendAbort,
    sendMemQuery,
    sendMemCommit,
    setConfig,
    shutdown,
    kill,
    on: emitter.on.bind(emitter),
  }
}

module.exports = {
  clientFactory,
  Client: clientFactory(spawn, EventEmitter),
}
