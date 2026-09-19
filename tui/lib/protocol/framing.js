const {
  MAGIC,
  PROTOCOL_VERSION,
  OP_STREAM_INPUT,
  OP_ABORT,
  OP_MEM_QUERY,
  OP_SET_CONFIG,
  OP_TOOL_RETURN,
  OP_MEM_COMMIT,
  OP_SET_SYSTEM,
  OP_SNAPSHOT_SAVE,
  OP_SNAPSHOT_LOAD,
  OP_RESUME,
  OP_PROBE_AUTONOMIC,
  OP_PING,
  OP_SHUTDOWN,
  MODE_TEXT,
} = require('./constants')

const headerBuffer = (opcode, msgId, payloadLen) => {
  const buf = Buffer.alloc(16)
  buf.writeUInt32LE(MAGIC, 0)
  buf.writeUInt16LE(PROTOCOL_VERSION, 4)
  buf.writeUInt16LE(msgId, 6)
  buf.writeUInt16LE(opcode, 8)
  buf.writeUInt16LE(0, 10)
  buf.writeUInt32LE(payloadLen, 12)
  return buf
}

const streamInputFrame = (text, flagsOrMsgId = 0, maybeMsgId = null) => {
  const textBytes = Buffer.from(text, 'utf-8')
  const payload = Buffer.alloc(8 + textBytes.length)
  payload.writeUInt8(MODE_TEXT, 0)

  let flags = 0
  let msgId = 1
  if (maybeMsgId !== null) {
    flags = flagsOrMsgId || 0
    msgId = maybeMsgId
  } else if (flagsOrMsgId > 0 && (flagsOrMsgId === 1 || flagsOrMsgId === 2 || flagsOrMsgId === 0x80)) {
    flags = flagsOrMsgId
    msgId = 1
  } else {
    msgId = flagsOrMsgId || 1
  }

  payload.writeUInt8(flags, 1)
  textBytes.copy(payload, 8)
  const hdr = headerBuffer(OP_STREAM_INPUT, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const abortFrame = (msgId = 0) => headerBuffer(OP_ABORT, msgId, 0)

const memCommitFrame = (msgId = 1) => headerBuffer(OP_MEM_COMMIT, msgId, 0)

const pingFrame = (msgId = 1) => headerBuffer(OP_PING, msgId, 0)

const shutdownFrame = () => headerBuffer(OP_SHUTDOWN, 0, 0)

const setSystemFrame = (systemJson, msgId = 1) => {
  const jsonBytes = Buffer.from(typeof systemJson === 'string' ? systemJson : JSON.stringify(systemJson), 'utf-8')
  const hdr = headerBuffer(OP_SET_SYSTEM, msgId, jsonBytes.length)
  return Buffer.concat([hdr, jsonBytes])
}

const memQueryFrame = (query, msgId = 1, topK = 5) => {
  const queryBytes = Buffer.from(query, 'utf-8')
  const payload = Buffer.alloc(24 + queryBytes.length)
  payload.writeUInt8(0x00, 0) // keywords
  payload.writeUInt8(topK, 1)
  payload.writeBigUInt64LE(0n, 8)
  payload.writeBigUInt64LE(0n, 16)
  queryBytes.copy(payload, 24)
  const hdr = headerBuffer(OP_MEM_QUERY, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const configFrame = (
  thinkingBudget = 512,
  temp = 1.0,
  topP = 0.95,
  qThresh = 0.001,
  maxTok = 64,
  minP = 0.05,
  repeatPenalty = 1.1,
  repeatLastN = 64,
  frequencyPenalty = 0.1,
  presencePenalty = 0.1,
  thinkingGate = false,
  msgId = 1,
) => {
  const payload = Buffer.alloc(41)
  payload.writeUInt32LE(thinkingBudget, 0)
  payload.writeFloatLE(temp, 4)
  payload.writeFloatLE(topP, 8)
  payload.writeFloatLE(qThresh, 12)
  payload.writeUInt32LE(maxTok, 16)
  payload.writeFloatLE(minP, 20)
  payload.writeFloatLE(repeatPenalty, 24)
  payload.writeUInt32LE(repeatLastN, 28)
  payload.writeFloatLE(frequencyPenalty, 32)
  payload.writeFloatLE(presencePenalty, 36)
  payload.writeUInt8(thinkingGate ? 1 : 0, 40)
  const hdr = headerBuffer(OP_SET_CONFIG, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const snapshotSaveFrame = (snapPath, streamId = '', msgId = 1) => {
  const pathBytes = Buffer.from(snapPath, 'utf-8')
  const idBytes = Buffer.from(streamId, 'utf-8')
  const payload = Buffer.alloc(2 + pathBytes.length + 2 + idBytes.length)
  payload.writeUInt16LE(pathBytes.length, 0)
  pathBytes.copy(payload, 2)
  payload.writeUInt16LE(idBytes.length, 2 + pathBytes.length)
  idBytes.copy(payload, 2 + pathBytes.length + 2)
  const hdr = headerBuffer(OP_SNAPSHOT_SAVE, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const snapshotLoadFrame = (snapPath, msgId = 1) => {
  const pathBytes = Buffer.from(snapPath, 'utf-8')
  const hdr = headerBuffer(OP_SNAPSHOT_LOAD, msgId, pathBytes.length)
  return Buffer.concat([hdr, pathBytes])
}

const resumeFrame = (msgId = 1, action = 0) => {
  if (!action) return headerBuffer(OP_RESUME, msgId, 0)
  const buf = Buffer.alloc(17)
  headerBuffer(OP_RESUME, msgId, 1).copy(buf, 0)
  buf[16] = action
  return buf
}

const probeAutonomicFrame = (
  prompt,
  candidates = [],
  maxDecodeTokens = 1,
  minCertainty = 0.6,
  msgId = 1,
) => {
  const promptBytes = Buffer.from(prompt || '', 'utf-8')
  const candByteArrays = candidates.map((c) => Buffer.from(c || '', 'utf-8'))
  let totalCandBytes = 0
  candByteArrays.forEach((b) => {
    totalCandBytes += 2 + b.length
  })

  const payload = Buffer.alloc(
    2 + 2 + 4 + 2 + promptBytes.length + totalCandBytes,
  )
  payload.writeUInt16LE(maxDecodeTokens, 0)
  payload.writeUInt16LE(candidates.length, 2)
  payload.writeFloatLE(minCertainty, 4)
  payload.writeUInt16LE(promptBytes.length, 8)
  promptBytes.copy(payload, 10)

  let off = 10 + promptBytes.length
  candByteArrays.forEach((b) => {
    payload.writeUInt16LE(b.length, off)
    off += 2
    b.copy(payload, off)
    off += b.length
  })

  const hdr = headerBuffer(OP_PROBE_AUTONOMIC, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const parseAutonomicResult = (payload) => {
  if (payload.length < 16) return null
  const winningIdx = payload.readUInt16LE(0)
  const confidence = payload.readFloatLE(2)
  const entropy = payload.readFloatLE(6)
  const costMs = payload.readFloatLE(10)
  const textLen = payload.readUInt16LE(14)
  const text =
    payload.length >= 16 + textLen
      ? payload
          .slice(16, 16 + textLen)
          .toString('utf-8')
          .replaceAll('\u2581', ' ')
      : ''
  return { winningIdx, confidence, entropy, costMs, text }
}

const toolReturnFrame = (toolName, result, callId = 1, status = 0, msgId = 1) => {
  const nameBytes = Buffer.from(toolName, 'utf-8')
  const jsonStr = typeof result === 'string' ? result : JSON.stringify(result)
  const jsonBytes = Buffer.from(jsonStr, 'utf-8')
  const payload = Buffer.alloc(6 + nameBytes.length + jsonBytes.length)
  payload.writeUInt16LE(callId, 0)
  payload.writeUInt16LE(status, 2)
  payload.writeUInt16LE(nameBytes.length, 4)
  nameBytes.copy(payload, 6)
  jsonBytes.copy(payload, 6 + nameBytes.length)
  const hdr = headerBuffer(OP_TOOL_RETURN, msgId, payload.length)
  return Buffer.concat([hdr, payload])
}

const parsedFrame = (buf) => {
  if (buf.length < 16) return null
  const magic = buf.readUInt32LE(0)
  if (magic !== MAGIC) return null
  const version = buf.readUInt16LE(4)
  const msgId = buf.readUInt16LE(6)
  const opcode = buf.readUInt16LE(8)
  const payloadLen = buf.readUInt32LE(12)
  if (buf.length < 16 + payloadLen) return null

  const payload = buf.subarray(16, 16 + payloadLen)
  const rest = buf.subarray(16 + payloadLen)
  return { header: { version, msgId, opcode, payloadLen }, payload, rest }
}

module.exports = {
  headerBuffer,
  streamInputFrame,
  abortFrame,
  memCommitFrame,
  setSystemFrame,
  pingFrame,
  shutdownFrame,
  resumeFrame,
  memQueryFrame,
  toolReturnFrame,
  snapshotSaveFrame,
  snapshotLoadFrame,
  probeAutonomicFrame,
  parseAutonomicResult,
  configFrame,
  parsedFrame,
}
