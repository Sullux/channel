const test = require('node:test')
const assert = require('node:assert')
const {
  headerBuffer,
  streamInputFrame,
  abortFrame,
  memQueryFrame,
  memCommitFrame,
  configFrame,
  snapshotSaveFrame,
  snapshotLoadFrame,
  resumeFrame,
  toolReturnFrame,
  probeAutonomicFrame,
  parseAutonomicResult,
  pingFrame,
  parsedFrame,
} = require('../lib/protocol/framing')
const {
  MAGIC,
  OP_STREAM_INPUT,
  OP_ABORT,
  OP_MEM_QUERY,
  OP_MEM_COMMIT,
  OP_SET_CONFIG,
  OP_TOOL_RETURN,
  OP_SNAPSHOT_SAVE,
  OP_SNAPSHOT_LOAD,
  OP_RESUME,
  RESUME_ACTION_CLOSE_THOUGHT,
  OP_TASK_TRIAGE,
  OP_READ_STREAM_OPEN,
  OP_READ_STREAM_CLOSE,
  OP_BACKLOG_TRIAGE,
  OP_TASK_TITLE,
  OP_PROBE_AUTONOMIC,
  OP_BACKLOG_ROUTED,
  OP_TASK_TITLE_RESULT,
  OP_AUTONOMIC_RESULT,
  BACKLOG_ACTION_ACK,
  READ_STATUS_CHUNK,
  OP_PING,
} = require('../lib/protocol/constants')

test('headerBuffer creates 16-byte valid binary header', () => {
  const hdr = headerBuffer(OP_STREAM_INPUT, 42, 100)
  assert.strictEqual(hdr.length, 16)
  assert.strictEqual(hdr.readUInt32LE(0), MAGIC)
  assert.strictEqual(hdr.readUInt16LE(4), 1) // version
  assert.strictEqual(hdr.readUInt16LE(6), 42) // msgId
  assert.strictEqual(hdr.readUInt16LE(8), OP_STREAM_INPUT) // opcode
  assert.strictEqual(hdr.readUInt32LE(12), 100) // payloadLen
})

test('streamInputFrame serializes text payload properly', () => {
  const frame = streamInputFrame('hello world', 7)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 7)
  assert.strictEqual(parsed.header.opcode, OP_STREAM_INPUT)
  assert.strictEqual(parsed.payload.readUInt8(0), 0) // MODE_TEXT
  assert.strictEqual(parsed.payload.subarray(8).toString('utf-8'), 'hello world')
})

test('memQueryFrame serializes query and topK correctly', () => {
  const frame = memQueryFrame('matrix multiplication', 9, 3)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 9)
  assert.strictEqual(parsed.header.opcode, OP_MEM_QUERY)
  assert.strictEqual(parsed.payload.readUInt8(1), 3) // topK
  assert.strictEqual(parsed.payload.subarray(24).toString('utf-8'), 'matrix multiplication')
})

test('memCommitFrame serializes commit opcode properly', () => {
  const frame = memCommitFrame(12)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 12)
  assert.strictEqual(parsed.header.opcode, OP_MEM_COMMIT)
  assert.strictEqual(parsed.payload.length, 0)
})

test('configFrame serializes penalty and runtime options properly', () => {
  const frame = configFrame(512, 0.7, 0.95, 0.001, 256, 0.05, 1.15, 64, 0.2, 0.3, true, 11)
  const parsed = parsedFrame(frame)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.header.msgId, 11)
  assert.strictEqual(parsed.header.opcode, OP_SET_CONFIG)
  assert.strictEqual(parsed.payload.length, 41)
  assert.strictEqual(parsed.payload.readUInt32LE(0), 512)
  assert.strictEqual(parsed.payload.readUInt32LE(16), 256)
  assert.strictEqual(parsed.payload.readUInt32LE(28), 64)
  assert.strictEqual(parsed.payload.readUInt8(40), 1)
})

test('snapshotSaveFrame and snapshotLoadFrame serialize properly', () => {
  const saveBuf = snapshotSaveFrame('/tmp/snap.bin', 'str-123', 5)
  const parsedSave = parsedFrame(saveBuf)
  assert.strictEqual(parsedSave.header.opcode, OP_SNAPSHOT_SAVE)
  assert.strictEqual(parsedSave.header.msgId, 5)

  const loadBuf = snapshotLoadFrame('/tmp/snap.bin', 6)
  const parsedLoad = parsedFrame(loadBuf)
  assert.strictEqual(parsedLoad.header.opcode, OP_SNAPSHOT_LOAD)
  assert.strictEqual(parsedLoad.header.msgId, 6)
  assert.strictEqual(parsedLoad.payload.toString('utf-8'), '/tmp/snap.bin')
})

test('resumeFrame serializes properly with zero-length payload', () => {
  const frame = resumeFrame(8)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_RESUME)
  assert.strictEqual(parsed.header.msgId, 8)
  assert.strictEqual(parsed.header.payloadLen, 0)
  assert.strictEqual(parsed.payload.length, 0)
})

test('resumeFrame serializes properly with action payload', () => {
  const frame = resumeFrame(9, RESUME_ACTION_CLOSE_THOUGHT)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_RESUME)
  assert.strictEqual(parsed.header.msgId, 9)
  assert.strictEqual(parsed.header.payloadLen, 1)
  assert.strictEqual(parsed.payload[0], RESUME_ACTION_CLOSE_THOUGHT)
})

test('toolReturnFrame serializes properly', () => {
  const frame = toolReturnFrame('ack', { status: 'ok' }, 7, 0, 12)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_TOOL_RETURN)
  assert.strictEqual(parsed.header.msgId, 12)
  assert.strictEqual(parsed.payload.readUInt16LE(0), 7) // callId
  assert.strictEqual(parsed.payload.readUInt16LE(2), 0) // status
  assert.strictEqual(parsed.payload.readUInt16LE(4), 3) // name_len ('ack')
  assert.strictEqual(parsed.payload.subarray(6, 9).toString('utf-8'), 'ack')
  assert.strictEqual(parsed.payload.subarray(9).toString('utf-8'), JSON.stringify({ status: 'ok' }))
})

test('parsedFrame handles incomplete buffers gracefully', () => {
  const frame = pingFrame(1)
  const partial = frame.subarray(0, 10)
  assert.strictEqual(parsedFrame(partial), null)
})

test('probeAutonomicFrame and parseAutonomicResult serialize and unpack properly', () => {
  const frame = probeAutonomicFrame('I should', ['wait', 'read'], 1, 0.75, 22)
  const parsed = parsedFrame(frame)
  assert.strictEqual(parsed.header.opcode, OP_PROBE_AUTONOMIC)
  assert.strictEqual(parsed.header.msgId, 22)
  assert.strictEqual(parsed.payload.readUInt16LE(0), 1) // maxDecodeTokens
  assert.strictEqual(parsed.payload.readUInt16LE(2), 2) // candidateCount
  assert.ok(Math.abs(parsed.payload.readFloatLE(4) - 0.75) < 0.001) // minCertainty

  const resultBuf = Buffer.alloc(16 + 8)
  resultBuf.writeUInt16LE(1, 0) // winningIdx
  resultBuf.writeFloatLE(0.88, 2) // confidence
  resultBuf.writeFloatLE(0.25, 6) // entropy
  resultBuf.writeFloatLE(15.4, 10) // costMs
  resultBuf.writeUInt16LE(8, 14) // textLen
  Buffer.from('finished', 'utf-8').copy(resultBuf, 16)

  const res = parseAutonomicResult(resultBuf)
  assert.strictEqual(res.winningIdx, 1)
  assert.ok(Math.abs(res.confidence - 0.88) < 0.001)
  assert.ok(Math.abs(res.entropy - 0.25) < 0.001)
  assert.ok(Math.abs(res.costMs - 15.4) < 0.001)
  assert.strictEqual(res.text, 'finished')
})
