const { test, describe, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const path = require('path')
const os = require('os')
const { streamLogFactory } = require('../lib/storage')

describe('StreamLog Storage Subsystem', () => {
  let tmpDir

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'stream-log-test-'))
  })

  afterEach(() => {
    if (tmpDir && fs.existsSync(tmpDir)) {
      fs.rmSync(tmpDir, { recursive: true, force: true })
    }
  })

  test('returns no-op logger when memoryDir is not provided', () => {
    const StreamLog = streamLogFactory(fs, path)
    const log = StreamLog(null)

    assert.equal(log.enabled, false)
    assert.doesNotThrow(() => log.append({ text: 'hello' }))
    const tail = log.loadTail()
    assert.deepEqual(tail.items, [])
    assert.equal(tail.startOffset, 0)
    assert.equal(tail.endOffset, 0)
  })

  test('appends and reads back stream items in small file mode', async () => {
    const StreamLog = streamLogFactory(fs, path)
    const log = StreamLog(tmpDir)

    assert.equal(log.enabled, true)
    log.append({ type: 'user', content: 'Hello 1', time: 1000 })
    log.append({ type: 'thought', content: 'Thinking 1', time: 1001 })
    log.append({ type: 'response', content: 'Response 1', time: 1002 })

    // Give writeStream a tick to flush
    await new Promise((r) => setTimeout(r, 50))

    const tail = log.loadTail(1024 * 1024)
    assert.equal(tail.items.length, 3)
    assert.equal(tail.items[0].content, 'Hello 1')
    assert.equal(tail.items[1].content, 'Thinking 1')
    assert.equal(tail.items[2].content, 'Response 1')
    assert.equal(tail.startOffset, 0)
    assert.ok(tail.endOffset > 0)

    log.close()
  })

  test('loads tail accurately from chunked file with partial line discard', async () => {
    const StreamLog = streamLogFactory(fs, path)
    const log = StreamLog(tmpDir)

    // Write 50 items
    for (let i = 0; i < 50; i++) {
      log.append({ id: i, type: 'thought', content: `Item number ${i} with extra padding data`, time: i * 100 })
    }
    await new Promise((r) => setTimeout(r, 50))

    // Read with small chunk size (e.g. 500 bytes)
    const tail = log.loadTail(500)
    assert.ok(tail.items.length > 0)
    assert.ok(tail.items.length < 50)
    assert.equal(tail.items[tail.items.length - 1].id, 49)
    assert.ok(tail.startOffset > 0)

    // Load previous page
    const prev = log.loadPrevious(tail.startOffset, 500)
    assert.ok(prev.items.length > 0)
    assert.ok(prev.startOffset < tail.startOffset)
    assert.equal(prev.items[prev.items.length - 1].id, tail.items[0].id - 1)

    log.close()
  })
})
