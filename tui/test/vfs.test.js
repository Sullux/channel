const { describe, it, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert')
const fs = require('fs')
const path = require('path')
const { vfsFactory, MAX_READ_LIMIT } = require('../lib/vfs')

const TEST_ROOT = path.resolve(__dirname, './.test_agent')

describe('Vfs (Virtual File Subsystem)', () => {
  let vfs

  beforeEach(() => {
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
    vfs = vfsFactory()(TEST_ROOT)
    vfs.init()
  })

  afterEach(() => {
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
  })

  it('initializes standard directory structure', () => {
    assert.strictEqual(fs.existsSync(path.join(TEST_ROOT, 'msg', 'user')), true)
    assert.strictEqual(fs.existsSync(path.join(TEST_ROOT, 'trm')), true)
    assert.strictEqual(fs.existsSync(path.join(TEST_ROOT, 'tmp')), true)
    assert.strictEqual(fs.existsSync(path.join(TEST_ROOT, 'notify', 'pending')), true)
  })

  it('saves user messages and enforces read-only (chmod 0444)', () => {
    const res = vfs.saveUserMessage('Hello world, please check status.')
    assert.strictEqual(res.id, '1001')
    assert.strictEqual(res.isTruncated, false)
    assert.strictEqual(res.preview, 'Hello world, please check status.')
    assert.strictEqual(fs.existsSync(res.path), true)

    const content = fs.readFileSync(res.path, 'utf-8')
    assert.strictEqual(content, 'Hello world, please check status.')

    // Check read-only bit
    const stat = fs.statSync(res.path)
    // 0o444 in mode & 0o777
    assert.strictEqual(stat.mode & 0o222, 0)
  })

  it('truncates preview for large user messages (> 128 chars)', () => {
    const longText = 'A'.repeat(200)
    const res = vfs.saveUserMessage(longText)
    assert.strictEqual(res.isTruncated, true)
    assert.strictEqual(res.preview.length, 128)
    assert.strictEqual(res.payload.length, 200)
  })

  it('reads bounded slices up to MAX_READ_LIMIT (512 chars)', () => {
    const longContent = '0123456789'.repeat(100) // 1000 chars
    const filePath = path.join(TEST_ROOT, 'tmp', 'test.log')
    fs.writeFileSync(filePath, longContent, 'utf-8')

    // Read first chunk
    const slice1 = vfs.read('tmp/test.log', 0, 1000)
    assert.strictEqual(slice1.bytesRead, MAX_READ_LIMIT)
    assert.strictEqual(slice1.content.length, 512)
    assert.strictEqual(slice1.eof, false)

    // Read second chunk
    const slice2 = vfs.read('tmp/test.log', 512, 1000)
    assert.strictEqual(slice2.bytesRead, 488) // 1000 - 512 = 488
    assert.strictEqual(slice2.eof, true)
  })

  it('prevents directory traversal outside filesystemRoot', () => {
    assert.throws(() => {
      vfs.resolvePath('../../etc/passwd')
    }, /Access denied/)
  })
})
