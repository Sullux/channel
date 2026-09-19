const { describe, it, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert')
const fs = require('fs')
const path = require('path')
const { vfsFactory } = require('../lib/vfs')
const { terminalManagerFactory } = require('../lib/terminal/manager')

const TEST_ROOT = path.resolve(__dirname, './.test_agent_trm')

describe('TerminalManager (Persistent Sessions)', () => {
  let vfs
  let notifications
  let trm

  beforeEach(() => {
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
    vfs = vfsFactory()(TEST_ROOT)
    vfs.init()
    notifications = []
    trm = terminalManagerFactory(Date.now)(vfs, (n) => notifications.push(n))
  })

  afterEach(() => {
    trm.closeAll()
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
  })

  it('opens named terminal session and creates screen.txt and stdout.log', async () => {
    const res = trm.open('dev', 'echo "hello pty"')
    assert.strictEqual(res.name, 'trm_dev')
    assert.strictEqual(res.status, 'opened')

    const screenAbs = vfs.resolvePath(res.screen)
    const stdoutAbs = vfs.resolvePath(res.stdout)

    assert.strictEqual(fs.existsSync(screenAbs), true)
    assert.strictEqual(fs.existsSync(stdoutAbs), true)

    // Wait briefly for shell output
    await new Promise((r) => setTimeout(r, 300))

    const screenText = trm.getScreen('dev').screen
    assert.ok(screenText !== undefined)
  })

  it('can send control keys via key()', () => {
    trm.open('repl')
    const res = trm.key('repl', 'ctrl+c')
    assert.strictEqual(res.status, 'key_sent')
    assert.strictEqual(res.key, 'ctrl+c')
  })

  it('can close named terminal session', () => {
    trm.open('server')
    const closeRes = trm.close('server')
    assert.strictEqual(closeRes.status, 'closed')
    assert.strictEqual(closeRes.name, 'trm_server')
  })
})
