const { describe, it, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert')
const fs = require('fs')
const path = require('path')
const { vfsFactory } = require('../lib/vfs')
const { commandRunnerFactory } = require('../lib/cmd')
const { TimerManager } = require('../lib/timers')

const TEST_ROOT = path.resolve(__dirname, './.test_agent_cmd')

describe('CommandRunner (Subshell Execution)', () => {
  let vfs
  let timers
  let notifications
  let runner

  beforeEach(() => {
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
    vfs = vfsFactory()(TEST_ROOT)
    vfs.init()
    timers = TimerManager()
    notifications = []
    runner = commandRunnerFactory(Date.now)(vfs, timers, (n) => notifications.push(n))
  })

  afterEach(() => {
    runner.killAll()
    timers.clearAll()
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
  })

  it('fast path: returns inline if <= 250ms and <= 128 chars', async () => {
    const res = await runner.run('echo "hello fast path"')
    assert.strictEqual(res.exit_code, 0)
    assert.strictEqual(res.output.trim(), 'hello fast path')
    assert.strictEqual(res.status, undefined)
  })

  it('spillover path: returns preview + log if <= 250ms and > 128 chars', async () => {
    // Generate 300 chars
    const res = await runner.run('python3 -c \'print("X" * 300)\'')
    assert.strictEqual(res.exit_code, 0)
    assert.strictEqual(res.preview.length, 128)
    assert.ok(res.log.endsWith('.stdout.log'))
    assert.strictEqual(fs.existsSync(vfs.resolvePath(res.log)), true)
  })

  it('async detach: detaches if > 250ms and notifies upon completion', async () => {
    const res = await runner.run('sleep 0.4 && echo "done delayed"', '1m')
    assert.strictEqual(res.status, 'running')
    assert.ok(res.id.startsWith('cmd_'))
    assert.ok(res.pid > 0)

    // Wait for process to finish
    await new Promise((r) => setTimeout(r, 500))

    assert.strictEqual(notifications.length, 1)
    assert.strictEqual(notifications[0].type, 'CMD_COMPLETE')
    assert.strictEqual(notifications[0].exit_code, 0)
    assert.ok(notifications[0].preview.includes('finished with code 0'))
  })

  it('can kill active running command', async () => {
    const res = await runner.run('sleep 10')
    assert.strictEqual(res.status, 'running')
    const killRes = runner.kill(res.id)
    assert.strictEqual(killRes.status, 'killed')
  })
})
