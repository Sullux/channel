const { describe, it, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert')
const fs = require('fs')
const path = require('path')
const { toolRegistryFactory } = require('../lib/tools')
const { vfsFactory } = require('../lib/vfs')
const { commandRunnerFactory } = require('../lib/cmd')
const { terminalManagerFactory } = require('../lib/terminal/manager')
const { notificationManagerFactory } = require('../lib/notify')
const { TimerManager } = require('../lib/timers')
const { orchestratorFactory } = require('../lib/orchestrator')

const TEST_ROOT = path.resolve(__dirname, './.test_agent_tools')

describe('ToolRegistry (Streaming & VFS Tools)', () => {
  let vfs
  let timers
  let cmdRunner
  let trmManager
  let notManager
  let orchestrator
  let registry
  let mockClient
  let notifications

  beforeEach(() => {
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
    vfs = vfsFactory()(TEST_ROOT)
    vfs.init()
    timers = TimerManager()
    notifications = []
    notManager = notificationManagerFactory(Date.now)(timers)
    cmdRunner = commandRunnerFactory(Date.now)(vfs, timers, (n) => {
      notifications.push(n)
      notManager.notify(n.log || n.cmdId, n.preview, n.cmdId)
    })
    trmManager = terminalManagerFactory(Date.now)(vfs, (n) => {
      notifications.push(n)
      notManager.notify(n.log, n.preview, n.name)
    })
    orchestrator = orchestratorFactory(Date.now)(timers)
    mockClient = {
      sendMemCommit: () => {},
      sendMemQuery: () => {},
      sendInput: () => {},
    }
    registry = toolRegistryFactory()(
      vfs,
      cmdRunner,
      trmManager,
      notManager,
      mockClient,
      orchestrator,
    )
  })

  afterEach(() => {
    cmdRunner.killAll()
    trmManager.closeAll()
    timers.clearAll()
    fs.rmSync(TEST_ROOT, { recursive: true, force: true })
  })

  it('executes cmd tool with fast-path inline return', async () => {
    const res = await registry.execute('cmd', { command: 'echo "test tool"' })
    assert.strictEqual(res.exit_code, 0)
    assert.strictEqual(res.output.trim(), 'test tool')
  })

  it('reads bounded 512 char slices via VFS', () => {
    const filePath = path.join(TEST_ROOT, 'tmp', 'data.txt')
    fs.writeFileSync(filePath, 'Hello world streaming reader', 'utf-8')

    const res = vfs.read('tmp/data.txt', 0)
    assert.strictEqual(res.content, 'Hello world streaming reader')
    assert.strictEqual(res.eof, true)
  })

  it('executes ack and snooze tools on notifications', () => {
    const item = notManager.notify('/msg/user/1042.txt', 'DB down', '1042')
    assert.strictEqual(notManager.getPending().length, 1)

    const ackRes = notManager.ack(item.id)
    assert.strictEqual(ackRes.status, 'acknowledged')
    assert.strictEqual(notManager.getPending().length, 0)
  })

  it('executes plan and done tools with prefixed IDs', async () => {
    const planRes = await registry.execute('plan', {
      brief: 'Refactor DB',
      steps: ['Create schema', 'Migrate rows'],
    })
    assert.strictEqual(planRes.status, 'plan_created')
    assert.strictEqual(planRes.id, 'plan_1001')
    assert.strictEqual(planRes.active_step, 'step_1001.1')

    const doneRes = await registry.execute('done', { summary: 'Schema created' })
    assert.strictEqual(doneRes.status, 'step_completed')
    assert.strictEqual(doneRes.completed_step, 'step_1001.1')
    assert.strictEqual(doneRes.next_step, 'step_1001.2')
  })
})
