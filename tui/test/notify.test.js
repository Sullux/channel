const { describe, it, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert')
const { notificationManagerFactory } = require('../lib/notify')
const { TimerManager } = require('../lib/timers')

describe('NotificationManager (Interrupt Queue)', () => {
  let timers
  let notManager

  beforeEach(() => {
    timers = TimerManager()
    notManager = notificationManagerFactory(Date.now)(timers)
  })

  afterEach(() => {
    timers.clearAll()
  })

  it('enqueues pending notification with not prefix without underscore', () => {
    const item = notManager.notify('/msg/user/1042.txt', 'DB down', '1042')
    assert.strictEqual(item.id, 'not101')
    assert.strictEqual(item.status, 'PENDING')
    assert.strictEqual(notManager.getPending().length, 1)
  })

  it('formats turn alerts rollup correctly', () => {
    notManager.notify('/msg/user/1042.txt', 'DB down', '1042')
    notManager.notify('/sys/cmd/cmd_101', 'Build ok', 'cmd_101')

    const rollup = notManager.formatTurnAlerts()
    assert.ok(rollup.includes('[Pending Alerts:'))
    assert.ok(rollup.includes('[Event: not101 | Source: /msg/user/1042.txt]: DB down'))
    assert.ok(rollup.includes('[Event: not102 | Source: /sys/cmd/cmd_101]: Build ok'))
  })

  it('ack() permanently removes notification from pending queue', () => {
    const item = notManager.notify('/sys/cmd/cmd_101', 'Build ok', 'cmd_101')
    const res = notManager.ack(item.id)
    assert.strictEqual(res.status, 'acknowledged')
    assert.strictEqual(notManager.getPending().length, 0)
  })

  it('snooze() temporarily suppresses notification from pending queue', () => {
    const item = notManager.notify('/sys/cmd/cmd_101', 'Build ok', 'cmd_101')
    const res = notManager.snooze(item.id, '30s')
    assert.strictEqual(res.status, 'snoozed')
    assert.strictEqual(notManager.getPending().length, 0)
    assert.strictEqual(notManager.getSnoozed().length, 1)

    // Manual wake test
    notManager.wake(item.id)
    assert.strictEqual(notManager.getPending().length, 1)
  })
})
