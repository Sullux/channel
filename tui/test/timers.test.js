const test = require('node:test')
const assert = require('node:assert')
const { timerManagerFactory } = require('../lib/timers')

test('TimerManager registers and triggers timeout', (t, done) => {
  let triggered = false
  const mockSetTimeout = (fn, delay) => {
    assert.strictEqual(delay, 5000)
    process.nextTick(() => {
      fn()
      assert.strictEqual(triggered, true)
      done()
    })
    return 123
  }
  const mockClearTimeout = () => {}
  const Timers = timerManagerFactory(mockSetTimeout, mockClearTimeout, () => 1000)
  const mgr = Timers()

  const res = mgr.setTimer(5, 'check build', ({ note }) => {
    triggered = true
    assert.strictEqual(note, 'check build')
  })

  assert.strictEqual(res.note, 'check build')
  assert.strictEqual(res.timerId, 'timer_1000_1')
})

test('TimerManager cancels pending timer', () => {
  let cleared = false
  const mockSetTimeout = () => 456
  const mockClearTimeout = (handle) => {
    assert.strictEqual(handle, 456)
    cleared = true
  }
  const Timers = timerManagerFactory(mockSetTimeout, mockClearTimeout, () => 2000)
  const mgr = Timers()

  const { timerId } = mgr.setTimer(10, 'long task', () => {})
  const cancelRes = mgr.cancelTimer(timerId)

  assert.strictEqual(cancelRes.success, true)
  assert.strictEqual(cleared, true)
  assert.strictEqual(mgr.getActiveCount(), 0)
})
