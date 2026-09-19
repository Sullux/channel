const timerManagerFactory = (
  setTimerFn = setTimeout,
  clearTimerFn = clearTimeout,
  nowFn = Date.now
) => () => {
  const activeTimers = new Map()
  let counter = 0

  const setTimer = (seconds, note, onTrigger) => {
    const id = `timer_${nowFn()}_${++counter}`
    const handle = setTimerFn(() => {
      activeTimers.delete(id)
      onTrigger({ id, note, seconds })
    }, seconds * 1000)

    activeTimers.set(id, { handle, note, seconds, createdAt: nowFn() })
    return { timerId: id, seconds, note }
  }

  const cancelTimer = (timerId) => {
    const entry = activeTimers.get(timerId)
    if (!entry) return { success: false, error: 'Timer not found' }
    clearTimerFn(entry.handle)
    activeTimers.delete(timerId)
    return { success: true, timerId }
  }

  const clearAll = () => {
    for (const [id, entry] of activeTimers.entries()) {
      clearTimerFn(entry.handle)
    }
    activeTimers.clear()
  }

  return {
    setTimer,
    cancelTimer,
    clearAll,
    getActiveCount: () => activeTimers.size,
  }
}

module.exports = {
  timerManagerFactory,
  TimerManager: timerManagerFactory(setTimeout, clearTimeout, Date.now),
}
