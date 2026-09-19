const TASK_STATUS = {
  ACTIVE: 'ACTIVE',
  PAUSED: 'PAUSED',
  DONE: 'DONE',
}

const taskManagerFactory = (now = Date.now) => (initialTasks = [], onChange) => {
  let seq = 1
  const tasks = new Map(
    initialTasks.map((t) => {
      seq = Math.max(seq, (t.id ?? 0) + 1)
      const task = { ...t }
      if (task.readStream?.offset !== undefined) {
        task.readStream = { ...task.readStream, offset: BigInt(task.readStream.offset) }
      }
      return [t.id, task]
    }),
  )
  let activeTaskId = initialTasks.find((t) => t.status === TASK_STATUS.ACTIVE)?.id

  const getAllTasks = () =>
    Array.from(tasks.values()).map((t) => ({
      ...t,
      readStream: t.readStream
        ? { ...t.readStream, offset: t.readStream.offset.toString() }
        : undefined,
    }))

  const notifyChange = () => {
    if (typeof onChange === 'function') {
      onChange(getAllTasks())
    }
  }

  const createTask = (title) => {
    const id = seq++
    for (const [, t] of tasks) {
      if (t.status === TASK_STATUS.ACTIVE) {
        t.status = TASK_STATUS.PAUSED
      }
    }
    const task = {
      id,
      title,
      status: TASK_STATUS.ACTIVE,
      created: now(),
      readStream: undefined,
      planSteps: [],
    }
    tasks.set(id, task)
    activeTaskId = id
    notifyChange()
    return { ...task }
  }

  const getTask = (id) => tasks.get(id)

  const getActiveTask = () => (activeTaskId ? tasks.get(activeTaskId) : undefined)

  const setActiveTask = (id) => {
    if (!tasks.has(id)) return undefined
    for (const [, t] of tasks) {
      if (t.status === TASK_STATUS.ACTIVE && t.id !== id) {
        t.status = TASK_STATUS.PAUSED
      }
    }
    const target = tasks.get(id)
    target.status = TASK_STATUS.ACTIVE
    activeTaskId = id
    notifyChange()
    return { ...target }
  }

  const getActiveTasks = () =>
    Array.from(tasks.values()).filter((t) => t.status !== TASK_STATUS.DONE)

  const updateTask = (id, updates) => {
    const task = tasks.get(id)
    if (!task) return undefined
    Object.assign(task, updates)
    notifyChange()
    return { ...task }
  }

  const attachReadStream = (id, { path, offset = 0n }) => {
    const task = tasks.get(id)
    if (!task) return undefined
    task.readStream = {
      path,
      offset: BigInt(offset),
      bytesRead: 0,
      isReading: true,
    }
    notifyChange()
    return { ...task.readStream }
  }

  const updateReadStream = (id, { offset, bytesRead = 0, isReading = true }) => {
    const task = tasks.get(id)
    if (!task?.readStream) return undefined
    if (offset !== undefined) task.readStream.offset = BigInt(offset)
    task.readStream.bytesRead += bytesRead
    task.readStream.isReading = isReading
    notifyChange()
    return { ...task.readStream }
  }

  const closeReadStream = (id) => {
    const task = tasks.get(id)
    if (!task) return undefined
    task.readStream = undefined
    notifyChange()
    return true
  }

  const completeTask = (id) => {
    const task = tasks.get(id)
    if (!task) return undefined
    task.status = TASK_STATUS.DONE
    if (task.readStream) task.readStream = undefined
    if (activeTaskId === id) {
      const next = getActiveTasks().find((t) => t.id !== id)
      activeTaskId = next?.id
      if (next) next.status = TASK_STATUS.ACTIVE
    }
    notifyChange()
    return { ...task }
  }

  const formatCandidates = () =>
    getActiveTasks().map((t) => ({ id: t.id, title: t.title }))

  return {
    createTask,
    getTask,
    getActiveTask,
    setActiveTask,
    getActiveTasks,
    getAllTasks,
    updateTask,
    attachReadStream,
    updateReadStream,
    closeReadStream,
    completeTask,
    formatCandidates,
  }
}

module.exports = {
  TASK_STATUS,
  taskManagerFactory,
  TaskManager: taskManagerFactory(Date.now),
}
