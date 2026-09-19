const test = require('node:test')
const assert = require('node:assert')
const { taskManagerFactory, TASK_STATUS } = require('../lib/tasks')

test('TaskManager creates task and marks it active', () => {
  let fakeTime = 1000
  const TaskManager = taskManagerFactory(() => fakeTime)()
  const t1 = TaskManager.createTask('Explain async compute')

  assert.strictEqual(t1.id, 1)
  assert.strictEqual(t1.title, 'Explain async compute')
  assert.strictEqual(t1.status, TASK_STATUS.ACTIVE)
  assert.strictEqual(t1.created, 1000)
  assert.strictEqual(TaskManager.getActiveTask()?.id, 1)
})

test('TaskManager pauses previous active task when new task is activated', () => {
  const TaskManager = taskManagerFactory(() => 2000)()
  const t1 = TaskManager.createTask('Task One')
  const t2 = TaskManager.createTask('Task Two')

  assert.strictEqual(t2.status, TASK_STATUS.ACTIVE)
  assert.strictEqual(TaskManager.getTask(t1.id)?.status, TASK_STATUS.PAUSED)
  assert.strictEqual(TaskManager.getActiveTask()?.id, t2.id)

  TaskManager.setActiveTask(t1.id)
  assert.strictEqual(TaskManager.getTask(t1.id)?.status, TASK_STATUS.ACTIVE)
  assert.strictEqual(TaskManager.getTask(t2.id)?.status, TASK_STATUS.PAUSED)
})

test('TaskManager attaches, updates, and closes read stream cursors', () => {
  const TaskManager = taskManagerFactory(() => 3000)()
  const t1 = TaskManager.createTask('Read big file')

  TaskManager.attachReadStream(t1.id, { path: 'data/log.txt', offset: 0n })
  let task = TaskManager.getTask(t1.id)
  assert.strictEqual(task?.readStream?.path, 'data/log.txt')
  assert.strictEqual(task?.readStream?.offset, 0n)
  assert.strictEqual(task?.readStream?.isReading, true)

  TaskManager.updateReadStream(t1.id, { offset: 512n, bytesRead: 512 })
  task = TaskManager.getTask(t1.id)
  assert.strictEqual(task?.readStream?.offset, 512n)
  assert.strictEqual(task?.readStream?.bytesRead, 512)

  TaskManager.closeReadStream(t1.id)
  task = TaskManager.getTask(t1.id)
  assert.strictEqual(task?.readStream, undefined)
})

test('TaskManager completes task and activates next candidate', () => {
  const TaskManager = taskManagerFactory(() => 4000)()
  const t1 = TaskManager.createTask('First Task')
  const t2 = TaskManager.createTask('Second Task')

  TaskManager.completeTask(t2.id)
  assert.strictEqual(TaskManager.getTask(t2.id)?.status, TASK_STATUS.DONE)
  assert.strictEqual(TaskManager.getActiveTask()?.id, t1.id)
  assert.strictEqual(TaskManager.getTask(t1.id)?.status, TASK_STATUS.ACTIVE)
})

test('TaskManager formatCandidates returns active candidate list', () => {
  const TaskManager = taskManagerFactory(() => 5000)()
  TaskManager.createTask('Refactor memory engine')
  TaskManager.createTask('Benchmark GEMV')

  const candidates = TaskManager.formatCandidates()
  assert.deepStrictEqual(candidates, [
    { id: 1, title: 'Refactor memory engine' },
    { id: 2, title: 'Benchmark GEMV' },
  ])
})

test('TaskManager routes steering events to active task and distinct events to new task', () => {
  const TaskManager = taskManagerFactory(() => 6000)()
  const t1 = TaskManager.createTask('Summarize 10GB file')
  assert.strictEqual(TaskManager.getActiveTask()?.id, t1.id)

  // Simulated triage outcome 1: event routed to existing Task 1 (steering)
  const routedEvent = { eventId: 103, taskId: t1.id, isNewTask: false }
  if (!routedEvent.isNewTask) {
    TaskManager.setActiveTask(routedEvent.taskId)
  }
  assert.strictEqual(TaskManager.getActiveTask()?.id, t1.id)
  assert.strictEqual(TaskManager.getTask(t1.id)?.status, TASK_STATUS.ACTIVE)

  // Simulated triage outcome 2: event routed as new task
  const newEvent = { eventId: 104, taskId: 0, isNewTask: true }
  if (newEvent.isNewTask) {
    TaskManager.createTask('Check weather in Tokyo')
  }
  assert.strictEqual(TaskManager.getActiveTask()?.id, 2)
  assert.strictEqual(TaskManager.getTask(t1.id)?.status, TASK_STATUS.PAUSED)
})

test('TaskManager fires onChange callback and serializes all tasks', () => {
  const changes = []
  const TaskManager = taskManagerFactory(() => 7000)(
    [{ id: 5, title: 'Existing Task', status: TASK_STATUS.ACTIVE }],
    (tasks) => changes.push(tasks),
  )

  const t2 = TaskManager.createTask('Second Task')
  assert.strictEqual(t2.id, 6)
  assert.strictEqual(changes.length, 1)
  assert.strictEqual(changes[0].length, 2)
  assert.strictEqual(changes[0][1].title, 'Second Task')
})
