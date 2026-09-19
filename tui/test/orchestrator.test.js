const { describe, it } = require('node:test')
const assert = require('node:assert')
const { orchestratorFactory, parseDurationMs } = require('../lib/orchestrator')
const { TimerManager } = require('../lib/timers')

describe('Orchestrator & Task Stack', () => {
  it('parses duration strings accurately', () => {
    assert.strictEqual(parseDurationMs('500ms'), 500)
    assert.strictEqual(parseDurationMs('30s'), 30000)
    assert.strictEqual(parseDurationMs('2m'), 120000)
    assert.strictEqual(parseDurationMs('1h'), 3600000)
    assert.strictEqual(parseDurationMs('invalid'), 10000)
  })

  it('creates hierarchical plans and pushes steps to stack in LIFO order', () => {
    let mockTime = 1000
    const Orchestrator = orchestratorFactory(() => mockTime)
    const orch = Orchestrator()

    const plan = orch.createPlan('Update packages', [
      'Scan packages',
      'Generate script',
      'Execute script',
    ])

    assert.strictEqual(plan.id, 'plan_1001')
    assert.strictEqual(plan.steps.length, 3)
    assert.strictEqual(plan.steps[0].id, 'step_1001.1')
    assert.strictEqual(plan.steps[1].id, 'step_1001.2')
    assert.strictEqual(plan.steps[2].id, 'step_1001.3')

    // Active step should be step 1
    const active = orch.getActiveStep()
    assert.strictEqual(active.id, 'step_1001.1')
    assert.strictEqual(active.status, 'IN_PROGRESS')
  })

  it('completes steps and cascades to parent plan completion', () => {
    let mockTime = 1000
    const Orchestrator = orchestratorFactory(() => mockTime)
    const orch = Orchestrator()

    orch.createPlan('Short task', ['Step 1', 'Step 2'])

    const step1Result = orch.completeActiveStep('Finished step 1')
    assert.strictEqual(step1Result.completedStep.id, 'step_1001.1')
    assert.strictEqual(step1Result.completedStep.status, 'DONE')
    assert.strictEqual(step1Result.nextStep.id, 'step_1001.2')

    const step2Result = orch.completeActiveStep('Finished step 2')
    assert.strictEqual(step2Result.completedStep.id, 'step_1001.2')
    assert.strictEqual(step2Result.nextStep, null)

    assert.strictEqual(orch.plans[0].status, 'DONE')
  })

  it('handles deferring steps with timer integration', () => {
    let mockTime = 1000
    const timers = TimerManager()
    const Orchestrator = orchestratorFactory(() => mockTime)
    const orch = Orchestrator(timers)

    orch.createPlan('Long build', ['Compile code', 'Verify binary'])
    let wokeUp = false

    const deferred = orch.deferActiveStep('1s', 'compiling LLVM', (step) => {
      wokeUp = true
      assert.strictEqual(step.id, 'step_1001.1')
    })
    assert.strictEqual(deferred.deferredStep.status, 'DEFERRED')
    assert.strictEqual(deferred.durationMs, 1000)

    // While deferred, active step should be step 2
    const activeNext = orch.getActiveStep()
    assert.strictEqual(activeNext.id, 'step_1001.2')
    timers.clearAll()
  })

  it('handles ask_user blocking state and wake-up thought generation', () => {
    let mockTime = 1000
    const Orchestrator = orchestratorFactory(() => mockTime)
    const orch = Orchestrator()

    orch.createPlan('Install software', ['Check sudo', 'Install curl'])
    const active = orch.getActiveStep()
    assert.strictEqual(active.id, 'step_1001.1')

    orch.askUserActiveStep('Need sudo access', 'Please run sudo apt update')
    assert.strictEqual(active.status, 'WAITING_FOR_USER')
    assert.strictEqual(active.waitingForUser.brief, 'Need sudo access')

    const waiting = orch.getWaitingForUserTasks()
    assert.strictEqual(waiting.length, 1)
    assert.strictEqual(waiting[0].id, 'step_1001.1')

    // Thought prefix should include the waiting task
    const thought = orch.buildThoughtPrefix('USER_PROMPT', { message: 'Done, sudo ran' })
    assert(thought.includes('step_1001.1: Need sudo access'))
    assert(thought.includes('Done, sudo ran'))
  })

  it('auto-wraps ad-hoc tool calls when no active plan exists', () => {
    let mockTime = 1000
    const Orchestrator = orchestratorFactory(() => mockTime)
    const orch = Orchestrator()

    const step = orch.autoWrapToolCall('terminal_write', 'git status')
    assert.strictEqual(step.brief, 'terminal_write: git status')
    assert.strictEqual(step.status, 'IN_PROGRESS')
    assert.strictEqual(orch.plans.length, 1)
  })

  it('injects cognitive saturation consolidation directive and clears upon plan creation', () => {
    let mockTime = 1000
    const Orchestrator = orchestratorFactory(() => mockTime)
    const orch = Orchestrator()

    // When not saturated, thought prefix is standard
    assert.strictEqual(orch.isSaturated(), false)
    const prefix1 = orch.buildThoughtPrefix('USER_PROMPT', { message: 'hi' })
    assert.strictEqual(prefix1.includes('capacity threshold reached'), false)

    // Saturated condition triggered by backend
    orch.setSaturated(true)
    assert.strictEqual(orch.isSaturated(), true)
    const prefix2 = orch.buildThoughtPrefix('USER_PROMPT', { message: 'hi' })
    assert.strictEqual(prefix2.includes('Working memory capacity threshold reached'), true)

    // Model creates a plan -> clears saturated state
    orch.createPlan('Consolidated architecture plan', ['Step 1', 'Step 2'])
    assert.strictEqual(orch.isSaturated(), false)
    const prefix3 = orch.buildThoughtPrefix('USER_PROMPT', { message: 'next' })
    assert.strictEqual(prefix3.includes('capacity threshold reached'), false)
  })
})
