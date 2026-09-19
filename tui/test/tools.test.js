const test = require('node:test')
const assert = require('node:assert')
const { ToolRegistry } = require('../lib/tools')
const { ToolParser, parseToolCall } = require('../lib/tools/parser')
const { Orchestrator } = require('../lib/orchestrator')
const { TimerManager } = require('../lib/timers')

test('parseToolCall extracts tool name and json arguments', () => {
  const input = 'Let me check: <|tool_call>call:terminal_write{"input": "cargo build\n", "watch": "completion"}<tool_call|>'
  const parsed = parseToolCall(input)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.name, 'terminal_write')
  assert.strictEqual(parsed.args.input, 'cargo build\n')
  assert.strictEqual(parsed.args.watch, 'completion')
})

test('parseToolCall handles Gemma 4 native quote tokens and unquoted keys', () => {
  const gemmaNative = '<|tool_call>call:read{offset:0,path:<|"|>msg/user/1002.txt<|"|>}<tool_call|>'
  const parsed = parseToolCall(gemmaNative)
  assert.notStrictEqual(parsed, null)
  assert.strictEqual(parsed.name, 'read')
  assert.strictEqual(parsed.args.offset, 0)
  assert.strictEqual(parsed.args.path, 'msg/user/1002.txt')
})

test('ToolRegistry executes cmd and key tools', async () => {
  const mockCmdRunner = {
    run: async (cmd, remind) => ({ exit_code: 0, output: 'ok\n' }),
  }
  const mockTrmManager = {
    key: (trm, k) => ({ status: 'key_sent', trm, key: k }),
  }
  const mockClient = { sendInput: () => {} }

  const registry = ToolRegistry(null, mockCmdRunner, mockTrmManager, null, mockClient, null)
  const res1 = await registry.execute('cmd', { command: 'git status\n' })
  assert.strictEqual(res1.exit_code, 0)
  assert.strictEqual(res1.output, 'ok\n')

  const res2 = await registry.execute('key', { trm: 'dev', name: 'ctrl+c' })
  assert.strictEqual(res2.status, 'key_sent')
  assert.strictEqual(res2.key, 'ctrl+c')

  let memQueried = ''
  let memTopK = 0
  const mockClient2 = {
    sendInput: () => {},
    sendMemQuery: (q, k) => {
      memQueried = q
      memTopK = k
    },
  }
  const registry2 = ToolRegistry(null, null, null, null, mockClient2, null)
  const res3 = await registry2.execute('recall', { query: 'archived thoughts on memory', top_k: 3 })
  assert.strictEqual(res3.status, 'query_submitted')
  assert.strictEqual(memQueried, 'archived thoughts on memory')
  assert.strictEqual(memTopK, 3)
})

test('ToolRegistry executes plan, done, snooze, and ask_user with orchestrator', async () => {
  const timers = TimerManager()
  const orch = Orchestrator(timers)
  let sentPayload = ''
  const mockClient = {
    sendInput: (p) => { sentPayload = p },
  }
  const registry = ToolRegistry(null, null, null, null, mockClient, orch)

  // 1. plan
  const planRes = await registry.execute('plan', {
    brief: 'Build Project',
    steps: ['Check dependencies', 'Compile'],
  })
  assert.strictEqual(planRes.status, 'plan_created')
  assert.strictEqual(planRes.id, 'plan_1001')
  assert.strictEqual(planRes.active_step, 'step_1001.1')

  // 2. ask_user
  const askRes = await registry.execute('ask_user', {
    brief: 'Need sudo permission',
    message: 'Please authorize apt install',
  })
  assert.strictEqual(askRes.status, 'waiting_for_user')
  assert.strictEqual(askRes.step, 'step_1001.1')

  // 3. done
  const doneRes = await registry.execute('done', { summary: 'Dependencies installed' })
  assert.strictEqual(doneRes.status, 'step_completed')
  assert.strictEqual(doneRes.completed_step, 'step_1001.1')
  assert.strictEqual(doneRes.next_step, 'step_1001.2')

  // 4. snooze
  const snoozeRes = await registry.execute('snooze', { id: 'step_1001.2', duration: '500ms' })
  assert.strictEqual(snoozeRes.status, 'step_snoozed')
  assert.strictEqual(snoozeRes.id, 'step_1001.2')
})

test('ToolParser intercepts streaming tool call and dispatches tool return', async () => {
  let returnedName = ''
  let returnedResult = null
  const mockRegistry = {
    execute: async (name, args) => ({ output: `executed ${name}` }),
  }
  const mockClient = {
    sendToolReturn: (name, result) => {
      returnedName = name
      returnedResult = result
    },
  }

  const parser = ToolParser(mockRegistry, mockClient)
  await parser.ingestChunk('I will run this: <|tool_call>call:terminal_reset{}<tool_call|>')

  assert.strictEqual(returnedName, 'terminal_reset')
  assert.deepStrictEqual(returnedResult, { output: 'executed terminal_reset' })
  assert.strictEqual(parser.syntaxTracker.isAtRest(), true)
})
