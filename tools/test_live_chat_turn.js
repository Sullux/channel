const fs = require('node:fs')
const path = require('node:path')
const { Client } = require('../tui/lib/client')

const promptKernel = fs.readFileSync(path.resolve(__dirname, '../tui/PROMPT_KERNEL.md'), 'utf-8')
const userPrompt = fs.readFileSync(path.resolve(__dirname, '../tui/PROMPT.md'), 'utf-8')
const fullSystem = `${userPrompt}\n\n${promptKernel}`

const modelPath = path.resolve(__dirname, '../../gemma-4-12B-it-qat-q4_0-unquantized')
const binPath = path.resolve(__dirname, '../zig-out/bin/infer')

console.log('Spawning client...')
const client = Client({
  binaryPath: binPath,
  modelPath,
  extraArgs: ['--gpu', '--q4', '--quiescence-threshold', '0.0'],
})

client.start()

const promptText = `<|turn>system\n<|think|>\n${fullSystem}\n<turn|>\n<|turn>user\nI am testing my new orchestration framework and inference engine. To start with, tell me about the low-level tools you currently have available to you.\n<turn|>\n<|turn>model\n`

console.log('Sending config and prompt...')
client.setConfig({
  thinkingBudget: 512,
  maxTokens: 512,
  temp: 0.7,
  topP: 0.95,
  minP: 0.05,
  repeatPenalty: 1.1,
  qThresh: 0.0,
})

client.sendInput(promptText)

let thoughtCount = 0
let contentCount = 0

client.on('thought', ({ text }) => {
  thoughtCount += 1
  process.stdout.write(`\x1b[35m${text}\x1b[0m`)
})

client.on('content', ({ text }) => {
  contentCount += 1
  process.stdout.write(`\x1b[32m${text}\x1b[0m`)
})

client.on('status', ({ tokSec, currentTok, status }) => {
  // console.log(`[Status] status=${status}, currentTok=${currentTok}, tokSec=${tokSec}`)
})

client.on('turnComplete', ({ tokSec, elapsedMs, totalTok }) => {
  console.log(`\n\n--- TURN COMPLETE ---`)
  console.log(`Speed: ${tokSec.toFixed(1)} tok/s, Total: ${totalTok} tokens in ${elapsedMs}ms`)
  console.log(`Thought chunks: ${thoughtCount}, Content chunks: ${contentCount}`)
  client.shutdown()
  setTimeout(() => process.exit(0), 100)
})

client.on('error', (err) => {
  console.error('Client error:', err)
  client.shutdown()
  process.exit(1)
})
