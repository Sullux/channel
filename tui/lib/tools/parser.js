const { SyntaxTracker } = require('../stream/syntax')

const TOOL_CALL_START = '<|tool_call>'
const TOOL_CALL_END = '<tool_call|>'

const parseToolCall = (str) => {
  const startIdx = str.indexOf(TOOL_CALL_START)
  if (startIdx === -1) return null
  const endIdx = str.indexOf(TOOL_CALL_END, startIdx)
  if (endIdx === -1) return null

  const raw = str.slice(startIdx + TOOL_CALL_START.length, endIdx).trim()
  const colonIdx = raw.indexOf(':')
  if (colonIdx === -1) return null

  const nameAndArgs = raw.slice(colonIdx + 1)
  const braceIdx = nameAndArgs.indexOf('{')
  if (braceIdx === -1) return { name: nameAndArgs.trim(), args: {} }

  const name = nameAndArgs.slice(0, braceIdx).trim()
  const argsStr = nameAndArgs.slice(braceIdx)
  try {
    return { name, args: JSON.parse(argsStr) }
  } catch (_) {
    try {
      const sanitized = argsStr
        .replace(/\r?\n/g, '\\n')
        .replaceAll('\u2581', ' ')
        .replaceAll('<|"|>', '"')
        .replace(/([{\s,])([a-zA-Z0-9_]+)\s*:/g, '$1"$2":')
      return { name, args: JSON.parse(sanitized) }
    } catch (_) {
      return { name, args: {} }
    }
  }
}

const toolParserFactory = () => (registry, client, store) => {
  let streamAccumulator = ''
  const syntaxTracker = SyntaxTracker()

  const executeCall = async (name, args) => {
    store?.flushActiveThought?.()
    store?.flushActiveResponse?.()

    store?.addStreamEntry?.({
      type: 'tool_call',
      title: `🛠️ TOOL: ${name}`,
      content: JSON.stringify(args, null, 2),
    })

    const result = await registry.execute(name, args)

    if (name === 'ask_user') {
      const msg = args.message || args.prompt || args.brief || 'User action required'
      store?.addConversationMessage?.({
        sender: 'Assistant',
        text: msg,
        waitingUser: true,
      })
    }

    store?.addStreamEntry?.({
      type: 'tool_result',
      title: `📦 RESULT: ${name}`,
      content: JSON.stringify(result, null, 2),
    })

    if (client?.sendToolReturn) {
      client.sendToolReturn(name, result)
    } else {
      client.sendInput(JSON.stringify(result))
    }
  }

  client?.on?.('toolCall', async ({ toolName, argsJson }) => {
    let args = {}
    let parseError = null
    try {
      args = JSON.parse(argsJson)
    } catch (e1) {
      try {
        const cleaned = argsJson
          .replace(/\r?\n/g, '\\n')
          .replaceAll('\u2581', ' ')
          .replaceAll('<|"|>', '"')
          .replace(/([{\s,])([a-zA-Z0-9_]+)\s*:/g, '$1"$2":')
        args = JSON.parse(cleaned)
      } catch (e2) {
        parseError = `Malformed JSON arguments for tool \`${toolName}\`: ${argsJson}`
      }
    }
    if (parseError) {
      await executeCall(toolName, { error: parseError })
    } else {
      await executeCall(toolName, args)
    }
  })

  const ingestChunk = async (chunk) => {
    syntaxTracker.ingestChunk(chunk)
    streamAccumulator += chunk
    const parsed = parseToolCall(streamAccumulator)
    if (!parsed) return

    const endIdx = streamAccumulator.indexOf(TOOL_CALL_END)
    streamAccumulator = streamAccumulator.slice(endIdx + TOOL_CALL_END.length)

    await executeCall(parsed.name, parsed.args)
  }

  const reset = () => {
    streamAccumulator = ''
    syntaxTracker.reset()
  }

  return { ingestChunk, reset, syntaxTracker }
}

module.exports = {
  parseToolCall,
  toolParserFactory,
  ToolParser: toolParserFactory(),
}
