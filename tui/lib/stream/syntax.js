const SyntaxTracker = () => {
  let inCodeFence = false
  let inInlineCode = false
  let inDoubleQuote = false
  let parenDepth = 0
  let braceDepth = 0
  let bracketDepth = 0

  const reset = () => {
    inCodeFence = false
    inInlineCode = false
    inDoubleQuote = false
    parenDepth = 0
    braceDepth = 0
    bracketDepth = 0
  }

  const ingestChunk = (chunk) => {
    if (!chunk || typeof chunk !== 'string') return
    for (let i = 0; i < chunk.length; i++) {
      const ch = chunk[i]
      const next2 = chunk.slice(i, i + 3)
      if (next2 === '```') {
        inCodeFence = !inCodeFence
        i += 2
        continue
      }
      if (ch === '`' && !inCodeFence) {
        inInlineCode = !inInlineCode
        continue
      }
      if (inCodeFence || inInlineCode) continue

      if (ch === '"' || ch === '“' || ch === '”') {
        inDoubleQuote = !inDoubleQuote
      } else if (ch === '(') {
        parenDepth++
      } else if (ch === ')' && parenDepth > 0) {
        parenDepth--
      } else if (ch === '{') {
        braceDepth++
      } else if (ch === '}' && braceDepth > 0) {
        braceDepth--
      } else if (ch === '[') {
        bracketDepth++
      } else if (ch === ']' && bracketDepth > 0) {
        bracketDepth--
      }
    }
  }

  const isAtRest = () => (
    !inCodeFence &&
    !inInlineCode &&
    !inDoubleQuote &&
    parenDepth === 0 &&
    braceDepth === 0 &&
    bracketDepth === 0
  )

  const isNaturalBoundary = (text) => {
    if (!text || !isAtRest()) return false
    if (text.endsWith('\n\n')) return true
    const trimmed = text.trimEnd()
    if (trimmed.endsWith('.') || trimmed.endsWith('?') || trimmed.endsWith('!')) {
      return true
    }
    if (trimmed.endsWith(':') && text.endsWith('\n')) return true
    return false
  }

  return {
    reset,
    ingestChunk,
    isAtRest,
    isNaturalBoundary,
    getStackDepth: () => ({
      inCodeFence,
      inInlineCode,
      inDoubleQuote,
      parenDepth,
      braceDepth,
      bracketDepth,
    }),
  }
}

module.exports = { SyntaxTracker }
