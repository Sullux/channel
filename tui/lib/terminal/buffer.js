const { strippedAnsi } = require('./ansi')

const bufferFactory = () => (options = {}) => {
  const maxRows = options.rows || 30
  const maxCols = options.cols || 100
  const maxHistory = options.maxHistory || 5000

  let scrollback = []
  let screen = Array.from({ length: maxRows }, () => ' '.repeat(maxCols))
  let cursorRow = 0
  let cursorCol = 0

  const appendToScrollback = (line) => {
    scrollback.push(line)
    if (scrollback.length > maxHistory) scrollback.shift()
  }

  const writeChar = (ch) => {
    if (ch === '\n') {
      appendToScrollback(screen[cursorRow].trimEnd())
      cursorRow++
      if (cursorRow >= maxRows) {
        screen.shift()
        screen.push(' '.repeat(maxCols))
        cursorRow = maxRows - 1
      }
      cursorCol = 0
      return
    }
    if (ch === '\r') {
      cursorCol = 0
      return
    }
    if (ch === '\b') {
      if (cursorCol > 0) cursorCol--
      return
    }
    if (cursorCol < maxCols && cursorRow < maxRows) {
      const line = screen[cursorRow]
      screen[cursorRow] = line.slice(0, cursorCol) + ch + line.slice(cursorCol + 1)
      cursorCol++
    }
  }

  const write = (text) => {
    const clean = strippedAnsi(text)
    for (let i = 0; i < clean.length; i++) {
      writeChar(clean[i])
    }
  }

  const screenText = () => screen.map((l) => l.trimEnd()).join('\n')

  const scrollbackSlice = (lineStart = 0, lineCount = maxRows) => {
    const start = Math.max(0, Math.min(lineStart, scrollback.length))
    return scrollback.slice(start, start + lineCount)
  }

  const pageSlice = (pageOffset = 0) => {
    const totalLines = scrollback.length
    const offset = Math.max(0, totalLines + pageOffset * maxRows)
    return scrollback.slice(offset, offset + maxRows)
  }

  const search = (pattern, maxResults = 10) => {
    const regex = new RegExp(pattern, 'i')
    const results = []
    for (let i = scrollback.length - 1; i >= 0 && results.length < maxResults; i--) {
      if (regex.test(scrollback[i])) {
        results.push({ lineIndex: i, text: scrollback[i] })
      }
    }
    return results
  }

  const reset = () => {
    scrollback = []
    screen = Array.from({ length: maxRows }, () => ' '.repeat(maxCols))
    cursorRow = 0
    cursorCol = 0
  }

  return {
    write,
    screenText,
    scrollbackSlice,
    pageSlice,
    search,
    reset,
    getCursor: () => ({ row: cursorRow, col: cursorCol }),
    getScrollbackLength: () => scrollback.length,
  }
}

module.exports = {
  bufferFactory,
  TerminalBuffer: bufferFactory(),
}
