const test = require('node:test')
const assert = require('node:assert')
const { TerminalBuffer } = require('../lib/terminal/buffer')

test('TerminalBuffer writes characters and line feeds correctly', () => {
  const buf = TerminalBuffer({ rows: 5, cols: 20 })
  buf.write('Hello\nWorld\n')
  const screen = buf.screenText().split('\n')
  assert.strictEqual(screen[0], 'Hello')
  assert.strictEqual(screen[1], 'World')
  assert.strictEqual(buf.getScrollbackLength(), 2)
})

test('TerminalBuffer strips ANSI escapes and supports carriage return', () => {
  const buf = TerminalBuffer({ rows: 5, cols: 20 })
  buf.write('\x1b[31mRed\x1b[0m\rBlue')
  const screen = buf.screenText().split('\n')
  assert.strictEqual(screen[0], 'Blue')
})

test('TerminalBuffer search finds patterns in scrollback', () => {
  const buf = TerminalBuffer({ rows: 5, cols: 40 })
  buf.write('line 1 ok\n')
  buf.write('line 2 ERROR: bad token\n')
  buf.write('line 3 ok\n')
  const res = buf.search('error')
  assert.strictEqual(res.length, 1)
  assert.strictEqual(res[0].lineIndex, 1)
  assert.strictEqual(res[0].text, 'line 2 ERROR: bad token')
})

test('TerminalBuffer pageSlice retrieves scrollback pages', () => {
  const buf = TerminalBuffer({ rows: 2, cols: 20 })
  for (let i = 1; i <= 6; i++) buf.write(`line ${i}\n`)
  const page1 = buf.pageSlice(-1)
  assert.strictEqual(page1.length, 2)
  assert.strictEqual(page1[0], 'line 5')
})
