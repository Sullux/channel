const { spawn } = require('child_process')
const { bufferFactory } = require('./buffer')

const KEY_MAP = {
  'ctrl+c': '\x03',
  'ctrl+d': '\x04',
  'ctrl+z': '\x1a',
  enter: '\n',
  tab: '\t',
  up: '\x1b[A',
  down: '\x1b[B',
  escape: '\x1b',
  space: ' ',
}

const sessionFactory = (spawnProc = spawn, BufferFactory = bufferFactory()) => (options = {}) => {
  const shell = options.shell || 'bash'
  const cwd = options.cwd || process.cwd()
  const buffer = BufferFactory(options)
  let proc = null
  let debounceTimer = null
  let watchCallback = null

  const spawnShell = () => {
    proc = spawnProc(shell, ['--norc', '+m'], {
      cwd,
      env: { ...process.env, TERM: 'xterm-256color', COLUMNS: '100', LINES: '30' },
      stdio: ['pipe', 'pipe', 'pipe'],
    })

    const onData = (chunk) => {
      const text = chunk.toString('utf-8')
      buffer.write(text)
      if (watchCallback) {
        clearTimeout(debounceTimer)
        debounceTimer = setTimeout(() => {
          if (watchCallback) watchCallback(buffer.screenText())
        }, options.debounceMs || 200)
      }
    }

    proc.stdout.on('data', onData)
    proc.stderr.on('data', onData)
  }

  spawnShell()

  const write = (input) => {
    if (proc?.stdin?.writable) proc.stdin.write(input)
  }

  const sendKey = (key) => {
    const seq = KEY_MAP[key?.toLowerCase()]
    if (seq && proc?.stdin?.writable) proc.stdin.write(seq)
  }

  const setWatch = (callback) => {
    watchCallback = callback
  }

  const reset = () => {
    if (proc) {
      proc.kill('SIGKILL')
      proc = null
    }
    buffer.reset()
    spawnShell()
  }

  const kill = () => {
    clearTimeout(debounceTimer)
    watchCallback = null
    if (proc) proc.kill('SIGKILL')
  }

  return {
    write,
    sendKey,
    setWatch,
    reset,
    kill,
    buffer,
    getScreenText: buffer.screenText,
  }
}

module.exports = {
  sessionFactory,
  TerminalSession: sessionFactory(spawn, bufferFactory()),
}
