const fs = require('fs')
const path = require('path')
const { TerminalSession } = require('./session')

const KEY_MAP = {
  'ctrl+c': 'ctrl+c',
  'ctrl+d': 'ctrl+d',
  'ctrl+z': 'ctrl+z',
  enter: 'enter',
  tab: 'tab',
  up: 'up',
  down: 'down',
  esc: 'escape',
  escape: 'escape',
  space: 'space',
}

const terminalManagerFactory = (now) => (vfs, onNotify) => {
  const sessions = new Map()

  const open = (name, command) => {
    const termName = name.startsWith('trm_') ? name : `trm_${name}`
    if (sessions.has(termName)) {
      return { status: 'already_open', name: termName }
    }

    const trmDirRel = path.join('trm', termName)
    const trmDirAbs = vfs.resolvePath(trmDirRel)
    fs.mkdirSync(trmDirAbs, { recursive: true })

    const stdoutRel = path.join(trmDirRel, 'stdout.log')
    const screenRel = path.join(trmDirRel, 'screen.txt')
    const stdoutAbs = vfs.resolvePath(stdoutRel)
    const screenAbs = vfs.resolvePath(screenRel)

    // Initialize files
    fs.writeFileSync(stdoutAbs, '', 'utf-8')
    fs.writeFileSync(screenAbs, '', 'utf-8')

    const session = TerminalSession({ rows: 24, cols: 80 })

    session.setWatch((screenText) => {
      try {
        fs.writeFileSync(screenAbs, screenText, 'utf-8')
      } catch (_) {}
    })

    if (command) {
      session.write(`${command}\n`)
    }

    sessions.set(termName, {
      name: termName,
      session,
      stdoutRel,
      screenRel,
      stdoutAbs,
      screenAbs,
      startTime: now(),
    })

    return {
      name: termName,
      status: 'opened',
      stdout: stdoutRel,
      screen: screenRel,
    }
  }

  const close = (name) => {
    const termName = name.startsWith('trm_') ? name : `trm_${name}`
    const entry = sessions.get(termName)
    if (!entry) return { status: 'not_found', name: termName }

    try {
      entry.session.kill?.()
    } catch (_) {}
    sessions.delete(termName)
    return { status: 'closed', name: termName }
  }

  const closeAll = () => {
    for (const [name, entry] of sessions.entries()) {
      try {
        entry.session.kill?.()
      } catch (_) {}
    }
    sessions.clear()
  }

  const write = (name, text) => {
    const termName = name.startsWith('trm_') ? name : `trm_${name}`
    const entry = sessions.get(termName)
    if (!entry) return { status: 'not_found', name: termName }

    entry.session.write(text)
    return { status: 'written', name: termName, bytes: text.length }
  }

  const key = (name, keyName) => {
    const termName = name.startsWith('trm_') ? name : `trm_${name}`
    const entry = sessions.get(termName)
    if (!entry) return { status: 'not_found', name: termName }

    const normalized = (keyName || '').toLowerCase().trim()
    const mapped = KEY_MAP[normalized] || normalized
    entry.session.sendKey(mapped)
    return { status: 'key_sent', name: termName, key: keyName }
  }

  const getScreen = (name) => {
    const termName = name.startsWith('trm_') ? name : `trm_${name}`
    const entry = sessions.get(termName)
    if (!entry) return { error: `Terminal not found: ${name}` }
    return {
      name: termName,
      screen: entry.session.getScreenText?.() || '',
    }
  }

  const getActive = () => Array.from(sessions.values()).map((s) => ({
    name: s.name,
    stdout: s.stdoutRel,
    screen: s.screenRel,
    startTime: s.startTime,
  }))

  return {
    open,
    close,
    closeAll,
    write,
    key,
    getScreen,
    getActive,
  }
}

const TerminalManager = terminalManagerFactory(Date.now)

module.exports = {
  terminalManagerFactory,
  TerminalManager,
  KEY_MAP,
}
