const { spawn } = require('child_process')

const copyToClipboard = (text, outStream = process.stdout) => {
  if (!text) return
  try {
    const b64 = Buffer.from(text, 'utf-8').toString('base64')
    outStream?.write?.(`\x1b]52;c;${b64}\x07`)
  } catch (_) {}

  try {
    if (process.platform === 'darwin') {
      const p = spawn('pbcopy', [], { stdio: ['pipe', 'ignore', 'ignore'] })
      p.stdin.end(text)
    } else if (process.platform === 'win32') {
      const p = spawn('clip.exe', [], { stdio: ['pipe', 'ignore', 'ignore'] })
      p.stdin.end(text)
    } else {
      const wl = spawn('wl-copy', [], { stdio: ['pipe', 'ignore', 'ignore'] })
      wl.on('error', () => {
        try {
          const xc = spawn('xclip', ['-selection', 'clipboard'], {
            stdio: ['pipe', 'ignore', 'ignore'],
          })
          xc.on('error', () => {})
          xc.stdin.end(text)
        } catch (_) {}
      })
      wl.stdin.end(text)
    }
  } catch (_) {}
}

const getSelectedText = (state) => {
  if (!state) return ''
  const { mode, selectedIdx, conversation, stream, plan, activeThought, activeResponse } = state
  if (mode === 'chat') {
    const idx = selectedIdx?.chat || 0
    if (idx < (conversation?.length || 0)) return conversation[idx]?.text || ''
    return activeResponse || ''
  }
  if (mode === 'stream') {
    const idx = selectedIdx?.stream || 0
    if (idx < (stream?.length || 0)) return stream[idx]?.content || ''
    if (activeThought) return activeThought
    return activeResponse || ''
  }
  if (mode === 'plan') {
    const idx = selectedIdx?.plan || 0
    if (idx < (plan?.length || 0)) {
      const item = plan[idx]
      return item?.summary || item?.description || item?.brief || ''
    }
  }
  return ''
}

module.exports = {
  copyToClipboard,
  getSelectedText,
}
