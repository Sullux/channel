const fs = require('fs')
const path = require('path')

const MAX_READ_LIMIT = 512 // strictly capped at 512 characters (~128 tokens)
const DEFAULT_PREVIEW_LIMIT = 128 // ~32 tokens

const vfsFactory = () => (rootDir) => {
  const root = path.resolve(rootDir || './.agent')
  let msgCounter = 1000

  const scanHighestMsgId = () => {
    try {
      const userDir = path.join(root, 'msg', 'user')
      if (fs.existsSync(userDir)) {
        const files = fs.readdirSync(userDir)
        for (const file of files) {
          const m = file.match(/^(\d+)\.txt$/)
          if (m) {
            const seq = parseInt(m[1], 10)
            if (seq > msgCounter) {
              msgCounter = seq
            }
          }
        }
      }
    } catch (_) {}
  }

  const init = () => {
    const dirs = [
      path.join(root, 'msg', 'user'),
      path.join(root, 'msg', 'assistant'),
      path.join(root, 'trm'),
      path.join(root, 'tmp'),
      path.join(root, 'notify', 'pending'),
      path.join(root, 'notify', 'snoozed'),
      path.join(root, 'mem', 'episodes'),
    ]
    for (const d of dirs) {
      fs.mkdirSync(d, { recursive: true })
    }
    scanHighestMsgId()
    return root
  }

  const resolvePath = (relPath) => {
    if (!relPath || typeof relPath !== 'string') {
      throw new Error('Invalid path')
    }
    const abs = path.isAbsolute(relPath) ? relPath : path.resolve(root, relPath)
    // Security / jail boundary check
    const normAbs = path.normalize(abs)
    const normRoot = path.normalize(root)
    if (!normAbs.startsWith(normRoot)) {
      throw new Error(`Access denied: path outside filesystemRoot (${normAbs})`)
    }
    return normAbs
  }

  const saveUserMessage = (text) => {
    init()
    msgCounter += 1
    const id = `${msgCounter}`
    const fileName = `${id}.txt`
    const filePath = path.join(root, 'msg', 'user', fileName)

    const payload = typeof text === 'string' ? text : String(text || '')

    // Remove existing file if present (even if 0444 read-only) before writing fresh
    if (fs.existsSync(filePath)) {
      try {
        fs.unlinkSync(filePath)
      } catch (_) {}
    }

    fs.writeFileSync(filePath, payload, 'utf-8')
    try {
      // Mark read-only (chmod 0444)
      fs.chmodSync(filePath, 0o444)
    } catch (_) {}

    // Estimate token count (rough rule of thumb: 4 chars/token)
    const tokenCount = Math.max(1, Math.round(payload.length / 4))
    const isTruncated = payload.length > DEFAULT_PREVIEW_LIMIT
    const preview = isTruncated
      ? payload.slice(0, DEFAULT_PREVIEW_LIMIT).trimEnd()
      : payload

    return {
      id,
      path: filePath,
      relPath: path.relative(root, filePath),
      payload,
      tokenCount,
      isTruncated,
      preview,
    }
  }

  const read = (relPath, offset = 0, limit = MAX_READ_LIMIT) => {
    const absPath = resolvePath(relPath)
    if (!fs.existsSync(absPath)) {
      return { error: `File not found: ${relPath}` }
    }

    const stat = fs.statSync(absPath)
    if (stat.isDirectory()) {
      return { error: `Cannot read directory: ${relPath}` }
    }

    const start = Math.max(0, parseInt(offset, 10) || 0)
    const maxLen = Math.min(MAX_READ_LIMIT, Math.max(0, parseInt(limit, 10) || MAX_READ_LIMIT))

    const fd = fs.openSync(absPath, 'r')
    try {
      const buffer = Buffer.alloc(maxLen)
      const bytesRead = fs.readSync(fd, buffer, 0, maxLen, start)
      const content = buffer.toString('utf-8', 0, bytesRead)
      return {
        path: relPath,
        offset: start,
        bytesRead,
        content,
        eof: start + bytesRead >= stat.size,
      }
    } finally {
      fs.closeSync(fd)
    }
  }

  const list = (relPath = '.') => {
    const absPath = resolvePath(relPath)
    if (!fs.existsSync(absPath)) {
      return { error: `Path not found: ${relPath}` }
    }
    const stat = fs.statSync(absPath)
    if (!stat.isDirectory()) {
      return { error: `Not a directory: ${relPath}` }
    }
    const entries = fs.readdirSync(absPath, { withFileTypes: true })
    return {
      path: relPath,
      entries: entries.map((e) => ({
        name: e.name,
        isDirectory: e.isDirectory(),
        isFile: e.isFile(),
      })),
    }
  }

  const remove = (relPath) => {
    const absPath = resolvePath(relPath)
    if (!fs.existsSync(absPath)) return false
    fs.rmSync(absPath, { recursive: true, force: true })
    return true
  }

  return {
    root,
    init,
    resolvePath,
    saveUserMessage,
    read,
    list,
    remove,
  }
}

const Vfs = vfsFactory()

module.exports = {
  vfsFactory,
  Vfs,
  MAX_READ_LIMIT,
  DEFAULT_PREVIEW_LIMIT,
}
