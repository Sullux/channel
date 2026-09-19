const fs = require('fs')
const path = require('path')

const STREAM_FILENAME = '.stream.jsonl'
const DEFAULT_CHUNK_SIZE = 256 * 1024 // 256 KB

const parseJsonLines = (text) => {
  if (!text) return []
  const items = []
  const lines = text.split('\n')
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue
    try {
      items.push(JSON.parse(trimmed))
    } catch (_) {}
  }
  return items
}

const streamLogFactory = (fsMod, pathMod) => (memoryDir) => {
  if (!memoryDir) {
    return {
      enabled: false,
      append: () => {},
      loadAll: () => [],
      loadTail: () => ({ items: [], startOffset: 0, endOffset: 0 }),
      loadPrevious: () => ({ items: [], startOffset: 0 }),
      close: () => {},
    }
  }

  const resolvedDir = pathMod.resolve(process.cwd(), memoryDir)
  fsMod.mkdirSync(resolvedDir, { recursive: true })
  const filePath = pathMod.join(resolvedDir, STREAM_FILENAME)

  let writeStream = null
  const getWriteStream = () => {
    if (!writeStream) {
      writeStream = fsMod.createWriteStream(filePath, { flags: 'a' })
    }
    return writeStream
  }

  const append = (item) => {
    if (!item) return
    const row = JSON.stringify(item) + '\n'
    getWriteStream().write(row)
  }

  const loadAll = () => {
    if (!fsMod.existsSync(filePath)) return []
    try {
      const content = fsMod.readFileSync(filePath, 'utf-8')
      return parseJsonLines(content)
    } catch (_) {
      return []
    }
  }

  const loadTail = (chunkSize = DEFAULT_CHUNK_SIZE) => {
    if (!fsMod.existsSync(filePath)) {
      return { items: [], startOffset: 0, endOffset: 0 }
    }
    const stat = fsMod.statSync(filePath)
    if (stat.size === 0) {
      return { items: [], startOffset: 0, endOffset: 0 }
    }
    if (stat.size <= chunkSize) {
      const content = fsMod.readFileSync(filePath, 'utf-8')
      return { items: parseJsonLines(content), startOffset: 0, endOffset: stat.size }
    }

    const fd = fsMod.openSync(filePath, 'r')
    try {
      const readOffset = stat.size - chunkSize
      const buffer = Buffer.alloc(chunkSize)
      fsMod.readSync(fd, buffer, 0, chunkSize, readOffset)
      const text = buffer.toString('utf-8')
      const firstNewline = text.indexOf('\n')
      if (firstNewline === -1) {
        return { items: [], startOffset: stat.size, endOffset: stat.size }
      }
      const validText = text.slice(firstNewline + 1)
      const startOffset = readOffset + firstNewline + 1
      return { items: parseJsonLines(validText), startOffset, endOffset: stat.size }
    } finally {
      fsMod.closeSync(fd)
    }
  }

  const loadPrevious = (currentStartOffset, chunkSize = DEFAULT_CHUNK_SIZE) => {
    if (!fsMod.existsSync(filePath) || currentStartOffset <= 0) {
      return { items: [], startOffset: 0 }
    }
    const readOffset = Math.max(0, currentStartOffset - chunkSize)
    const readLength = currentStartOffset - readOffset
    const fd = fsMod.openSync(filePath, 'r')
    try {
      const buffer = Buffer.alloc(readLength)
      fsMod.readSync(fd, buffer, 0, readLength, readOffset)
      const text = buffer.toString('utf-8')
      let alignOffset = 0
      let validText = text
      if (readOffset > 0) {
        const firstNewline = text.indexOf('\n')
        if (firstNewline === -1) return { items: [], startOffset: currentStartOffset }
        alignOffset = firstNewline + 1
        validText = text.slice(alignOffset)
      }
      return { items: parseJsonLines(validText), startOffset: readOffset + alignOffset }
    } finally {
      fsMod.closeSync(fd)
    }
  }

  const close = () => {
    if (writeStream) {
      writeStream.end()
      writeStream = null
    }
  }

  return {
    enabled: true,
    filePath,
    append,
    loadAll,
    loadTail,
    loadPrevious,
    close,
  }
}

module.exports = {
  streamLogFactory,
  StreamLog: streamLogFactory(fs, path),
}
