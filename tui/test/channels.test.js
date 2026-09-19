const test = require('node:test')
const assert = require('node:assert')
const fs = require('node:fs')
const path = require('node:path')
const { ChannelManager } = require('../lib/channels')
const { Vfs } = require('../lib/vfs')

test('ChannelManager registers default chat channel and manages focus', () => {
  const cm = ChannelManager()
  assert.strictEqual(cm.getFocused()?.id, 'chat/user')
  assert.strictEqual(cm.getFocused()?.isFocused, true)

  cm.registerChannel({ id: 'trm/build', path: 'trm/build/stdout.log', type: 'hybrid' })
  assert.strictEqual(cm.getFocused()?.id, 'chat/user')

  cm.setFocus('trm/build')
  assert.strictEqual(cm.getFocused()?.id, 'trm/build')
  assert.strictEqual(cm.getChannel('chat/user')?.isFocused, false)
})

test('ChannelManager reads chunks sequentially advancing cursor', () => {
  const tmpDir = path.resolve(__dirname, '../.test_channels_vfs')
  fs.rmSync(tmpDir, { recursive: true, force: true })
  const vfs = Vfs(tmpDir)

  const saved = vfs.saveUserMessage('1234567890abcdefghij') // 20 chars
  const cm = ChannelManager()
  cm.registerChannel({ id: 'msg/1', path: saved.relPath, type: 'pull' })

  // Read first 10 chars
  const chunk1 = cm.readChunk('msg/1', vfs, 10)
  assert.strictEqual(chunk1.content, '1234567890')
  assert.strictEqual(chunk1.cursor, 10)
  assert.strictEqual(chunk1.eof, false)

  // Read next 10 chars
  const chunk2 = cm.readChunk('msg/1', vfs, 10)
  assert.strictEqual(chunk2.content, 'abcdefghij')
  assert.strictEqual(chunk2.cursor, 20)
  assert.strictEqual(chunk2.eof, true)

  // Further read hits EOF with 0 bytes
  const chunk3 = cm.readChunk('msg/1', vfs, 10)
  assert.strictEqual(chunk3.content, '')
  assert.strictEqual(chunk3.bytesRead, 0)
  assert.strictEqual(chunk3.eof, true)

  fs.rmSync(tmpDir, { recursive: true, force: true })
})
