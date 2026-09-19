const channelManagerFactory = () => () => {
  const channels = new Map()
  let focusedId = 'chat/user'

  const registerChannel = ({
    id,
    path = '',
    type = 'push',
    isFocused = false,
    cursor = 0,
  }) => {
    const existing = channels.get(id)
    const ch = {
      id,
      path: path || existing?.path || '',
      type,
      cursor: existing ? existing.cursor : cursor,
      isFocused: Boolean(isFocused),
      status: 'ACTIVE',
    }
    channels.set(id, ch)
    if (isFocused) setFocus(id)
    return { ...ch }
  }

  const getChannel = (id) => {
    const ch = channels.get(id)
    return ch ? { ...ch } : undefined
  }

  const setFocus = (id) => {
    if (!channels.has(id)) {
      registerChannel({ id, isFocused: true })
    }
    focusedId = id
    for (const [key, ch] of channels.entries()) {
      ch.isFocused = key === id
    }
    return getFocused()
  }

  const getFocused = () => {
    const ch = channels.get(focusedId)
    return ch ? { ...ch } : undefined
  }

  const updateCursor = (id, newCursor) => {
    const ch = channels.get(id)
    if (!ch) return undefined
    ch.cursor = Math.max(0, parseInt(newCursor, 10) || 0)
    return { ...ch }
  }

  const readChunk = (id, vfs, limit = 512) => {
    const ch = channels.get(id)
    if (!ch || !ch.path || !vfs) {
      return { error: `Channel ${id} not found or has no path`, eof: true }
    }
    const res = vfs.read(ch.path, ch.cursor, limit)
    if (res.error) return res
    ch.cursor += res.bytesRead
    return {
      id,
      path: ch.path,
      content: res.content,
      bytesRead: res.bytesRead,
      cursor: ch.cursor,
      eof: res.eof,
    }
  }

  const listChannels = () =>
    Array.from(channels.values()).map((ch) => ({ ...ch }))

  const serialize = () => ({
    focusedId,
    channels: listChannels(),
  })

  // Initialize default chat channel
  registerChannel({ id: 'chat/user', type: 'push', isFocused: true })

  return {
    registerChannel,
    getChannel,
    setFocus,
    getFocused,
    updateCursor,
    readChunk,
    listChannels,
    serialize,
  }
}

module.exports = {
  channelManagerFactory,
  ChannelManager: channelManagerFactory(),
}
