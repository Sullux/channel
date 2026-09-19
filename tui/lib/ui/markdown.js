const { parse } = require('@sullux/markdown-compiler')
const rich = require('@sullux/tui/lib/controls/rich')

const mapInlines = (children = [], style = {}, node = {}) =>
  children.flatMap((c) => {
    if (c.type === 'text') return [{ ...style, text: c.value }]
    if (c.type === 'bold') return mapInlines(c.children, { ...style, bold: true, ...(node.bold || {}) }, node)
    if (c.type === 'italic') return mapInlines(c.children, { ...style, italic: true, ...(node.italic || {}) }, node)
    if (c.type === 'strikethrough') return mapInlines(c.children, { ...style, strikethrough: true }, node)
    if (c.type === 'code') return [{ ...style, ...(node.code || {}), text: ` ${c.value} ` }]
    if (c.type === 'link') return mapInlines(c.children, { ...style, underline: true, ...(node.link || {}) }, node)
    return c.type === 'br' ? [{ text: '\n' }] : []
  })

const mapBlocks = (blocks = [], node = {}) => {
  const base = { fg: node.fg, bg: node.bg }
  return blocks.flatMap((b, idx) => {
    const isLast = idx === blocks.length - 1
    const endBreak = isLast ? [{ text: '\n' }] : [{ text: '\n\n' }]

    if (b.type === 'header') {
      const hStyle = b.level === 1 ? (node.h1 || { bold: true, underline: true })
        : b.level === 2 ? (node.h2 || { bold: true })
        : (node.h3 || { bold: true, italic: true })
      return [...mapInlines(b.children, { ...base, ...hStyle }, node), ...endBreak]
    }
    if (b.type === 'paragraph') {
      return [...mapInlines(b.children, base, node), ...endBreak]
    }
    if (b.type === 'bulletList' || b.type === 'orderedList') {
      const isOrd = b.type === 'orderedList'
      const bulletStyle = { ...base, ...(node.bullet || { bold: true }) }
      const items = (b.items || []).flatMap((kids, i) => {
        const indent = ' '.repeat(2 + (kids.depth || 0) * 2)
        const isItemOrd = kids.listType ? kids.listType === 'ordered' : isOrd
        const itemNum = kids.order != null ? kids.order : (i + 1)
        const prefix = isItemOrd ? `${indent}${itemNum}. ` : `${indent}• `
        return [
          { ...bulletStyle, text: prefix },
          ...mapInlines(kids, base, node),
          { text: '\n' },
        ]
      })
      return [...items, ...(isLast ? [] : [{ text: '\n' }])]
    }
    if (b.type === 'codeBlock') {
      const langText = b.language ? `\`\`\` ${b.language}\n` : '```\n'
      const codeStyle = { ...base, ...(node.codeBlock || {}) }
      return [
        { ...base, italic: true, text: langText },
        { ...codeStyle, text: `${b.value || ''}\n` },
        { ...base, italic: true, text: `\`\`\`\n` },
        ...(isLast ? [] : [{ text: '\n' }]),
      ]
    }
    if (b.type === 'blockquote') {
      const qStyle = { ...base, ...(node.blockquote || { italic: true }) }
      const inner = (b.children || []).flatMap((c) =>
        c.type === 'paragraph' ? mapInlines(c.children, qStyle, node) : [],
      )
      return [{ ...base, text: '│ ' }, ...inner, { text: '\n' }, ...(isLast ? [] : [{ text: '\n' }])]
    }
    return b.type === 'hr' ? [{ ...base, text: '───\n' }, ...(isLast ? [] : [{ text: '\n' }])] : []
  })
}

const syncMarkdownNode = (node) => {
  if (node._rawMarkdown !== node.text) {
    const raw = String(node.text || '')
    node.inner = mapBlocks(parse(raw).blocks, node)
    node._rawMarkdown = node.text
    delete node._flowItems
  }
  return node
}

const MarkdownControl = {
  onMeasure: (node, constraints) => rich.onMeasure(syncMarkdownNode(node), constraints),
  onLayout: (node, bounds) => rich.onLayout(syncMarkdownNode(node), bounds),
  onRender: (node, grid) => rich.onRender(syncMarkdownNode(node), grid),
}

module.exports = {
  MarkdownControl,
  mapBlocks,
  mapInlines,
}
