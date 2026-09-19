const { describe, it } = require('node:test')
const assert = require('node:assert')
const { MarkdownControl, mapBlocks } = require('../lib/ui/markdown')
const { parse } = require('@sullux/markdown-compiler')

describe('MarkdownControl', () => {
  it('maps headings, paragraphs, lists, and code blocks into styled spans', () => {
    const md = [
      '# Heading 1',
      '## Heading 2',
      'Regular paragraph with **bold**, *italic*, and `code`.',
      '- Bullet A',
      '- Bullet B',
      '1. Step 1',
      '2. Step 2',
      '```js',
      'const a = 10',
      '```',
      '> Quoted insight',
    ].join('\n\n')

    const ast = parse(md)
    const spans = mapBlocks(ast.blocks, {
      fg: '#ffffff',
      h1: { bold: true, underline: true },
      code: { bg: '#1f2335' },
    })

    assert.ok(spans.length > 0)
    const boldSpan = spans.find((s) => s.text === 'bold')
    assert.ok(boldSpan)
    assert.strictEqual(boldSpan.bold, true)

    const italicSpan = spans.find((s) => s.text === 'italic')
    assert.ok(italicSpan)
    assert.strictEqual(italicSpan.italic, true)

    const codeSpan = spans.find((s) => s.text === ' code ')
    assert.ok(codeSpan)
    assert.strictEqual(codeSpan.bg, '#1f2335')

    const h1Span = spans.find((s) => s.text === 'Heading 1')
    assert.ok(h1Span)
    assert.strictEqual(h1Span.bold, true)
    assert.strictEqual(h1Span.underline, true)
  })

  it('measures, layouts, and renders through MarkdownControl contract', () => {
    const node = {
      type: 'markdown',
      text: '# Test Header\n\nHere is a list:\n- Item 1\n- Item 2',
      fg: '#ffffff',
    }
    const bounds = { x: 0, y: 0, width: 80, height: 20 }
    const measured = MarkdownControl.onMeasure(node, bounds)
    assert.ok(measured.width > 0)
    assert.ok(measured.height > 0)

    MarkdownControl.onLayout(node, bounds)
    const grid = Array.from({ length: 20 }, () =>
      Array.from({ length: 80 }, () => ({ char: ' ', style: '' })),
    )
    MarkdownControl.onRender(node, grid)

    const firstLine = grid[0].map((c) => c.char).join('')
    assert.ok(firstLine.includes('Test Header'))
  })

  it('indents nested sub-lists at 2, 4, and 6 spaces without blank lines between items', () => {
    const md = [
      '* Item 1',
      '  * Sub-item 1.1',
      '    * Sub-sub-item 1.1.1',
      '* Item 2',
    ].join('\n')

    const ast = parse(md)
    const spans = mapBlocks(ast.blocks, {})
    const bulletTexts = spans.map((s) => s.text).filter((t) => t !== '\n')
    assert.strictEqual(bulletTexts[0], '  • ')
    assert.strictEqual(bulletTexts[1], 'Item 1')
    assert.strictEqual(bulletTexts[2], '    • ')
    assert.strictEqual(bulletTexts[3], 'Sub-item 1.1')
    assert.strictEqual(bulletTexts[4], '      • ')
    assert.strictEqual(bulletTexts[5], 'Sub-sub-item 1.1.1')
    assert.strictEqual(bulletTexts[6], '  • ')
    assert.strictEqual(bulletTexts[7], 'Item 2')

    const node = { type: 'markdown', text: md }
    const size = MarkdownControl.onMeasure(node, { maxWidth: 80, maxHeight: 100 })
    assert.strictEqual(size.height, 4)
  })

  it('preserves single blank line between paragraphs and headings', () => {
    const md = '# Title\n\nParagraph 1\n\nParagraph 2'
    const node = { type: 'markdown', text: md }
    const size = MarkdownControl.onMeasure(node, { maxWidth: 80, maxHeight: 100 })
    assert.strictEqual(size.height, 5)
  })
})
