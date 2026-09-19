const { describe, it } = require('node:test')
const assert = require('node:assert')
const { SyntaxTracker } = require('../lib/stream/syntax')

describe('SyntaxTracker & Elastic Syntactic Unit Gating', () => {
  it('starts at rest and tracks simple text boundaries', () => {
    const tracker = SyntaxTracker()
    assert.strictEqual(tracker.isAtRest(), true)

    tracker.ingestChunk('Hello world.')
    assert.strictEqual(tracker.isAtRest(), true)
    assert.strictEqual(tracker.isNaturalBoundary('Hello world.'), true)
  })

  it('prohibits yielding inside unclosed code fences', () => {
    const tracker = SyntaxTracker()
    tracker.ingestChunk('```python\ndef foo():\n  return 42\n')
    assert.strictEqual(tracker.isAtRest(), false)
    assert.strictEqual(tracker.getStackDepth().inCodeFence, true)
    assert.strictEqual(tracker.isNaturalBoundary('```python\ndef foo():\n  return 42\n'), false)

    tracker.ingestChunk('```\n')
    assert.strictEqual(tracker.isAtRest(), true)
    assert.strictEqual(tracker.getStackDepth().inCodeFence, false)
  })

  it('prohibits yielding inside unclosed double quotes and parentheses', () => {
    const tracker = SyntaxTracker()
    tracker.ingestChunk('He said, "This is an important point')
    assert.strictEqual(tracker.isAtRest(), false)
    assert.strictEqual(tracker.getStackDepth().inDoubleQuote, true)

    tracker.ingestChunk('" and continued.')
    assert.strictEqual(tracker.isAtRest(), true)
    assert.strictEqual(tracker.isNaturalBoundary('He said, "This is an important point" and continued.'), true)

    tracker.ingestChunk(' (nested detail')
    assert.strictEqual(tracker.isAtRest(), false)
    assert.strictEqual(tracker.getStackDepth().parenDepth, 1)

    tracker.ingestChunk(')')
    assert.strictEqual(tracker.isAtRest(), true)
  })
})
