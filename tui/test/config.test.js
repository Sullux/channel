const { describe, it } = require('node:test')
const assert = require('node:assert')
const path = require('path')
const fs = require('fs')
const os = require('os')
const { loadConfig, loadSystemPrompt, parseCliArgs } = require('../index')

describe('Config & Path Resolution', () => {
  it('parses CLI arguments for --config and -c', () => {
    assert.strictEqual(parseCliArgs(['-c', 'foo.json']).configPath, 'foo.json')
    assert.strictEqual(parseCliArgs(['--config', 'bar.json']).configPath, 'bar.json')
    assert.strictEqual(parseCliArgs(['--config=baz.json']).configPath, 'baz.json')
  })

  it('pre-resolves relative config paths using the config file directory as base', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tui-config-test-'))
    const cfgFile = path.join(tmpDir, 'custom-cfg.json')
    const customPrompt = path.join(tmpDir, 'CUSTOM_PROMPT.md')
    fs.writeFileSync(customPrompt, 'Custom Prompt Content')
    fs.writeFileSync(
      cfgFile,
      JSON.stringify({
        modelPath: './models/gemma',
        memoryDir: './memory_store',
        promptPath: './CUSTOM_PROMPT.md',
      }),
    )

    const cfg = loadConfig(cfgFile)
    assert.strictEqual(cfg.modelPath, path.join(tmpDir, 'models/gemma'))
    assert.strictEqual(cfg.memoryDir, path.join(tmpDir, 'memory_store'))
    assert.strictEqual(cfg.promptPath, customPrompt)

    const fullPrompt = loadSystemPrompt(cfg.promptPath)
    assert.ok(fullPrompt.includes('Custom Prompt Content'))

    fs.rmSync(tmpDir, { recursive: true, force: true })
  })

  it('loads default config when explicit path not provided', () => {
    const cfg = loadConfig()
    assert.ok(path.isAbsolute(cfg.modelPath))
    assert.ok(path.isAbsolute(cfg.memoryDir))
    assert.ok(path.isAbsolute(cfg.promptPath))
  })

  it('discovers config.json in CWD before falling back to __dirname', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tui-cwd-config-'))
    const cwdCfg = path.join(tmpDir, 'config.json')
    fs.writeFileSync(
      cwdCfg,
      JSON.stringify({
        memoryDir: './cwd_memory',
      }),
    )

    const origCwd = process.cwd()
    try {
      process.chdir(tmpDir)
      const cfg = loadConfig()
      assert.strictEqual(cfg.memoryDir, path.join(tmpDir, 'cwd_memory'))
    } finally {
      process.chdir(origCwd)
      fs.rmSync(tmpDir, { recursive: true, force: true })
    }
  })
})
