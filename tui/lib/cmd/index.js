const { spawn } = require('child_process')
const fs = require('fs')
const path = require('path')

const FAST_PATH_TIMEOUT_MS = 250
const FAST_PATH_MAX_CHARS = 128

const parseDurationMs = (str) => {
  if (!str || typeof str !== 'string') return 60000
  const match = str.trim().match(/^(\d+(?:\.\d+)?)\s*(ms|s|m|h)?$/i)
  if (!match) return 60000
  const val = parseFloat(match[1])
  const unit = (match[2] || 's').toLowerCase()
  if (unit === 'ms') return Math.round(val)
  if (unit === 's') return Math.round(val * 1000)
  if (unit === 'm') return Math.round(val * 60 * 1000)
  if (unit === 'h') return Math.round(val * 3600 * 1000)
  return 60000
}

const commandRunnerFactory = (now) => (vfs, timers, onNotify) => {
  let cmdCounter = 100
  const activeProcesses = new Map()

  const run = async (command, remind = '1m') => {
    cmdCounter += 1
    const id = `cmd_${cmdCounter}`
    const logRelPath = path.join('tmp', `${id}.stdout.log`)
    const logAbsPath = vfs.resolvePath(logRelPath)

    // Ensure tmp folder exists
    fs.mkdirSync(path.dirname(logAbsPath), { recursive: true })
    const outFd = fs.openSync(logAbsPath, 'w')

    let isSettled = false
    let outputBuffer = ''
    let exitCode = null
    let errorObj = null

    const proc = spawn('/bin/sh', ['-c', command], {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    })

    const pid = proc.pid
    activeProcesses.set(id, {
      id,
      pid,
      command,
      proc,
      logRelPath,
      logAbsPath,
      startTime: now(),
      remind,
    })

    proc.stdout.on('data', (chunk) => {
      fs.writeSync(outFd, chunk)
      if (!isSettled && outputBuffer.length < FAST_PATH_MAX_CHARS * 2) {
        outputBuffer += chunk.toString('utf-8')
      }
    })

    proc.stderr.on('data', (chunk) => {
      fs.writeSync(outFd, chunk)
      if (!isSettled && outputBuffer.length < FAST_PATH_MAX_CHARS * 2) {
        outputBuffer += chunk.toString('utf-8')
      }
    })

    const exitPromise = new Promise((resolve) => {
      proc.on('close', (code) => {
        exitCode = code
        try {
          fs.closeSync(outFd)
        } catch (_) {}
        resolve({ finished: true, code })
      })

      proc.on('error', (err) => {
        errorObj = err
        try {
          fs.closeSync(outFd)
        } catch (_) {}
        resolve({ finished: true, error: err })
      })
    })

    const timeoutPromise = new Promise((resolve) => {
      setTimeout(() => resolve({ finished: false }), FAST_PATH_TIMEOUT_MS)
    })

    const raceResult = await Promise.race([exitPromise, timeoutPromise])

    if (raceResult.finished) {
      isSettled = true
      activeProcesses.delete(id)

      if (errorObj) {
        return { id, exit_code: -1, error: errorObj.message }
      }

      // Check Fast Path vs Spillover
      if (outputBuffer.length <= FAST_PATH_MAX_CHARS) {
        return {
          id,
          exit_code: exitCode,
          output: outputBuffer,
        }
      }

      const preview = outputBuffer.slice(0, FAST_PATH_MAX_CHARS).trimEnd()
      return {
        id,
        exit_code: exitCode,
        preview,
        log: logRelPath,
      }
    }

    // Process still running at 250ms -> Async detach
    isSettled = true

    // Schedule reminder if enabled
    let timerHandle = null
    if (remind !== false) {
      const remindMs = parseDurationMs(remind)
      const sec = Math.max(1, Math.round(remindMs / 1000))
      if (typeof timers?.setTimer === 'function') {
        const tRes = timers.setTimer(sec, `remind_${id}`, () => {
          const item = activeProcesses.get(id)
          if (!item) return
          onNotify?.({
            type: 'CMD_HEARTBEAT',
            cmdId: id,
            command,
            pid,
            elapsed: `${sec}s`,
            log: logRelPath,
            preview: `Command '${command}' is still running after ${remind}`,
          })
        })
        timerHandle = tRes.timerId
      }
    }

    // Handle eventual process exit
    proc.on('close', (code) => {
      activeProcesses.delete(id)
      if (timerHandle) timers?.cancelTimer?.(timerHandle)
      onNotify?.({
        type: 'CMD_COMPLETE',
        cmdId: id,
        command,
        pid,
        exit_code: code,
        log: logRelPath,
        preview: `Command '${command}' finished with code ${code}`,
      })
    })

    return {
      id,
      status: 'running',
      pid,
      log: logRelPath,
      remind,
    }
  }

  const kill = (id, signal = 'SIGTERM') => {
    const item = activeProcesses.get(id)
    if (!item) return { status: 'not_found', id }
    try {
      item.proc.kill(signal)
      activeProcesses.delete(id)
      return { status: 'killed', id, signal }
    } catch (err) {
      return { status: 'error', id, error: err.message }
    }
  }

  const killAll = () => {
    for (const [id, item] of activeProcesses.entries()) {
      try {
        item.proc.kill('SIGTERM')
      } catch (_) {}
    }
    activeProcesses.clear()
  }

  const getActive = () => Array.from(activeProcesses.values()).map((p) => ({
    id: p.id,
    pid: p.pid,
    command: p.command,
    startTime: p.startTime,
  }))

  return {
    run,
    kill,
    killAll,
    getActive,
  }
}

const CommandRunner = commandRunnerFactory(Date.now)

module.exports = {
  commandRunnerFactory,
  CommandRunner,
  FAST_PATH_TIMEOUT_MS,
  FAST_PATH_MAX_CHARS,
}
