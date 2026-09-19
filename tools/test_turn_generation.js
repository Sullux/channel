#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const constants = require('../tui/lib/protocol/constants');
const framing = require('../tui/lib/protocol/framing');

const modelPath = process.argv[2] || '../gemma-4-12B-it';
const extraArgs = process.argv.slice(3);
if (extraArgs.length === 0) {
  extraArgs.push('--gpu', '--mixed');
}

const binPath = path.resolve(__dirname, '../zig-out/bin/infer');
const child = spawn(binPath, ['--model', modelPath, '--serve', ...extraArgs], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let rxBuffer = Buffer.alloc(0);
child.stdout.on('data', (chunk) => {
  rxBuffer = Buffer.concat([rxBuffer, chunk]);
  while (true) {
    const frame = framing.parsedFrame(rxBuffer);
    if (!frame) break;
    rxBuffer = rxBuffer.subarray(16 + frame.header.payloadLen);

    if (frame.header.opcode === constants.OP_STREAM_THOUGHT) {
      const text = frame.payload.subarray(24).toString('utf-8');
      process.stdout.write(`\x1b[33m${text}\x1b[0m`);
    } else if (frame.header.opcode === constants.OP_STREAM_CONTENT) {
      const text = frame.payload.subarray(24).toString('utf-8');
      process.stdout.write(`\x1b[32m${text}\x1b[0m`);
    } else if (frame.header.opcode === constants.OP_TURN_COMPLETE) {
      console.log('\n\n\x1b[36m[Turn Complete]\x1b[0m');
      const shutdownHdr = framing.headerBuffer(constants.OP_SHUTDOWN, 999, 0);
      child.stdin.write(shutdownHdr);
    }
  }
});

function makeConfigFrame(maxTokens = 128, temp = 0.7) {
  const payload = Buffer.alloc(20);
  payload.writeUInt32LE(512, 0);
  payload.writeFloatLE(temp, 4);
  payload.writeFloatLE(0.95, 8);
  payload.writeFloatLE(0.0, 12);
  payload.writeUInt32LE(maxTokens, 16);
  const hdr = framing.headerBuffer(constants.OP_SET_CONFIG, 1, payload.length);
  return Buffer.concat([hdr, payload]);
}

const kernelPath = path.resolve(__dirname, '../tui/PROMPT_KERNEL.md');
const kernel = fs.existsSync(kernelPath) ? fs.readFileSync(kernelPath, 'utf-8').trim() : '';
const prompt = `<|turn>system\n<|think|>\n${kernel}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n<|channel>thought\n`;

child.stdin.write(makeConfigFrame(128, 0.7));
child.stdin.write(framing.streamInputFrame(prompt, 1));
