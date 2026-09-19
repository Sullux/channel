#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');
const constants = require('../tui/lib/protocol/constants');
const framing = require('../tui/lib/protocol/framing');

const modelPath = '../gemma-4-12B-it-qat-q4_0-unquantized';
const binPath = path.resolve(__dirname, '../zig-out/bin/infer');
const child = spawn(binPath, ['--model', modelPath, '--serve', '--gpu', '--q4'], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let rxBuffer = Buffer.alloc(0);
let currentTurn = 1;
let turnStart = 0;
let isReady = false;

child.stdout.on('data', (chunk) => {
  rxBuffer = Buffer.concat([rxBuffer, chunk]);
  while (true) {
    const frame = framing.parsedFrame(rxBuffer);
    if (!frame) break;
    rxBuffer = rxBuffer.subarray(16 + frame.header.payloadLen);

    if (frame.header.opcode === constants.OP_STATUS) {
      const status = frame.payload.readUInt8(0);
      if (status === 0 && !isReady) {
        isReady = true;
        sendTurn1();
      }
    } else if (frame.header.opcode === constants.OP_STREAM_THOUGHT) {
      const text = frame.payload.subarray(24).toString('utf-8').replace(/\u2581/g, ' ');
      process.stdout.write(`\x1b[33m${text}\x1b[0m`);
    } else if (frame.header.opcode === constants.OP_STREAM_CONTENT) {
      const text = frame.payload.subarray(24).toString('utf-8').replace(/\u2581/g, ' ');
      process.stdout.write(`\x1b[32m${text}\x1b[0m`);
    } else if (frame.header.opcode === constants.OP_TURN_COMPLETE) {
      const totalTok = frame.payload.readUInt32LE(0);
      const elapsedMs = frame.payload.readUInt32LE(4);
      const tokSec = frame.payload.readFloatLE(8);
      console.log(`\n\n\x1b[36m[Turn ${currentTurn} Complete: ${totalTok} tokens in ${elapsedMs} ms (${tokSec.toFixed(2)} tok/s)]\x1b[0m\n`);
      if (currentTurn === 1) {
        currentTurn = 2;
        setTimeout(sendTurn2, 100);
      } else {
        const shutdownHdr = framing.headerBuffer(constants.OP_SHUTDOWN, 999, 0);
        child.stdin.write(shutdownHdr);
      }
    }
  }
});

function sendTurn1() {
  console.log('\x1b[34m--- SENDING TURN 1: "How are you doing today?" ---\x1b[0m');
  turnStart = Date.now();
  child.stdin.write(framing.streamInputFrame("How are you doing today?", 1));
}

function sendTurn2() {
  console.log('\x1b[34m--- SENDING TURN 2: "What tools do you see that you have available?" ---\x1b[0m');
  turnStart = Date.now();
  child.stdin.write(framing.streamInputFrame("What tools do you see that you have available?", 2));
}
