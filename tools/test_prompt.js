import { spawn } from 'child_process';
import fs from 'fs';

const MAGIC = 0x53554C58;
const VERSION = 1;
const OP_STREAM_INPUT = 0x0001;
const OP_SET_CONFIG = 0x0004;
const OP_SHUTDOWN = 0x000F;
const OP_STREAM_CONTENT = 0x0101;
const OP_STREAM_THOUGHT = 0x0102;
const OP_TURN_COMPLETE = 0x0103;

function makeHeader(opcode, msgId, payloadLen) {
  const buf = Buffer.alloc(16);
  buf.writeUInt32LE(MAGIC, 0);
  buf.writeUInt16LE(VERSION, 4);
  buf.writeUInt16LE(msgId, 6);
  buf.writeUInt16LE(opcode, 8);
  buf.writeUInt16LE(0, 10);
  buf.writeUInt32LE(payloadLen, 12);
  return buf;
}

function makeStreamInputFrame(text, msgId = 1) {
  const textBytes = Buffer.from(text, 'utf-8');
  const payload = Buffer.alloc(8 + textBytes.length);
  payload.writeUInt8(0x00, 0);
  textBytes.copy(payload, 8);
  const hdr = makeHeader(OP_STREAM_INPUT, msgId, payload.length);
  return Buffer.concat([hdr, payload]);
}

function makeConfigFrame(maxTokens = 128) {
  const payload = Buffer.alloc(20);
  payload.writeUInt32LE(512, 0);
  payload.writeFloatLE(0.7, 4);
  payload.writeFloatLE(0.95, 8);
  payload.writeFloatLE(0.001, 12);
  payload.writeUInt32LE(maxTokens, 16);
  const hdr = makeHeader(OP_SET_CONFIG, 1, payload.length);
  return Buffer.concat([hdr, payload]);
}

async function testPrompt() {
  const proc = spawn('./zig-out/bin/infer', ['--model', '../gemma-4-12B-it', '--serve', '--gpu', '--q4', '--quiescence'], {
    stdio: ['pipe', 'pipe', 'inherit'],
  });

  let rx = Buffer.alloc(0);
  let thoughts = '';
  let content = '';
  let complete = false;

  proc.stdout.on('data', (chunk) => {
    rx = Buffer.concat([rx, chunk]);
    while (rx.length >= 16) {
      const magic = rx.readUInt32LE(0);
      if (magic !== MAGIC) break;
      const opcode = rx.readUInt16LE(8);
      const payloadLen = rx.readUInt32LE(12);
      if (rx.length < 16 + payloadLen) break;

      const payload = rx.subarray(16, 16 + payloadLen);
      rx = rx.subarray(16 + payloadLen);

      if (opcode === OP_STREAM_THOUGHT) {
        const t = payload.subarray(24).toString('utf-8');
        thoughts += t;
      } else if (opcode === OP_STREAM_CONTENT) {
        const t = payload.subarray(24).toString('utf-8');
        content += t;
      } else if (opcode === OP_TURN_COMPLETE) {
        const tokSec = payload.readFloatLE(8);
        console.log(`\nTurn complete: ${tokSec.toFixed(1)} tok/s`);
        complete = true;
      }
    }
  });

  proc.stdin.write(makeConfigFrame(128));
  const kernel = fs.readFileSync('tui/PROMPT_KERNEL.md', 'utf-8').trim();
  const prompt = `<|turn>system\n${kernel}\n<turn|>\n<|turn>user\nHow are you doing today?\n<turn|>\n<|turn>model\n`;
  proc.stdin.write(makeStreamInputFrame(prompt, 1));

  const start = Date.now();
  while (!complete && Date.now() - start < 30000) {
    await new Promise((r) => setTimeout(r, 100));
  }

  proc.stdin.write(makeHeader(OP_SHUTDOWN, 0, 0));
  await new Promise((r) => proc.on('close', r));

  console.log('THOUGHTS:', JSON.stringify(thoughts));
  console.log('CONTENT:', JSON.stringify(content));
}

testPrompt();
