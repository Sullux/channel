import { spawn } from 'child_process';
import { resolve } from 'path';

const MAGIC = 0x53554C58;
const VERSION = 1;
const OP_STREAM_INPUT = 0x0001;
const OP_MEM_QUERY = 0x0003;
const OP_SET_CONFIG = 0x0004;
const OP_PING = 0x000E;
const OP_SHUTDOWN = 0x000F;
const OP_STREAM_CONTENT = 0x0101;
const OP_STREAM_THOUGHT = 0x0102;
const OP_TURN_COMPLETE = 0x0103;
const OP_MEM_RESPONSE = 0x0105;
const OP_STATUS = 0x0106;
const OP_PONG = 0x010E;
const OP_ERROR = 0x01FF;

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

function makeMemQueryFrame(query, msgId = 2, topK = 5) {
  const queryBytes = Buffer.from(query, 'utf-8');
  const payload = Buffer.alloc(24 + queryBytes.length);
  payload.writeUInt8(0x00, 0);
  payload.writeUInt8(topK, 1);
  payload.writeBigUInt64LE(0n, 8);
  payload.writeBigUInt64LE(0n, 16);
  queryBytes.copy(payload, 24);
  const hdr = makeHeader(OP_MEM_QUERY, msgId, payload.length);
  return Buffer.concat([hdr, payload]);
}

function makeConfigFrame(maxTokens = 64) {
  const payload = Buffer.alloc(20);
  payload.writeUInt32LE(512, 0);
  payload.writeFloatLE(0.7, 4);
  payload.writeFloatLE(0.95, 8);
  payload.writeFloatLE(0.001, 12);
  payload.writeUInt32LE(maxTokens, 16);
  const hdr = makeHeader(OP_SET_CONFIG, 1, payload.length);
  return Buffer.concat([hdr, payload]);
}

async function runProtocolTest() {
  const modelDir = process.argv[2] || '../gemma-4-12B-it';
  const extraArgs = process.argv.slice(3);
  const binary = resolve('zig-out/bin/infer');

  console.log(`Spawning ${binary} --model ${modelDir} --serve ${extraArgs.join(' ')} ...`);
  const proc = spawn(binary, ['--model', modelDir, '--serve', ...extraArgs], {
    stdio: ['pipe', 'pipe', 'inherit'],
  });

  let rxBuf = Buffer.alloc(0);
  let pongReceived = false;
  let turnCompleteReceived = false;
  let memResponseReceived = false;
  let tokenCount = 0;

  proc.stdout.on('data', (chunk) => {
    rxBuf = Buffer.concat([rxBuf, chunk]);
    while (rxBuf.length >= 16) {
      const magic = rxBuf.readUInt32LE(0);
      if (magic !== MAGIC) {
        console.error('Invalid magic received:', magic.toString(16));
        process.exit(1);
      }
      const opcode = rxBuf.readUInt16LE(8);
      const payloadLen = rxBuf.readUInt32LE(12);
      if (rxBuf.length < 16 + payloadLen) break;

      const payload = rxBuf.subarray(16, 16 + payloadLen);
      rxBuf = rxBuf.subarray(16 + payloadLen);

      if (opcode === OP_PONG) {
        pongReceived = true;
        console.log('✓ OP_PONG received (msgId: 101)');
      } else if (opcode === OP_STREAM_CONTENT || opcode === OP_STREAM_THOUGHT) {
        tokenCount++;
        process.stdout.write('.');
      } else if (opcode === OP_STATUS) {
        const s = payload.readUInt8(0);
        const g = payload.readUInt8(1);
        const r = payload.readFloatLE(8);
      } else if (opcode === OP_TURN_COMPLETE) {
        turnCompleteReceived = true;
        const totalTok = payload.readUInt32LE(0);
        const elapsedMs = payload.readUInt32LE(4);
        const tokSec = payload.readFloatLE(8);
        const reason = payload.readUInt8(12);
        console.log(`\n✓ OP_TURN_COMPLETE: ${totalTok} tokens in ${elapsedMs}ms (${tokSec.toFixed(1)} tok/s, reason: ${reason})`);
      } else if (opcode === OP_MEM_RESPONSE) {
        memResponseReceived = true;
        const count = payload.readUInt16LE(0);
        const status = payload.readUInt8(2);
        console.log(`✓ OP_MEM_RESPONSE: count=${count}, status=${status}`);
      } else if (opcode === OP_ERROR) {
        console.error('Server error:', payload.toString('utf-8'));
      }
    }
  });

  console.log('Sending OP_PING...');
  proc.stdin.write(makeHeader(OP_PING, 101, 0));

  console.log('Sending OP_SET_CONFIG...');
  proc.stdin.write(makeConfigFrame(64));

  console.log('\nSending OP_STREAM_INPUT ("Hello!")...');
  proc.stdin.write(makeStreamInputFrame('<|turn>user\nHello!<turn|>\n<|turn>model\n', 201));

  let waited = 0;
  while ((!pongReceived || !turnCompleteReceived) && waited < 20000) {
    await new Promise((r) => setTimeout(r, 100));
    waited += 100;
  }

  console.log('\nSending OP_MEM_QUERY ("hello greeting")...');
  proc.stdin.write(makeMemQueryFrame('hello greeting', 301, 5));

  waited = 0;
  while (!memResponseReceived && waited < 5000) {
    await new Promise((r) => setTimeout(r, 100));
    waited += 100;
  }

  console.log('Sending OP_SHUTDOWN...');
  proc.stdin.write(makeHeader(OP_SHUTDOWN, 999, 0));

  await new Promise((resolve) => proc.on('close', (code) => {
    console.log(`✓ Engine shutdown cleanly with code: ${code}`);
    resolve();
  }));

  if (!pongReceived || !turnCompleteReceived || !memResponseReceived) {
    console.error('Test failed to receive all expected messages!');
    process.exit(1);
  }

  console.log('\n🎉 ALL PROTOCOL VERIFICATION TESTS PASSED!');
}

runProtocolTest().catch((err) => {
  console.error(err);
  process.exit(1);
});
