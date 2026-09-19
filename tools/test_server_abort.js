import { spawn } from 'child_process';

const MAGIC = 0x53554C58;
const VERSION = 1;
const OP_STREAM_INPUT = 0x0001;
const OP_ABORT = 0x0002;
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
  payload.writeUInt8(0x00, 0); // mode = text
  textBytes.copy(payload, 8);
  const hdr = makeHeader(OP_STREAM_INPUT, msgId, payload.length);
  return Buffer.concat([hdr, payload]);
}

function makeAbortFrame(msgId = 99) {
  return makeHeader(OP_ABORT, msgId, 0);
}

function makeShutdownFrame() {
  return makeHeader(OP_SHUTDOWN, 0, 0);
}

async function runAbortTest() {
  console.log('Testing OP_ABORT with ./zig-out/bin/infer --model ../gemma-4-E2B --serve ...');
  const proc = spawn('./zig-out/bin/infer', ['--model', '../gemma-4-E2B', '--serve'], {
    stdio: ['pipe', 'pipe', 'inherit'],
  });

  let rxBuffer = Buffer.alloc(0);
  let tokensReceived = 0;
  let turnComplete = null;

  proc.stdout.on('data', (chunk) => {
    rxBuffer = Buffer.concat([rxBuffer, chunk]);
    while (rxBuffer.length >= 16) {
      const magic = rxBuffer.readUInt32LE(0);
      if (magic !== MAGIC) break;
      const opcode = rxBuffer.readUInt16LE(8);
      const payloadLen = rxBuffer.readUInt32LE(12);
      if (rxBuffer.length < 16 + payloadLen) break;

      const payload = rxBuffer.subarray(16, 16 + payloadLen);
      rxBuffer = rxBuffer.subarray(16 + payloadLen);

      if (opcode === OP_STREAM_CONTENT || opcode === OP_STREAM_THOUGHT) {
        tokensReceived++;
        if (tokensReceived === 5) {
          console.log(`\nEmitted 5 tokens. Sending OP_ABORT immediately!`);
          proc.stdin.write(makeAbortFrame());
        }
      } else if (opcode === OP_TURN_COMPLETE) {
        const totalTok = payload.readUInt32LE(0);
        const elapsedMs = payload.readUInt32LE(4);
        const reason = payload.readUInt8(12);
        turnComplete = { totalTok, elapsedMs, reason };
        console.log(`✓ OP_TURN_COMPLETE received with reason code: ${reason} (2 = ABORTED)`);
      }
    }
  });

  proc.stdin.write(makeStreamInputFrame('Tell me a long 500 word story about galaxies and stars', 1));

  const start = Date.now();
  while (!turnComplete && Date.now() - start < 10000) {
    await new Promise((r) => setTimeout(r, 100));
  }

  proc.stdin.write(makeShutdownFrame());
  await new Promise((r) => proc.on('close', r));

  if (!turnComplete) throw new Error('Turn complete not received after abort');
  if (turnComplete.reason !== 2) throw new Error(`Expected stop reason 2 (ABORTED), got ${turnComplete.reason}`);

  console.log(`✓ Generation successfully halted at token ${tokensReceived}!`);
  console.log('🎉 OP_ABORT TEST PASSED!');
}

runAbortTest().catch((err) => {
  console.error('Abort test failed:', err);
  process.exit(1);
});
