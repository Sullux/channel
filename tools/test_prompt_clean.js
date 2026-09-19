const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const constants = require('../tui/lib/protocol/constants');
const framing = require('../tui/lib/protocol/framing');

const kernel = fs.readFileSync(path.resolve(__dirname, '../tui/PROMPT_KERNEL.md'), 'utf-8').trim();
const prompt = `<|turn>system\n<|think|>\n${kernel}\n<turn|>\n<|turn>user\nHow are you doing today?<turn|>\n<|turn>model\n`;

function runEngine(args) {
  return new Promise((resolve) => {
    const binPath = path.resolve(__dirname, '../zig-out/bin/infer');
    const child = spawn(binPath, ['--model', '../gemma-4-12B-it', '--serve', ...args], {
      stdio: ['pipe', 'pipe', 'inherit'],
    });

    let rxBuffer = Buffer.alloc(0);
    let output = '';

    child.stdout.on('data', (chunk) => {
      rxBuffer = Buffer.concat([rxBuffer, chunk]);
      while (true) {
        const frame = framing.parsedFrame(rxBuffer);
        if (!frame) break;
        rxBuffer = rxBuffer.subarray(16 + frame.header.payloadLen);
        if (frame.header.opcode === constants.OP_STREAM_THOUGHT) {
          const text = frame.payload.subarray(24).toString('utf-8');
          output += '[THK]' + text;
        } else if (frame.header.opcode === constants.OP_STREAM_CONTENT) {
          const text = frame.payload.subarray(24).toString('utf-8');
          output += '[CNT]' + text;
        } else if (frame.header.opcode === constants.OP_TURN_COMPLETE) {
          child.stdin.write(framing.shutdownFrame());
        }
      }
    });

    child.on('close', () => resolve(output));

    const cfg = framing.configFrame(512, 0.0, 1.0, 0.0, 200, 1);
    child.stdin.write(cfg);
    child.stdin.write(framing.streamInputFrame(prompt, 1));
  });
}

async function main() {
  console.log('Running test without trailing thought token...');
  const res = await runEngine(['--gpu', '--mixed']);
  console.log('Result:\n' + res.replace(/\u2581/g, ' '));
}

main();
