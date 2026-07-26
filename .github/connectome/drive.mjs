#!/usr/bin/env node
// Drives one scheduled run of the connectome-host headless daemon.
//
// The host is started separately with `--headless --exit-when-idle` and listens
// on a Unix socket at $DATA_DIR/ipc.sock. This connects, subscribes to the
// event stream, sends a single kickoff message, and mirrors what the agent does
// into stdout so the Actions log stays readable. The host exits on its own once
// the agent goes quiescent; we exit when the socket closes.

import { connect } from 'node:net';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
const SOCKET = process.env.IPC_SOCKET || join(DATA_DIR, 'ipc.sock');
const READY_TIMEOUT_MS = Number(process.env.READY_TIMEOUT_MS || 180_000);

const KICKOFF =
  process.env.KICKOFF_MESSAGE ||
  [
    'A scheduled run has just started. The checkout is fresh and your Chronicle',
    'store has been restored, so anything you remember is genuinely yours.',
    '',
    'The runner is yours until you stop. When you have nothing further you want',
    'to do, just say so and stop — the run ends cleanly and your memory is saved.',
  ].join('\n');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForSocket() {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (existsSync(SOCKET)) return true;
    await sleep(500);
  }
  return false;
}

if (!(await waitForSocket())) {
  console.error(`[drive] socket never appeared at ${SOCKET} after ${READY_TIMEOUT_MS} ms`);
  process.exit(1);
}

const sock = connect(SOCKET);
let buffer = '';
let kicked = false;
let sawInference = false;

const send = (obj) => sock.write(`${JSON.stringify(obj)}\n`);

sock.on('connect', () => {
  console.log(`[drive] connected to ${SOCKET}`);
  // Empty subscription would mean "nothing"; name the events we mirror.
  send({
    type: 'subscribe',
    events: [
      'lifecycle',
      'inference:speech',
      'inference:started',
      'inference:completed',
      'inference:failed',
      'tool:started',
      'command-output',
    ],
  });
});

function kickOnce(why) {
  if (kicked) return;
  kicked = true;
  console.log(`[drive] sending kickoff (${why})`);
  send({ type: 'text', content: KICKOFF });
}

// If the host never emits a ready lifecycle we still want the run to happen.
setTimeout(() => kickOnce('ready-timeout fallback'), 20_000);

sock.on('data', (chunk) => {
  buffer += chunk.toString();
  let nl;
  while ((nl = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line) continue;
    let ev;
    try {
      ev = JSON.parse(line);
    } catch {
      continue;
    }

    if (ev.type === 'lifecycle') {
      console.log(`[lifecycle] ${ev.phase}${ev.reason ? ` (${ev.reason})` : ''}`);
      if (ev.phase === 'ready') kickOnce('lifecycle:ready');
      continue;
    }
    if (ev.type === 'inference:started') {
      sawInference = true;
      continue;
    }
    if (ev.type === 'inference:failed') {
      console.error(`[inference failed] ${ev.error ?? ''}`);
      continue;
    }
    if (ev.type === 'inference:speech' && ev.content) {
      console.log(`\n${String(ev.content).trimEnd()}`);
      continue;
    }
    if (ev.type === 'tool:started') {
      const input = ev.input ? JSON.stringify(ev.input) : '';
      console.log(`  · ${ev.tool}${input ? ` ${input.slice(0, 160)}` : ''}`);
      continue;
    }
    if (ev.type === 'command-output' && ev.text) console.log(ev.text);
  }
});

sock.on('error', (err) => {
  console.error(`[drive] socket error: ${err.message}`);
  process.exit(1);
});

sock.on('close', () => {
  console.log(`[drive] socket closed; inference observed: ${sawInference}`);
  process.exit(0);
});
