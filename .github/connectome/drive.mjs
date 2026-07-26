#!/usr/bin/env node
// Drive one bounded Connectome run over the headless JSONL socket.
//
// The host deliberately runs without --exit-when-idle. Its 500 ms idle signal
// observes agent states, not the framework event queue, so treating that signal
// as an instruction to stop can close the queue while a late tool result is
// still scheduling the next inference. We instead require a completed,
// user-visible turn and a quiet interval, then request an explicit graceful
// shutdown.

import { connect } from 'node:net';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
const SOCKET = process.env.IPC_SOCKET || join(DATA_DIR, 'ipc.sock');
const READY_TIMEOUT_MS = Number(process.env.READY_TIMEOUT_MS || 180_000);
const IDLE_SETTLE_MS = Number(process.env.IDLE_SETTLE_MS || 5_000);
const RUN_ID = process.env.GITHUB_RUN_ID || 'unknown';

const KICKOFF =
  process.env.KICKOFF_MESSAGE ||
  [
    `Scheduled run ${RUN_ID} has started. The checkout is fresh and your Chronicle`,
    'store has been restored, so anything you remember is genuinely yours.',
    '',
    'The runner is yours until you stop. When you have nothing further you want',
    'to do, say so explicitly; your completed response is the handoff that lets',
    'the runner save your memory and end this run cleanly.',
  ].join('\n');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForSocket() {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (existsSync(SOCKET)) return;
    await sleep(500);
  }
  throw new Error(`socket never appeared at ${SOCKET} after ${READY_TIMEOUT_MS} ms`);
}

try {
  await waitForSocket();
} catch (error) {
  console.error(`[drive] ${error.message}`);
  process.exit(1);
}

const sock = connect(SOCKET);
let buffer = '';
let kicked = false;
let sawInference = false;
let sawCompletion = false;
let sawFinalSpeech = false;
let sawExiting = false;
let shutdownRequested = false;
let terminalFailure = null;
let lastInferenceFailure = null;
let idleSettleTimer = null;

const send = (message) => {
  if (!sock.writable) {
    terminalFailure ??= 'headless socket became unwritable';
    return false;
  }
  return sock.write(`${JSON.stringify(message)}\n`);
};

const cancelIdleSettlement = () => {
  if (idleSettleTimer !== null) {
    clearTimeout(idleSettleTimer);
    idleSettleTimer = null;
  }
};

const requestShutdown = (failure = null) => {
  if (shutdownRequested) return;
  shutdownRequested = true;
  terminalFailure ??= failure;
  console.log(
    failure
      ? `[drive] refusing a false-green run: ${failure}`
      : `[drive] completed turn stayed quiescent for ${IDLE_SETTLE_MS} ms; requesting shutdown`,
  );
  send({ type: 'shutdown', graceful: true });
};

const settleIdle = () => {
  cancelIdleSettlement();
  idleSettleTimer = setTimeout(() => {
    idleSettleTimer = null;
    if (!sawInference) {
      requestShutdown('the kickoff caused no inference');
    } else if (!sawCompletion) {
      requestShutdown(
        `inference never completed${lastInferenceFailure ? `: ${lastInferenceFailure}` : ''}`,
      );
    } else if (!sawFinalSpeech) {
      requestShutdown(
        `the agent produced no completed final response${
          lastInferenceFailure ? `; last inference failure: ${lastInferenceFailure}` : ''
        }`,
      );
    } else {
      requestShutdown();
    }
  }, IDLE_SETTLE_MS);
};

const readyFallback = setTimeout(() => kickOnce('ready-timeout fallback'), 20_000);

function kickOnce(reason) {
  if (kicked) return;
  kicked = true;
  clearTimeout(readyFallback);
  console.log(`[drive] sending unique kickoff for run ${RUN_ID} (${reason})`);
  send({ type: 'text', content: KICKOFF });
}

sock.on('connect', () => {
  console.log(`[drive] connected to ${SOCKET}`);
  send({
    type: 'subscribe',
    events: ['lifecycle', 'inference:*', 'tool:*', 'command-output'],
  });
});

sock.on('data', (chunk) => {
  buffer += chunk.toString();
  let newline;
  while ((newline = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;

    let event;
    try {
      event = JSON.parse(line);
    } catch {
      console.error(`[drive] discarded malformed JSONL: ${line.slice(0, 200)}`);
      continue;
    }

    if (event.type === 'lifecycle') {
      console.log(`[lifecycle] ${event.phase}${event.reason ? ` (${event.reason})` : ''}`);
      if (event.phase === 'ready') kickOnce('lifecycle:ready');
      if (event.phase === 'idle') settleIdle();
      if (event.phase === 'exiting') sawExiting = true;
      continue;
    }

    // Any work after lifecycle:idle proves that the host's state-only idle
    // observation raced the event queue. Cancel the pending shutdown and wait
    // for the next confirmed idle transition.
    cancelIdleSettlement();

    if (event.type === 'inference:started') {
      sawInference = true;
      continue;
    }
    if (event.type === 'inference:completed') {
      sawCompletion = true;
      continue;
    }
    if (event.type === 'inference:failed') {
      lastInferenceFailure = String(event.error ?? 'unknown inference failure');
      console.error(`[inference failed] ${lastInferenceFailure}`);
      continue;
    }
    if (event.type === 'inference:speech' && event.content) {
      sawFinalSpeech = true;
      console.log(`\n${String(event.content).trimEnd()}`);
      continue;
    }
    if (event.type === 'tool:started') {
      const input = event.input ? JSON.stringify(event.input) : '';
      console.log(`  · ${event.tool}${input ? ` ${input.slice(0, 160)}` : ''}`);
      continue;
    }
    if (event.type === 'tool:failed') {
      console.error(`[tool failed] ${event.tool ?? 'unknown'}: ${event.error ?? ''}`);
      continue;
    }
    if (event.type === 'command-output' && event.text) console.log(event.text);
  }
});

sock.on('error', (error) => {
  terminalFailure ??= `socket error: ${error.message}`;
  console.error(`[drive] ${terminalFailure}`);
});

sock.on('close', () => {
  clearTimeout(readyFallback);
  cancelIdleSettlement();

  if (!shutdownRequested) {
    terminalFailure ??= 'host closed the socket before the driver requested shutdown';
  } else if (!sawExiting) {
    terminalFailure ??= 'host closed without acknowledging graceful shutdown';
  }

  console.log(
    `[drive] socket closed; inference=${sawInference} completed=${sawCompletion} ` +
      `finalSpeech=${sawFinalSpeech} exiting=${sawExiting}`,
  );
  if (terminalFailure) {
    console.error(`[drive] failed: ${terminalFailure}`);
    process.exit(1);
  }
  process.exit(0);
});
