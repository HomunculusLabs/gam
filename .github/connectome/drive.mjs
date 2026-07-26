#!/usr/bin/env node
// Drive one bounded Connectome run over the headless JSONL socket.
//
// The host deliberately runs without --exit-when-idle. Its 500 ms idle signal
// observes agent states, not the framework event queue, so treating that signal
// as an instruction to stop can close the queue while a late tool result is
// still scheduling the next inference. Lifecycle idleness is therefore
// diagnostic only. We arm shutdown solely from a terminal framework event:
// either a completed, user-visible turn or an explicit inference failure.
// Any subsequent work cancels that decision; only a quiet terminal interval
// permits an explicit graceful shutdown.

import { connect } from 'node:net';
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
const SOCKET = process.env.IPC_SOCKET || join(DATA_DIR, 'ipc.sock');
const READY_TIMEOUT_MS = Number(process.env.READY_TIMEOUT_MS || 180_000);
const IDLE_SETTLE_MS = Number(process.env.IDLE_SETTLE_MS || 5_000);
// Once inference has failed, the host gets a bounded window to produce a real
// completion before the run is declared unable to proceed. Unbounded, a host
// that answers a failure with a retry that never terminates leaves no armed
// timer at all: the retry's `inference:started` cancels the terminal
// settlement and nothing re-arms it, because `inference:started` is not itself
// a terminal candidate. Runs 30188621552 and 30189939720 each sat in exactly
// that state for ~5 h until the 360-minute job timeout, and a timed-out job
// reports `cancelled` — indistinguishable from "still running".
const RECOVERY_TIMEOUT_MS = Number(process.env.RECOVERY_TIMEOUT_MS || 600_000);
// A graceful shutdown asks the host to close the socket. A host that is already
// wedged cannot honour it, so the request itself needs a deadline.
const SHUTDOWN_GRACE_MS = Number(process.env.SHUTDOWN_GRACE_MS || 30_000);
const RUN_ID = process.env.GITHUB_RUN_ID || 'unknown';
const RUN_MARKER = process.env.RUN_MARKER;
const HOST_PID = Number(process.env.CONNECTOME_HOST_PID);

if (!RUN_MARKER || !Number.isSafeInteger(HOST_PID) || HOST_PID <= 0) {
  console.error('[drive] RUN_MARKER and a valid CONNECTOME_HOST_PID are required');
  process.exit(1);
}

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
    // The workflow runs on Linux, so procfs gives us a read-only liveness
    // check. A startup crash should surface in one polling interval, without
    // sending any signal to the Claude host.
    if (!existsSync(`/proc/${HOST_PID}`)) {
      throw new Error(`host process ${HOST_PID} exited before opening ${SOCKET}`);
    }
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
let terminalSettleTimer = null;
let recoveryTimer = null;
let shutdownGraceTimer = null;

const send = (message) => {
  if (!sock.writable) {
    terminalFailure ??= 'headless socket became unwritable';
    return false;
  }
  return sock.write(`${JSON.stringify(message)}\n`);
};

const cancelTerminalSettlement = () => {
  if (terminalSettleTimer !== null) {
    clearTimeout(terminalSettleTimer);
    terminalSettleTimer = null;
  }
};

const clearRecovery = () => {
  if (recoveryTimer !== null) {
    clearTimeout(recoveryTimer);
    recoveryTimer = null;
  }
};

// Report what the run reached and leave. Used both when the host closes the
// socket and when it stops answering, so neither path can outlive the job.
const finish = () => {
  clearTimeout(readyFallback);
  cancelTerminalSettlement();
  clearRecovery();
  if (shutdownGraceTimer !== null) {
    clearTimeout(shutdownGraceTimer);
    shutdownGraceTimer = null;
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
};

const requestShutdown = (failure = null) => {
  if (shutdownRequested) return;
  shutdownRequested = true;
  terminalFailure ??= failure;
  clearRecovery();
  console.log(
    failure
      ? `[drive] refusing a false-green run: ${failure}`
      : `[drive] completed turn stayed quiescent for ${IDLE_SETTLE_MS} ms; requesting shutdown`,
  );
  send({ type: 'shutdown', graceful: true });
  shutdownGraceTimer = setTimeout(() => {
    shutdownGraceTimer = null;
    terminalFailure ??= `host ignored a graceful shutdown for ${SHUTDOWN_GRACE_MS} ms`;
    console.error(`[drive] ${terminalFailure}`);
    finish();
  }, SHUTDOWN_GRACE_MS);
};

// Armed by the first inference failure and cleared only by a real completion.
// Deliberately NOT cleared by cancelTerminalSettlement: the whole defect is
// that a silent non-terminal event can erase the terminal verdict, so the
// bound on "the host stopped making progress" must survive that erasure.
const armRecovery = (failure) => {
  if (recoveryTimer !== null) return;
  recoveryTimer = setTimeout(() => {
    recoveryTimer = null;
    requestShutdown(`no inference completed within ${RECOVERY_TIMEOUT_MS} ms of ${failure}`);
  }, RECOVERY_TIMEOUT_MS);
};

const settleTerminal = (failure = null) => {
  cancelTerminalSettlement();
  terminalSettleTimer = setTimeout(() => {
    terminalSettleTimer = null;
    requestShutdown(failure);
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
  writeFileSync(RUN_MARKER, `${RUN_ID}\n`, { flag: 'wx' });
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
      if (event.phase === 'idle' && !sawCompletion && !lastInferenceFailure) {
        console.log('[drive] premature lifecycle idle ignored; no terminal inference event yet');
      }
      if (event.phase === 'exiting') sawExiting = true;
      continue;
    }

    // A framework event after a terminal candidate means work continued.
    // Cancel the candidate before interpreting this event; a new terminal
    // event below may arm a fresh quiet interval.
    cancelTerminalSettlement();

    if (event.type === 'inference:started') {
      sawInference = true;
      continue;
    }
    if (event.type === 'inference:completed') {
      sawCompletion = true;
      lastInferenceFailure = null;
      clearRecovery();
      continue;
    }
    if (event.type === 'inference:failed') {
      lastInferenceFailure = String(event.error ?? 'unknown inference failure');
      console.error(`[inference failed] ${lastInferenceFailure}`);
      settleTerminal(`inference failed: ${lastInferenceFailure}`);
      armRecovery(lastInferenceFailure);
      continue;
    }
    if (event.type === 'inference:speech' && event.content) {
      sawFinalSpeech = true;
      console.log(`\n${String(event.content).trimEnd()}`);
      if (sawCompletion) settleTerminal();
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
  if (!shutdownRequested) {
    terminalFailure ??= 'host closed the socket before the driver requested shutdown';
  } else if (!sawExiting) {
    terminalFailure ??= 'host closed without acknowledging graceful shutdown';
  }
  finish();
});
