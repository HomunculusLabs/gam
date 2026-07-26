#!/usr/bin/env node
// Drive one bounded Connectome run over the headless JSONL socket.
//
// The host deliberately runs without --exit-when-idle. Its 500 ms idle signal
// observes agent states, not the framework event queue, so treating that signal
// as an instruction to stop can close the queue while a late tool result is
// still scheduling the next inference. Lifecycle idleness is therefore
// diagnostic only.
//
// TERMINATION HAS FIVE INDEPENDENT ROUTES, in descending preference. Every one
// of them ends the run; none of them can be starved by the others:
//
//   1. A completed, user-visible turn (`inference:completed` then
//      `inference:speech`) or an explicit `inference:turn_ended`, followed by a
//      quiet interval. This is the success certificate.
//   2. A terminal failure event — `inference:exhausted` or `inference:aborted`.
//      These are the framework's real terminal failure states
//      (agent-tree-reducer.ts: both flip the node to `cancelled`). The one
//      operational exception is Anthropic's account-capacity limit: a
//      scheduled worker that has no capacity left defers cleanly to the next
//      scheduled run, while still emitting a warning and persisting its memory.
//   3. The retry budget being spent: MAX_INFERENCE_ATTEMPTS consecutive
//      `inference:failed` with no intervening progress. `inference:failed` is
//      ATTEMPT-level telemetry — startAgentStream emits it before consulting
//      DefaultErrorPolicy and then retries — so a single one is not terminal,
//      but a full run of them is, whether or not a terminal event follows.
//   4. The recovery window (RECOVERY_TIMEOUT_MS): a failure has been seen and
//      no completion followed it. Armed by failure, cleared by completion, and
//      immune to the cancellation that erased the verdict in the first place.
//   5. Backstops that cannot be reasoned away: a stall watchdog (no
//      substantive event for STALL_TIMEOUT_MS) and an absolute deadline
//      (RUN_DEADLINE_MS). Both sit well inside the Actions job cap so the run
//      still saves its memory and prints its logs.
//
// Route 5 exists because routes 1-4 all depend on recognising event NAMES, and
// this driver has already been wedged once by a name it did not know: an
// unhandled event cancelled the terminal candidate and then matched nothing, so
// run 30189939720 sat silent from 06:46:09Z to the 11:47:09Z Actions ceiling —
// five hours of a six-hour job spent doing nothing at all. A protocol-level
// gate that only ever ARMS on names it knows must have a route that needs no
// name whatsoever.

import { connect } from 'node:net';
import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const DATA_DIR = process.env.DATA_DIR || './data';
const SOCKET = process.env.IPC_SOCKET || join(DATA_DIR, 'ipc.sock');
const READY_TIMEOUT_MS = Number(process.env.READY_TIMEOUT_MS || 180_000);
const IDLE_SETTLE_MS = Number(process.env.IDLE_SETTLE_MS || 5_000);
// No substantive (non-lifecycle) event for this long means the run is dead.
// Must exceed the longest legitimate silence, which is one tool call: the MCPL
// transport abandons a tools/call at 60 s, so any real gap is minutes, not
// tens of minutes.
const STALL_TIMEOUT_MS = Number(process.env.STALL_TIMEOUT_MS || 20 * 60_000);
// Absolute cap on the agent stage. Deliberately far below the job's
// timeout-minutes so "Save memory", the host-log tail and the artifact upload
// all still run.
const RUN_DEADLINE_MS = Number(process.env.RUN_DEADLINE_MS || 300 * 60_000);
// A graceful shutdown asks the host to close the socket. A host that is already
// wedged cannot honour it, so the request itself needs a deadline.
const SHUTDOWN_GRACE_MS = Number(process.env.SHUTDOWN_GRACE_MS || 30_000);
// Once inference has failed, the host gets a bounded window to produce a real
// completion before the run is declared unable to proceed. This bound is armed
// by a failure and cleared only by a completion, and — unlike the terminal
// candidate — it is deliberately immune to cancellation, because the defect
// being guarded is precisely that a silent non-terminal event can erase the
// terminal verdict. Runs 30188621552 and 30189939720 each sat in exactly that
// state for ~5 h until the 360-minute job timeout, and a timed-out job reports
// `cancelled` — indistinguishable from "still running".
const RECOVERY_TIMEOUT_MS = Number(process.env.RECOVERY_TIMEOUT_MS || 600_000);
const HEARTBEAT_MS = Number(process.env.HEARTBEAT_MS || 60_000);
// agent-framework 0.6.8: initial attempt + three DefaultErrorPolicy retries.
const MAX_INFERENCE_ATTEMPTS = Number(process.env.MAX_INFERENCE_ATTEMPTS || 4);
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

// ---------------------------------------------------------------------------
// Logging. Every line carries elapsed time: the previous driver printed bare
// lines, so a live run was indistinguishable from a hung one and a wedged run
// was indistinguishable from a slow build.
// ---------------------------------------------------------------------------
const START = Date.now();
const elapsedMs = () => Date.now() - START;
const clock = (ms = elapsedMs()) => {
  const total = Math.floor(ms / 1000);
  const h = String(Math.floor(total / 3600)).padStart(2, '0');
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const s = String(total % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
};
const say = (line) => console.log(`[${clock()}] ${line}`);
const cry = (line) => console.error(`[${clock()}] ${line}`);

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
  cry(`[drive] ${error.message}`);
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
let terminalSettleTimer = null;
let shutdownGraceTimer = null;
let recoveryTimer = null;
let lastSubstantiveAt = Date.now();
let consecutiveFailures = 0;
let lastInferenceFailure = null;
let rateLimitDeferred = false;
const counts = { inferences: 0, completions: 0, failures: 0, tools: 0, toolFailures: 0, events: 0 };
const unknownEventTypes = new Set();

// Terminal FAILURE states in agent-framework 0.6.8. Both flip the agent node
// to `cancelled` in the host's reducer; neither is followed by more work.
const TERMINAL_FAILURE_EVENTS = new Set(['inference:exhausted', 'inference:aborted']);
const ACCOUNT_RATE_LIMIT = /This request would exceed your account's rate limit/i;
const isAccountRateLimit = (...errors) =>
  errors.some((error) => ACCOUNT_RATE_LIMIT.test(String(error ?? '')));

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

// Armed by the first inference failure, cleared only by a real completion, and
// never cleared by cancelTerminalSettlement — see RECOVERY_TIMEOUT_MS.
const armRecovery = (failure) => {
  if (recoveryTimer !== null) return;
  recoveryTimer = setTimeout(() => {
    recoveryTimer = null;
    requestShutdown(`no inference completed within ${RECOVERY_TIMEOUT_MS} ms of ${failure}`);
  }, RECOVERY_TIMEOUT_MS);
  recoveryTimer.unref?.();
};

const requestShutdown = (failure = null) => {
  if (shutdownRequested) return;
  shutdownRequested = true;
  terminalFailure ??= failure;
  clearRecovery();
  say(
    failure
      ? `[drive] refusing a false-green run: ${failure}`
      : rateLimitDeferred
        ? '[drive] Anthropic account capacity exhausted; deferring to the next scheduled run'
      : `[drive] completed turn stayed quiescent for ${IDLE_SETTLE_MS} ms; requesting shutdown`,
  );
  send({ type: 'shutdown', graceful: true });

  // A host that ignores graceful shutdown must not reproduce the very hang
  // this driver exists to prevent.
  shutdownGraceTimer = setTimeout(() => {
    shutdownGraceTimer = null;
    terminalFailure ??= `host ignored a graceful shutdown for ${SHUTDOWN_GRACE_MS} ms`;
    cry(
      `[drive] host did not close the socket within ${SHUTDOWN_GRACE_MS} ms of a ` +
        'graceful shutdown request; abandoning it',
    );
    summarise('shutdown not honoured');
    process.exit(1);
  }, SHUTDOWN_GRACE_MS);
  shutdownGraceTimer.unref?.();
};

const settleTerminal = (failure = null) => {
  cancelTerminalSettlement();
  terminalSettleTimer = setTimeout(() => {
    terminalSettleTimer = null;
    requestShutdown(failure);
  }, IDLE_SETTLE_MS);
};

const deferForAccountRateLimit = () => {
  const firstDeferral = !rateLimitDeferred;
  rateLimitDeferred = true;
  clearRecovery();
  cancelTerminalSettlement();
  if (firstDeferral) {
    say(
      "::warning title=Connectome deferred by Anthropic capacity::Anthropic's account " +
        'rate limit is exhausted; memory will be saved and work will resume on a later scheduled run',
    );
  }
  settleTerminal();
};

// --- backstops --------------------------------------------------------------
// These two need no knowledge of any event name, which is the whole point.
const STALL_POLL_MS = Math.min(30_000, Math.max(250, Math.floor(STALL_TIMEOUT_MS / 4)));
const stallWatchdog = setInterval(() => {
  if (shutdownRequested) return;
  const quietMs = Date.now() - lastSubstantiveAt;
  if (quietMs >= STALL_TIMEOUT_MS) {
    requestShutdown(
      `no substantive framework event for ${Math.round(quietMs / 1000)} s ` +
        `(stall timeout ${Math.round(STALL_TIMEOUT_MS / 1000)} s)`,
    );
  }
}, STALL_POLL_MS);
stallWatchdog.unref?.();

const runDeadline = setTimeout(() => {
  requestShutdown(
    `absolute run deadline of ${Math.round(RUN_DEADLINE_MS / 60_000)} min reached`,
  );
}, RUN_DEADLINE_MS);
runDeadline.unref?.();

const heartbeat = setInterval(() => {
  if (shutdownRequested) return;
  const quiet = Math.round((Date.now() - lastSubstantiveAt) / 1000);
  say(
    `[heartbeat] events=${counts.events} inferences=${counts.inferences} ` +
      `completed=${counts.completions} failed=${counts.failures} ` +
      `tools=${counts.tools} (${counts.toolFailures} failed) ` +
      `quiet=${quiet}s deadline_in=${clock(Math.max(0, RUN_DEADLINE_MS - elapsedMs()))}`,
  );
}, HEARTBEAT_MS);
heartbeat.unref?.();

function summarise(why) {
  clearInterval(stallWatchdog);
  clearInterval(heartbeat);
  clearTimeout(runDeadline);
  clearRecovery();
  if (shutdownGraceTimer !== null) {
    clearTimeout(shutdownGraceTimer);
    shutdownGraceTimer = null;
  }
  say(
    `[drive] finished after ${clock()} (${why}); inference=${sawInference} ` +
      `completed=${sawCompletion} finalSpeech=${sawFinalSpeech} exiting=${sawExiting}`,
  );
  say(
    `[drive] totals: events=${counts.events} inferences=${counts.inferences} ` +
      `completions=${counts.completions} inference_failures=${counts.failures} ` +
      `tool_calls=${counts.tools} tool_failures=${counts.toolFailures}`,
  );
  if (unknownEventTypes.size > 0) {
    say(`[drive] unhandled event types seen: ${[...unknownEventTypes].sort().join(', ')}`);
  }
  if (lastInferenceFailure) say(`[drive] last inference error: ${lastInferenceFailure}`);
}

const readyFallback = setTimeout(() => kickOnce('ready-timeout fallback'), 20_000);

function kickOnce(reason) {
  if (kicked) return;
  kicked = true;
  clearTimeout(readyFallback);
  say(`[drive] sending unique kickoff for run ${RUN_ID} (${reason})`);
  send({ type: 'text', content: KICKOFF });
}

sock.on('connect', () => {
  say(`[drive] connected to ${SOCKET}`);
  say(
    `[drive] budgets: stall=${Math.round(STALL_TIMEOUT_MS / 60_000)}min ` +
      `deadline=${Math.round(RUN_DEADLINE_MS / 60_000)}min ` +
      `recovery=${Math.round(RECOVERY_TIMEOUT_MS / 60_000)}min ` +
      `settle=${IDLE_SETTLE_MS}ms max_attempts=${MAX_INFERENCE_ATTEMPTS}`,
  );
  // A pre-existing marker would throw here, inside an event handler, killing
  // the driver with a bare stack trace instead of a diagnosis.
  try {
    writeFileSync(RUN_MARKER, `${RUN_ID}\n`, { flag: 'wx' });
  } catch (error) {
    cry(`[drive] could not create run marker ${RUN_MARKER}: ${error.message}`);
  }
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
      cry(`[drive] discarded malformed JSONL: ${line.slice(0, 200)}`);
      continue;
    }

    counts.events += 1;

    if (event.type === 'lifecycle') {
      say(`[lifecycle] ${event.phase}${event.reason ? ` (${event.reason})` : ''}`);
      if (event.phase === 'ready') kickOnce('lifecycle:ready');
      if (event.phase === 'idle' && !sawCompletion && !lastInferenceFailure) {
        say('[drive] lifecycle idle is diagnostic only; no terminal inference event yet');
      }
      if (event.phase === 'exiting') sawExiting = true;
      // Deliberately NOT counted as substantive: lifecycle is a 500 ms poll of
      // agent state, so letting it feed the stall watchdog would let a dead
      // agent look alive forever.
      continue;
    }

    lastSubstantiveAt = Date.now();

    // --- terminal failure: the framework has settled the agent -------------
    if (TERMINAL_FAILURE_EVENTS.has(event.type)) {
      const why = String(event.error ?? event.reason ?? 'no reason given');
      cry(`[${event.type}] ${why}`);
      if (event.type === 'inference:exhausted' && isAccountRateLimit(why, lastInferenceFailure)) {
        deferForAccountRateLimit();
        continue;
      }
      cancelTerminalSettlement();
      settleTerminal(`${event.type}: ${why}`);
      continue;
    }

    // --- terminal success --------------------------------------------------
    if (event.type === 'inference:turn_ended') {
      say('[inference] turn ended by the agent');
      sawCompletion = true;
      counts.completions += 1;
      consecutiveFailures = 0;
      lastInferenceFailure = null;
      clearRecovery();
      cancelTerminalSettlement();
      settleTerminal();
      continue;
    }
    if (event.type === 'inference:completed') {
      sawCompletion = true;
      counts.completions += 1;
      consecutiveFailures = 0;
      lastInferenceFailure = null;
      clearRecovery();
      cancelTerminalSettlement();
      continue;
    }
    if (event.type === 'inference:speech' && event.content) {
      sawFinalSpeech = true;
      // Real output is real progress, so it breaks a run of failed attempts.
      // "Consecutive" has to mean "with nothing in between", or a run that
      // survives one transient error per turn would accumulate a phantom
      // exhaustion over hours of healthy work.
      consecutiveFailures = 0;
      console.log(`\n${String(event.content).trimEnd()}`);
      cancelTerminalSettlement();
      if (sawCompletion) settleTerminal();
      continue;
    }

    // --- attempt telemetry, NOT a terminal state ---------------------------
    // Arming shutdown on a single one of these is what made an early
    // transient error look like a finished run. Arming on nothing at all is
    // what let four of them wedge the driver for five hours. The retry budget
    // is the honest line: spend it, and the run is over.
    if (event.type === 'inference:failed') {
      counts.failures += 1;
      consecutiveFailures += 1;
      lastInferenceFailure = String(event.error ?? 'unknown inference failure');
      cry(
        `[inference failed] attempt ${consecutiveFailures}/${MAX_INFERENCE_ATTEMPTS}: ` +
          lastInferenceFailure,
      );
      cancelTerminalSettlement();
      if (consecutiveFailures >= MAX_INFERENCE_ATTEMPTS) {
        if (isAccountRateLimit(lastInferenceFailure)) {
          deferForAccountRateLimit();
          continue;
        }
        settleTerminal(
          `inference failed on all ${consecutiveFailures} attempts: ${lastInferenceFailure}`,
        );
      }
      // Independent of the retry count, and immune to cancellation: if no
      // completion arrives within the recovery window, the run is over.
      armRecovery(lastInferenceFailure);
      continue;
    }

    if (event.type === 'inference:started') {
      sawInference = true;
      counts.inferences += 1;
      cancelTerminalSettlement();
      continue;
    }
    if (event.type === 'tool:started') {
      counts.tools += 1;
      consecutiveFailures = 0;
      const input = event.input ? JSON.stringify(event.input) : '';
      say(`  · ${event.tool}${input ? ` ${input.slice(0, 160)}` : ''}`);
      cancelTerminalSettlement();
      continue;
    }
    if (event.type === 'tool:failed') {
      counts.toolFailures += 1;
      cry(`[tool failed] ${event.tool ?? 'unknown'}: ${event.error ?? ''}`);
      cancelTerminalSettlement();
      continue;
    }
    if (event.type === 'command-output' && event.text) {
      console.log(event.text);
      cancelTerminalSettlement();
      continue;
    }

    // An event this driver does not know. Cancelling the terminal candidate is
    // the conservative choice — it cannot cause a premature shutdown — and is
    // only safe because the stall watchdog, the deadline and the recovery
    // window do not depend on recognising anything. Report each new name once
    // so the next protocol drift is visible in the log instead of silent.
    if (!unknownEventTypes.has(event.type)) {
      unknownEventTypes.add(event.type);
      say(`[drive] unhandled event type "${event.type}" (treated as progress)`);
    }
    cancelTerminalSettlement();
  }
});

sock.on('error', (error) => {
  terminalFailure ??= `socket error: ${error.message}`;
  cry(`[drive] ${terminalFailure}`);
});

sock.on('close', () => {
  clearTimeout(readyFallback);
  cancelTerminalSettlement();

  if (!shutdownRequested) {
    terminalFailure ??= 'host closed the socket before the driver requested shutdown';
  } else if (!sawExiting) {
    terminalFailure ??= 'host closed without acknowledging graceful shutdown';
  }

  summarise(terminalFailure ? 'failed' : rateLimitDeferred ? 'rate-limit deferred' : 'clean');
  if (terminalFailure) {
    cry(`[drive] failed: ${terminalFailure}`);
    process.exit(1);
  }
  process.exit(0);
});
