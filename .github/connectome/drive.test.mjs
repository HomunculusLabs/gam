// Socket-level regression for drive.mjs.
//
// The primary case (S1) is the exact wedge that burned 5h01m of run
// 30189939720: four attempt-level `inference:failed` events followed by the
// framework's terminal `inference:exhausted`. Its stall timeout and deadline
// are deliberately set to ten minutes so ONLY correct event handling can make
// it pass — a backstop cannot rescue it inside the test's own budget.

import { createServer } from 'node:net';
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const DRIVE = process.env.DRIVE_PATH || new URL('./drive.mjs', import.meta.url).pathname;
const SCENARIO_BUDGET_MS = 25_000;

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : ` — ${detail}`}`);
  if (!cond) failures++;
};

async function scenario({ name, env = {}, play, expectExit, expect = [], reject = [] }) {
  console.log(`\n── ${name}`);
  const dir = mkdtempSync(join(tmpdir(), 'drive-test-'));
  const sockPath = join(dir, 'ipc.sock');

  const inbox = [];
  const waiters = [];
  let conn = null;
  const connWaiters = [];

  const server = createServer((c) => {
    conn = c;
    let buf = '';
    c.on('data', (d) => {
      buf += d.toString();
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          continue;
        }
        inbox.push(msg);
        for (const w of waiters.splice(0)) w();
      }
    });
    for (const w of connWaiters.splice(0)) w();
  });
  await new Promise((r) => server.listen(sockPath, r));

  let stdout = '';
  const child = spawn('node', [DRIVE], {
    env: {
      ...process.env,
      IPC_SOCKET: sockPath,
      RUN_MARKER: join(dir, 'marker'),
      CONNECTOME_HOST_PID: String(process.pid),
      GITHUB_RUN_ID: 'test-run',
      IDLE_SETTLE_MS: '400',
      HEARTBEAT_MS: '600000',
      STALL_TIMEOUT_MS: '600000',
      RUN_DEADLINE_MS: '600000',
      SHUTDOWN_GRACE_MS: '600000',
      RECOVERY_TIMEOUT_MS: '600000',
      ...env,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (d) => (stdout += d.toString()));
  child.stderr.on('data', (d) => (stdout += d.toString()));

  const exited = new Promise((resolve) => child.on('exit', (code) => resolve(code)));

  const send = (event) => conn?.write(`${JSON.stringify(event)}\n`);
  const awaitConn = () =>
    conn ? Promise.resolve() : new Promise((r) => connWaiters.push(r));
  /** Wait until a message matching `pred` has arrived from the driver. */
  const waitFor = (pred, label, timeoutMs = 8000) =>
    new Promise((resolve, reject_) => {
      const tryNow = () => {
        if (inbox.some(pred)) return resolve(true);
        return false;
      };
      if (tryNow()) return;
      const t = setTimeout(() => reject_(new Error(`timed out waiting for ${label}`)), timeoutMs);
      const step = () => {
        if (tryNow()) {
          clearTimeout(t);
          return;
        }
        waiters.push(step);
      };
      waiters.push(step);
    });
  const isShutdown = (m) => m.type === 'shutdown';
  const isKickoff = (m) => m.type === 'text';

  let exitCode = null;
  let error = null;
  try {
    await Promise.race([
      (async () => {
        await awaitConn();
        await play({ send, waitFor, isShutdown, isKickoff, conn: () => conn, inbox });
        exitCode = await exited;
      })(),
      new Promise((_, rej) =>
        setTimeout(
          () => rej(new Error(`scenario exceeded ${SCENARIO_BUDGET_MS} ms — the driver hung`)),
          SCENARIO_BUDGET_MS,
        ),
      ),
    ]);
  } catch (e) {
    error = e;
  }

  try {
    child.kill('SIGKILL');
  } catch { /* already gone */ }
  server.close();

  if (error) {
    check(name, false, error.message);
  } else {
    check(`${name}: exit code ${expectExit}`, exitCode === expectExit, `got ${exitCode}`);
    for (const re of expect) {
      check(`${name}: stdout matches ${re}`, re.test(stdout), stdout.slice(-600));
    }
    for (const re of reject) {
      check(`${name}: stdout must NOT match ${re}`, !re.test(stdout), stdout.slice(-600));
    }
  }
  return stdout;
}

const OVER_BUDGET =
  'Adaptive picker exhausted but 165494 tokens still exceed hard budget 163616';

// S1 — THE REGRESSION. Stall + deadline are 10 minutes: only correct handling
// of inference:exhausted can finish this inside the scenario budget.
await scenario({
  name: 'S1 four inference:failed then inference:exhausted terminates promptly',
  env: { STALL_TIMEOUT_MS: '600000', RUN_DEADLINE_MS: '600000' },
  expectExit: 1,
  expect: [
    /inference:exhausted/,
    /refusing a false-green run/,
    /attempt 4\/4/,
  ],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    for (let i = 0; i < 4; i++) {
      send({ type: 'inference:failed', error: OVER_BUDGET });
      if (i < 3) send({ type: 'inference:started' });
    }
    send({ type: 'inference:exhausted', error: 'retries exhausted' });
    await waitFor(isShutdown, 'graceful shutdown request', 6000);
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

// S2 — the success certificate still works.
await scenario({
  name: 'S2 completed turn plus final speech exits clean',
  expectExit: 0,
  expect: [/completed turn stayed quiescent/, /all done here/],
  reject: [/refusing a false-green run/],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    send({ type: 'inference:completed' });
    send({ type: 'inference:speech', content: 'all done here' });
    await waitFor(isShutdown, 'graceful shutdown request');
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

// S3 — a single transient failure that the framework retries successfully must
// NOT be treated as terminal. This is the false-green bug from run 30188446634.
await scenario({
  name: 'S3 one transient failure followed by a successful retry is not terminal',
  expectExit: 0,
  expect: [/attempt 1\/4/, /recovered fine/],
  reject: [/refusing a false-green run/, /attempt 2\/4/],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    send({ type: 'inference:failed', error: 'transient 529' });
    send({ type: 'inference:started' });
    send({ type: 'tool:started', tool: 'bash', input: { command: 'cargo check' } });
    send({ type: 'inference:completed' });
    send({ type: 'inference:speech', content: 'recovered fine' });
    await waitFor(isShutdown, 'graceful shutdown request');
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

// S4 — an event name the driver does not know must not be able to wedge it.
// This is the general form of the S1 bug: the unknown event cancels the
// terminal candidate, and only a name-independent backstop can recover.
await scenario({
  name: 'S4 unknown event that cancels a terminal candidate cannot wedge the run',
  env: { STALL_TIMEOUT_MS: '2500' },
  expectExit: 1,
  expect: [
    /unhandled event type "inference:stream_restarted"/,
    /no substantive framework event for/,
  ],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    send({ type: 'inference:completed' });
    send({ type: 'inference:speech', content: 'arming a terminal candidate' });
    send({ type: 'inference:stream_restarted' });
    // Lifecycle idle must NOT feed the watchdog — a 500 ms state poll cannot be
    // allowed to make a dead agent look alive.
    const noise = setInterval(() => send({ type: 'lifecycle', phase: 'idle' }), 300);
    await waitFor(isShutdown, 'stall-watchdog shutdown', 12_000);
    clearInterval(noise);
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

// S5 — the absolute deadline holds even while work is still flowing.
await scenario({
  name: 'S5 absolute run deadline terminates a run that never stops working',
  env: { RUN_DEADLINE_MS: '2500' },
  expectExit: 1,
  expect: [/absolute run deadline/],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    const busy = setInterval(() => send({ type: 'tool:started', tool: 'bash' }), 300);
    await waitFor(isShutdown, 'deadline shutdown', 12_000);
    clearInterval(busy);
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

// S6 — the other terminal failure state in the framework reducer.
await scenario({
  name: 'S6 inference:aborted is terminal',
  expectExit: 1,
  expect: [/inference:aborted/, /refusing a false-green run/],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    send({ type: 'inference:aborted', reason: 'user cancel' });
    await waitFor(isShutdown, 'graceful shutdown request', 6000);
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

// S7 — a host that ignores graceful shutdown must not recreate the hang.
await scenario({
  name: 'S7 host ignoring graceful shutdown does not hang the driver',
  env: { SHUTDOWN_GRACE_MS: '1500' },
  expectExit: 1,
  expect: [/did not close the socket within 1500 ms/],
  play: async ({ send, waitFor, isShutdown, isKickoff }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    send({ type: 'inference:exhausted', error: 'retries exhausted' });
    await waitFor(isShutdown, 'graceful shutdown request', 6000);
    // Deliberately never acknowledge and never close.
  },
});

// S8 — the pre-existing false-green guard: an unrequested close is a failure.
await scenario({
  name: 'S8 host closing before the driver asks is a failure',
  expectExit: 1,
  expect: [/closed the socket before the driver requested shutdown/],
  play: async ({ send, waitFor, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    conn().end();
  },
});

// S9 — the recovery window: a failure was seen, work keeps churning, and no
// completion ever arrives. Every churn event cancels the terminal candidate and
// keeps the stall watchdog fed, so only a bound that survives cancellation can
// end this. MAX_INFERENCE_ATTEMPTS is raised so the retry budget cannot be what
// fires, isolating the recovery route.
await scenario({
  name: 'S9 recovery window ends a run that fails once and then only churns',
  env: {
    RECOVERY_TIMEOUT_MS: '2500',
    MAX_INFERENCE_ATTEMPTS: '99',
    STALL_TIMEOUT_MS: '600000',
    RUN_DEADLINE_MS: '600000',
  },
  expectExit: 1,
  expect: [/no inference completed within 2500 ms/],
  play: async ({ send, waitFor, isShutdown, isKickoff, conn }) => {
    send({ type: 'lifecycle', phase: 'ready' });
    await waitFor(isKickoff, 'kickoff');
    send({ type: 'inference:started' });
    send({ type: 'inference:failed', error: 'over budget' });
    // Each of these cancels the terminal candidate and refreshes the watchdog.
    const churn = setInterval(() => send({ type: 'inference:started' }), 400);
    await waitFor(isShutdown, 'recovery-window shutdown', 12_000);
    clearInterval(churn);
    send({ type: 'lifecycle', phase: 'exiting' });
    conn().end();
  },
});

console.log(`\n${failures === 0 ? 'ALL DRIVE CHECKS PASSED' : `${failures} DRIVE CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
