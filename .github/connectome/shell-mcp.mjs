#!/usr/bin/env node
// Minimal MCP stdio server exposing a `bash` tool.
//
// Connectome's workspace module covers read/write/edit/glob/grep/ls, but this
// agent works on a Rust workspace: it needs cargo, git, gh and the gam CLI.
// Vendored rather than pulled from npm so the exact code the agent can execute
// through is reviewable in-tree.
//
// Protocol: JSON-RPC 2.0, newline-delimited, over stdin/stdout. stdout carries
// protocol traffic ONLY — all diagnostics go to stderr.
//
// WHY EVERY COMMAND IS A BACKGROUND JOB
//
// agent-framework abandons any tools/call that has not been answered in
// DEFAULT_REQUEST_TIMEOUT_MS = 60 s (mcpl/server-connection.ts) and reports it
// to the agent as `MCPL server "sh" did not respond ... the server may be
// hung`. That limit is settable only through `McplServerConfig.requestTimeoutMs`,
// and the recipe's `mcpServers` schema (host recipe.ts) exposes no such field —
// so from in here it cannot be raised at all.
//
// This server used to advertise a 30-minute default and a 6-hour maximum and
// tell the agent to raise `timeout_ms` for slow builds. On a 4-core runner
// building a Rust workspace, that contract was a lie in the common case: the
// call died at 60 s, the agent could not tell an abandoned call from a crashed
// one, and it fell back to firing `sleep 58; tail log` at the wall — burning a
// large share of one run relearning the limit rather than working.
//
// So a command is never run *inside* a request any more. It is spawned as a
// detached job writing to a log file, and the request returns whatever the job
// has produced by the time the safe window closes. A finished job answers with
// its output and exit code exactly as before, which is still the common case. A
// job that is still running answers with its output so far plus a `job_id`; the
// agent calls `bash` again with that id to keep reading. Long builds work, no
// request ever approaches 60 s, and nothing has to lie about timeouts.

import { spawn } from 'node:child_process';
import { createWriteStream, mkdtempSync, openSync, readSync, closeSync, statSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PROTOCOL_VERSION = '2024-11-05';
// The framework's per-request ceiling. Not configurable from the recipe; the
// only safe response is to stay comfortably inside it.
const MCPL_REQUEST_TIMEOUT_MS = 60_000;
// How long a single tools/call may block before answering with a job handle.
const DEFAULT_WAIT_MS = 45_000;
const MAX_WAIT_MS = 50_000;
// Total wall-clock a job's command may consume across any number of polls.
const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000;
const MAX_TIMEOUT_MS = 6 * 60 * 60 * 1000;
const MAX_OUTPUT_CHARS = 200_000;
const MAX_RETAINED_JOBS = 100;
const CWD = process.env.AGENT_REPO_DIR || process.cwd();

const log = (...a) => process.stderr.write(`[shell-mcp] ${a.join(' ')}\n`);

const JOB_DIR = mkdtempSync(join(tmpdir(), 'gam-shell-jobs-'));
/** @type {Map<string, any>} */
const jobs = new Map();
let nextJobId = 1;

const TOOLS = [
  {
    name: 'bash',
    description:
      'Run a shell command in the gam repository and return its combined ' +
      'stdout/stderr and exit code.\n\n' +
      'Long-running commands are fully supported, but a single call can only ' +
      'block for about 45 seconds — the MCP transport abandons any call that ' +
      'takes 60 s. So a command that is still running when the window closes ' +
      'returns its output so far plus a `job_id`, and the command KEEPS ' +
      'RUNNING. Call `bash` again with that `job_id` (and no `command`) to read ' +
      'the next chunk of output; repeat until you get an exit code. Nothing is ' +
      'lost between polls and you never see the same output twice.\n\n' +
      'Use `timeout_ms` for the total budget the command itself is allowed ' +
      '(default 30 min, max 6 h) — it is NOT a per-call limit and does not ' +
      'need to be raised to survive a slow build. Do not wrap commands in ' +
      '`sleep` to wait for them; poll the job instead. Output beyond 200k ' +
      'characters per read is truncated in the middle.',
    inputSchema: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: 'The shell command to run. Omit when polling a `job_id`.',
        },
        job_id: {
          type: 'string',
          description:
            'Poll an existing job started by an earlier call instead of running ' +
            'a new command. Returns only output produced since your last read.',
        },
        timeout_ms: {
          type: 'number',
          description:
            `Total wall-clock budget for the command across all polls ` +
            `(default ${DEFAULT_TIMEOUT_MS}, max ${MAX_TIMEOUT_MS}). Ignored when polling.`,
        },
        wait_ms: {
          type: 'number',
          description:
            `How long this single call may block waiting for completion ` +
            `(default ${DEFAULT_WAIT_MS}, max ${MAX_WAIT_MS}).`,
        },
      },
      additionalProperties: false,
    },
  },
];

function truncate(s) {
  if (s.length <= MAX_OUTPUT_CHARS) return s;
  const half = Math.floor(MAX_OUTPUT_CHARS / 2);
  const cut = s.length - MAX_OUTPUT_CHARS;
  return `${s.slice(0, half)}\n\n... [${cut} characters truncated] ...\n\n${s.slice(-half)}`;
}

/** Read everything written since this job was last read, and advance its cursor. */
function readDelta(job) {
  let size;
  try {
    size = statSync(job.file).size;
  } catch {
    return '';
  }
  if (size <= job.cursor) return '';
  const span = size - job.cursor;
  // A runaway job must not be able to make this read unbounded.
  const cap = MAX_OUTPUT_CHARS * 2;
  let text = '';
  const fd = openSync(job.file, 'r');
  try {
    if (span <= cap) {
      const buf = Buffer.allocUnsafe(span);
      readSync(fd, buf, 0, span, job.cursor);
      text = buf.toString('utf8');
    } else {
      const half = Math.floor(cap / 2);
      const head = Buffer.allocUnsafe(half);
      readSync(fd, head, 0, half, job.cursor);
      const tailBuf = Buffer.allocUnsafe(half);
      readSync(fd, tailBuf, 0, half, size - half);
      text =
        `${head.toString('utf8')}\n\n... [${span - cap} characters truncated] ...\n\n` +
        tailBuf.toString('utf8');
    }
  } finally {
    closeSync(fd);
  }
  job.cursor = size;
  return text;
}

function reapOldJobs() {
  if (jobs.size <= MAX_RETAINED_JOBS) return;
  // Oldest-first; only ever discard jobs that have already finished, so a live
  // build can never lose its handle.
  for (const [id, job] of jobs) {
    if (jobs.size <= MAX_RETAINED_JOBS) break;
    if (!job.done) continue;
    try {
      rmSync(job.file, { force: true });
    } catch {
      /* the temp dir goes away with the process anyway */
    }
    jobs.delete(id);
  }
}

function startJob(command, timeoutMs) {
  const limit = Math.min(Math.max(Number(timeoutMs) || DEFAULT_TIMEOUT_MS, 1000), MAX_TIMEOUT_MS);
  const id = `j${nextJobId++}`;
  const file = join(JOB_DIR, `${id}.log`);
  const stream = createWriteStream(file);

  // detached: the command becomes a process-group leader, so a timeout can
  // kill the whole tree (cargo -> rustc -> ...) instead of just the shell.
  const child = spawn('bash', ['-lc', command], {
    cwd: CWD,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  });

  const job = {
    id,
    command,
    file,
    cursor: 0,
    startedAt: Date.now(),
    limit,
    done: false,
    exitCode: null,
    signal: null,
    timedOut: false,
    spawnError: null,
    waiters: [],
  };
  jobs.set(id, job);
  reapOldJobs();

  // Both streams share one file so interleaving is preserved. `end: false`
  // keeps either pipe finishing early from truncating the other.
  let openPipes = 2;
  let childClosed = false;
  let streamFinished = false;

  const settleIfReady = () => {
    if (!childClosed || !streamFinished || job.done) return;
    job.done = true;
    job.finishedAt = Date.now();
    for (const resolve of job.waiters.splice(0)) resolve();
  };

  stream.on('finish', () => {
    streamFinished = true;
    settleIfReady();
  });
  stream.on('error', (err) => {
    log(`job ${id} log write failed: ${err.message}`);
    streamFinished = true;
    settleIfReady();
  });

  const closePipe = () => {
    if (--openPipes === 0) stream.end();
  };
  child.stdout.on('end', closePipe);
  child.stderr.on('end', closePipe);
  child.stdout.pipe(stream, { end: false });
  child.stderr.pipe(stream, { end: false });

  const timer = setTimeout(() => {
    job.timedOut = true;
    try {
      process.kill(-child.pid, 'SIGKILL');
    } catch {
      child.kill('SIGKILL');
    }
  }, limit);
  timer.unref?.();

  child.on('error', (err) => {
    clearTimeout(timer);
    job.spawnError = err.message;
    childClosed = true;
    // No pipes will end if the spawn itself failed.
    stream.end();
    settleIfReady();
  });

  child.on('close', (code, signal) => {
    clearTimeout(timer);
    job.exitCode = code;
    job.signal = signal;
    childClosed = true;
    settleIfReady();
  });

  return job;
}

/** Resolve once the job is done, or once `waitMs` has elapsed. */
function waitForJob(job, waitMs) {
  if (job.done) return Promise.resolve();
  return new Promise((resolve) => {
    let settled = false;
    const once = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    job.waiters.push(once);
    const t = setTimeout(once, waitMs);
    t.unref?.();
  });
}

function describe(job) {
  const secs = ((Date.now() - job.startedAt) / 1000).toFixed(1);
  if (job.spawnError) return { status: `failed to spawn: ${job.spawnError}`, isError: true };
  if (!job.done) {
    return {
      status:
        `still running after ${secs}s — job ${job.id}; call bash again with ` +
        `{"job_id":"${job.id}"} to read more output. The command is unaffected ` +
        `by this call returning.`,
      isError: false,
    };
  }
  if (job.timedOut) {
    return {
      status: `timed out after ${job.limit} ms (killed), ${secs}s elapsed`,
      isError: true,
    };
  }
  const code = job.exitCode ?? `signal ${job.signal}`;
  return {
    status: `exit code ${code}, ${secs}s elapsed`,
    isError: job.exitCode !== 0 && job.exitCode != null,
  };
}

async function runOrPoll(args) {
  const command = args?.command;
  const jobId = args?.job_id;
  const wait = Math.min(Math.max(Number(args?.wait_ms) || DEFAULT_WAIT_MS, 100), MAX_WAIT_MS);

  if (typeof jobId === 'string' && jobId) {
    if (typeof command === 'string' && command.trim()) {
      return {
        text: 'pass either `command` (to start a job) or `job_id` (to poll one), not both',
        isError: true,
      };
    }
    const job = jobs.get(jobId);
    if (!job) {
      return {
        text:
          `no such job "${jobId}" — it either never existed or was reaped after ` +
          `finishing. Re-run the command if you still need its output.`,
        isError: true,
      };
    }
    await waitForJob(job, wait);
    const delta = readDelta(job);
    const { status, isError } = describe(job);
    return { text: `${truncate(delta) || '(no new output)'}\n\n[${status}]`, isError };
  }

  if (typeof command !== 'string' || !command.trim()) {
    return { text: 'bash requires a non-empty `command` string (or a `job_id` to poll)', isError: true };
  }

  const job = startJob(command, args?.timeout_ms);
  await waitForJob(job, wait);
  const delta = readDelta(job);
  const { status, isError } = describe(job);
  return { text: `${truncate(delta) || '(no output)'}\n\n[${status}]`, isError };
}

async function handle(msg) {
  const { id, method, params } = msg;
  const reply = (result) => ({ jsonrpc: '2.0', id, result });
  const fail = (code, message) => ({ jsonrpc: '2.0', id, error: { code, message } });

  switch (method) {
    case 'initialize':
      return reply({
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: { name: 'gam-shell', version: '2.0.0' },
      });
    case 'ping':
      return reply({});
    case 'tools/list':
      return reply({ tools: TOOLS });
    case 'tools/call': {
      if (params?.name !== 'bash') return fail(-32602, `unknown tool: ${params?.name}`);
      const args = params?.arguments ?? {};
      if (typeof args.command === 'string' && args.command.trim()) {
        log(`bash: ${args.command.split('\n')[0].slice(0, 160)}`);
      } else if (args.job_id) {
        log(`poll: ${args.job_id}`);
      }
      const { text, isError } = await runOrPoll(args);
      return reply({ content: [{ type: 'text', text }], isError });
    }
    default:
      return fail(-32601, `method not found: ${method}`);
  }
}

// Requests are dispatched concurrently, so track in-flight work: stdin closing
// must not kill a `cargo build` whose result the host is still waiting on.
let inFlight = 0;
let stdinEnded = false;

function maybeExit() {
  if (stdinEnded && inFlight === 0) process.exit(0);
}

async function dispatch(msg) {
  inFlight += 1;
  try {
    const res = await handle(msg);
    if (res) process.stdout.write(`${JSON.stringify(res)}\n`);
  } catch (err) {
    process.stdout.write(
      `${JSON.stringify({ jsonrpc: '2.0', id: msg.id, error: { code: -32603, message: String(err) } })}\n`,
    );
  } finally {
    inFlight -= 1;
    maybeExit();
  }
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let nl;
  while ((nl = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      log('dropped unparseable line');
      continue;
    }
    // Notifications carry no id and expect no response.
    if (msg.id === undefined) continue;
    void dispatch(msg);
  }
});

process.stdin.on('end', () => {
  stdinEnded = true;
  maybeExit();
});

log(
  `ready — cwd=${CWD}, jobs in ${JOB_DIR}, ` +
    `call window ${DEFAULT_WAIT_MS}ms of the transport's ${MCPL_REQUEST_TIMEOUT_MS}ms`,
);
