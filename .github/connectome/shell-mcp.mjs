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

import { spawn } from 'node:child_process';

const PROTOCOL_VERSION = '2024-11-05';
const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000;
const MAX_TIMEOUT_MS = 6 * 60 * 60 * 1000;
const MAX_OUTPUT_CHARS = 200_000;
const CWD = process.env.AGENT_REPO_DIR || process.cwd();

const log = (...a) => process.stderr.write(`[shell-mcp] ${a.join(' ')}\n`);

const TOOLS = [
  {
    name: 'bash',
    description:
      'Run a shell command in the gam repository and return its combined ' +
      'stdout/stderr and exit code. Long-running builds are fine — raise ' +
      '`timeout_ms` for them. Output beyond 200k characters is truncated in ' +
      'the middle; redirect to a file and read it back if you need all of it.',
    inputSchema: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The shell command to run.' },
        timeout_ms: {
          type: 'number',
          description: `Timeout in milliseconds (default ${DEFAULT_TIMEOUT_MS}, max ${MAX_TIMEOUT_MS}).`,
        },
      },
      required: ['command'],
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

function runBash(command, timeoutMs) {
  return new Promise((resolve) => {
    const limit = Math.min(Math.max(Number(timeoutMs) || DEFAULT_TIMEOUT_MS, 1000), MAX_TIMEOUT_MS);
    // detached: the command becomes a process-group leader, so a timeout can
    // kill the whole tree (cargo -> rustc -> ...) instead of just the shell.
    const child = spawn('bash', ['-lc', command], {
      cwd: CWD,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true,
    });

    let out = '';
    let timedOut = false;
    const append = (chunk) => {
      out += chunk;
      // Keep memory bounded on runaway output; the middle is dropped later.
      if (out.length > MAX_OUTPUT_CHARS * 4) out = out.slice(-MAX_OUTPUT_CHARS * 2);
    };
    child.stdout.on('data', (d) => append(d.toString()));
    child.stderr.on('data', (d) => append(d.toString()));

    const timer = setTimeout(() => {
      timedOut = true;
      try { process.kill(-child.pid, 'SIGKILL'); } catch { child.kill('SIGKILL'); }
    }, limit);

    child.on('error', (err) => {
      clearTimeout(timer);
      resolve({ text: `failed to spawn: ${err.message}`, isError: true });
    });

    child.on('close', (code, signal) => {
      clearTimeout(timer);
      const status = timedOut
        ? `timed out after ${limit} ms (killed)`
        : `exit code ${code ?? `signal ${signal}`}`;
      resolve({
        text: `${truncate(out) || '(no output)'}\n\n[${status}]`,
        isError: timedOut || (code !== 0 && code != null),
      });
    });
  });
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
        serverInfo: { name: 'gam-shell', version: '1.0.0' },
      });
    case 'ping':
      return reply({});
    case 'tools/list':
      return reply({ tools: TOOLS });
    case 'tools/call': {
      if (params?.name !== 'bash') return fail(-32602, `unknown tool: ${params?.name}`);
      const command = params?.arguments?.command;
      if (typeof command !== 'string' || !command.trim()) {
        return fail(-32602, 'bash requires a non-empty `command` string');
      }
      log(`bash: ${command.split('\n')[0].slice(0, 160)}`);
      const { text, isError } = await runBash(command, params?.arguments?.timeout_ms);
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

log(`ready — cwd=${CWD}`);
