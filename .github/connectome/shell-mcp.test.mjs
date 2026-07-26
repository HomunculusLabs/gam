// Drive shell-mcp.mjs over real stdio and assert the job protocol.
import { spawn } from 'node:child_process';

const srv = spawn('node', [new URL('./shell-mcp.mjs', import.meta.url).pathname], {
  stdio: ['pipe', 'pipe', 'pipe'],
  env: { ...process.env, AGENT_REPO_DIR: process.cwd() },
});
srv.stderr.on('data', (d) => process.stderr.write(`  [srv] ${d}`));

let buf = '';
const pending = new Map();
srv.stdout.on('data', (d) => {
  buf += d.toString();
  let nl;
  while ((nl = buf.indexOf('\n')) !== -1) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    const msg = JSON.parse(line);
    pending.get(msg.id)?.(msg);
    pending.delete(msg.id);
  }
});

let id = 0;
const rpc = (method, params) =>
  new Promise((resolve) => {
    const myId = ++id;
    pending.set(myId, resolve);
    srv.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id: myId, method, params })}\n`);
  });
const bash = (args) => rpc('tools/call', { name: 'bash', arguments: args });

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : ` — ${detail}`}`);
  if (!cond) failures++;
};
const textOf = (r) => r.result?.content?.[0]?.text ?? JSON.stringify(r);

// 1. handshake
const init = await rpc('initialize', {});
check('initialize returns protocolVersion', init.result?.protocolVersion === '2024-11-05', JSON.stringify(init));

const list = await rpc('tools/list', {});
const schema = list.result?.tools?.[0]?.inputSchema?.properties ?? {};
check('bash schema exposes job_id + wait_ms', 'job_id' in schema && 'wait_ms' in schema, Object.keys(schema).join(','));
check('command is no longer a required field', !(list.result.tools[0].inputSchema.required?.length), JSON.stringify(list.result.tools[0].inputSchema.required));

// 2. fast command completes inline (the common case, unchanged behaviour)
const fast = await bash({ command: 'echo hello; exit 3' });
const fastText = textOf(fast);
check('fast command returns output inline', fastText.includes('hello'), fastText);
check('fast command reports exit code 3', /exit code 3/.test(fastText), fastText);
check('nonzero exit sets isError', fast.result.isError === true, JSON.stringify(fast.result.isError));
check('fast command needs no job handle', !/still running/.test(fastText), fastText);

// 3. slow command returns a job handle instead of blocking past the window
const t0 = Date.now();
const slow = await bash({ command: 'for i in 1 2 3 4 5; do echo line$i; sleep 1; done', wait_ms: 1500 });
const slowMs = Date.now() - t0;
const slowText = textOf(slow);
check('slow call returns inside its wait window', slowMs < 5000, `${slowMs}ms`);
check('slow call reports still running', /still running/.test(slowText), slowText);
check('slow call already shows partial output', /line1/.test(slowText), slowText);
check('still-running is not an error', slow.result.isError === false, JSON.stringify(slow.result.isError));
const jobId = slowText.match(/job (j\d+)/)?.[1];
check('slow call yields a job id', Boolean(jobId), slowText);

// 4. polling returns ONLY new output, then the exit code
const poll1 = await bash({ job_id: jobId, wait_ms: 10_000 });
const poll1Text = textOf(poll1);
check('poll waits for completion and reports exit code 0', /exit code 0/.test(poll1Text), poll1Text);
check('poll returns later lines', /line5/.test(poll1Text), poll1Text);
check('poll does NOT repeat already-read output', !/line1/.test(poll1Text), poll1Text);
check('successful job is not an error', poll1.result.isError === false, JSON.stringify(poll1.result.isError));

// 5. re-polling a finished job is idempotent, not a crash
const poll2 = await bash({ job_id: jobId });
const poll2Text = textOf(poll2);
check('re-poll of finished job says no new output', /no new output/.test(poll2Text), poll2Text);
check('re-poll still reports the exit code', /exit code 0/.test(poll2Text), poll2Text);

// 6. no call may ever approach the transport's 60s limit
const t1 = Date.now();
const capped = await bash({ command: 'sleep 60', wait_ms: 999_999 });
const cappedMs = Date.now() - t1;
check('wait_ms is clamped below the 60s MCPL cap', cappedMs < 55_000, `${cappedMs}ms`);
check('clamped call still returns a job handle', /still running/.test(textOf(capped)), textOf(capped));

// 7. the command's own budget still kills it
const timed = await bash({ command: 'sleep 30', timeout_ms: 1500, wait_ms: 6000 });
const timedText = textOf(timed);
check('timeout_ms kills the job', /timed out after 1500 ms/.test(timedText), timedText);
check('timed-out job is an error', timed.result.isError === true, JSON.stringify(timed.result.isError));

// 8. a killed process group takes its children with it
const tree = await bash({ command: 'sleep 40 & sleep 41 & wait', timeout_ms: 1200, wait_ms: 6000 });
check('process-group kill settles the job', /timed out/.test(textOf(tree)), textOf(tree));

// 9. argument validation
const neither = await bash({});
check('neither command nor job_id is rejected', neither.result.isError === true, textOf(neither));
const bogus = await bash({ job_id: 'j99999' });
check('unknown job id is rejected clearly', /no such job/.test(textOf(bogus)), textOf(bogus));
const both = await bash({ command: 'echo x', job_id: jobId });
check('command + job_id together is rejected', /not both/.test(textOf(both)), textOf(both));

// 10. interleaved stdout/stderr both land in the log
const mixed = await bash({ command: 'echo out; echo err >&2', wait_ms: 5000 });
check('stderr is captured alongside stdout', /out/.test(textOf(mixed)) && /err/.test(textOf(mixed)), textOf(mixed));

// 11. large output is bounded, not unbounded
const big = await bash({ command: 'yes abcdefghij | head -c 900000', wait_ms: 20_000 });
const bigText = textOf(big);
check('large output is truncated in the middle', /characters truncated/.test(bigText), `len=${bigText.length}`);
check('truncated payload stays bounded', bigText.length < 500_000, `len=${bigText.length}`);

console.log(`\n${failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`}`);
srv.stdin.end();
process.exit(failures === 0 ? 0 : 1);
