#!/usr/bin/env node
// Union-merge lesson shards from parallel agent runs into one global file.
//
// Chronicle history cannot be merged — two runs that diverge from a common
// ancestor have two genuinely different narratives, and picking one is the
// honest outcome. Lessons are different: they are id-addressed facts, so the
// union of every run's lessons is well-defined and loses nothing.
//
// That split is what makes parallelism survivable. Episodic memory forks per
// lineage; semantic memory converges across all of them, so a thing one agent
// learned the hard way is not stranded in a lineage nobody inherits.
//
// Conflict rule is the lessons module's own (lessons-module.ts:427): same id,
// newest `updated` wins. Reimplementing it differently here would make the
// merged file disagree with what the module would have produced itself.
//
// Usage: merge-lessons.mjs <out.json> <shard.json>...

import { readFileSync, writeFileSync } from 'node:fs';

const [out, ...shards] = process.argv.slice(2);
if (!out) {
  console.error('usage: merge-lessons.mjs <out.json> <shard.json>...');
  process.exit(2);
}

const byId = new Map();
let read = 0, seen = 0, superseded = 0;

const stamp = (l) => Date.parse(l?.updated ?? l?.created ?? '') || 0;

for (const path of shards) {
  let doc;
  try {
    doc = JSON.parse(readFileSync(path, 'utf-8'));
  } catch (err) {
    console.error(`  skip ${path}: ${err.message}`);
    continue;
  }
  const lessons = Array.isArray(doc?.lessons) ? doc.lessons : [];
  read += 1;
  for (const l of lessons) {
    if (!l || typeof l.id !== 'string') continue;
    seen += 1;
    const prev = byId.get(l.id);
    if (!prev) { byId.set(l.id, l); continue; }
    superseded += 1;
    if (stamp(l) > stamp(prev)) byId.set(l.id, l);
  }
}

// Deterministic order so the written file is byte-stable when nothing changed —
// an unstable file would make every run look like it modified the shared state.
const merged = [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
writeFileSync(out, `${JSON.stringify({ lessons: merged }, null, 2)}\n`);

console.error(
  `merged ${read}/${shards.length} shard(s): ${seen} lesson rows -> ${merged.length} unique` +
  (superseded ? ` (${superseded} id collision(s) resolved by newest 'updated')` : ''),
);
