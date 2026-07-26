#!/usr/bin/env node
// Gate the recipe's context arithmetic.
//
// The autobiographical strategy pins `recentWindowTokens` into the prompt
// verbatim: the adaptive picker may fold the middle of the history, but it can
// never fold the recent window or the system prompt. So once
//
//     systemPrompt + recentWindowTokens  ~=  contextBudgetTokens - RESERVE
//
// there is no amount of folding that can fit, every attempt fails identically,
// and the agent is wedged for the rest of the run. That is not hypothetical:
// with recentWindowTokens = 120000 against contextBudgetTokens = 180000, run
// 30189939720 died with
//
//   Adaptive picker exhausted but 165494 tokens still exceed hard budget 163616
//   (head=2542, tail=119978, middle=42974 across 137 chunks)
//
// — over by 1878 tokens, with the pinned window alone eating 73% of the budget.
//
// This check requires the pinned window to leave at least as much foldable room
// as it takes for itself, so the picker always has somewhere to work.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The framework reserves its max output allowance off the top of the budget;
// 180000 - 163616 = 16384 exactly, as observed in the failure above.
const OUTPUT_RESERVE_TOKENS = 16_384;
const here = dirname(fileURLToPath(import.meta.url));
const path = process.env.RECIPE_PATH || join(here, 'recipe.json');

let recipe;
try {
  recipe = JSON.parse(readFileSync(path, 'utf8'));
} catch (error) {
  console.error(`FAIL  ${path} is not valid JSON: ${error.message}`);
  process.exit(1);
}

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : ` — ${detail}`}`);
  if (!cond) failures++;
};

const agent = recipe.agent ?? {};
const strategy = agent.strategy ?? {};
const budget = agent.contextBudgetTokens;
const recent = strategy.recentWindowTokens;

check('recipe declares contextBudgetTokens', Number.isFinite(budget), String(budget));
check('recipe declares strategy.recentWindowTokens', Number.isFinite(recent), String(recent));

if (Number.isFinite(budget) && Number.isFinite(recent)) {
  const hard = budget - OUTPUT_RESERVE_TOKENS;
  const foldable = hard - recent;
  check(
    `pinned recent window (${recent}) leaves at least as much foldable room as it takes ` +
      `(hard budget ${hard}, foldable ${foldable})`,
    recent <= foldable,
    `recentWindowTokens must be <= ${Math.floor(hard / 2)} for contextBudgetTokens=${budget}`,
  );
  check(
    'maxStreamTokens fits inside the hard budget',
    !Number.isFinite(agent.maxStreamTokens) || agent.maxStreamTokens <= hard,
    `maxStreamTokens=${agent.maxStreamTokens} vs hard budget ${hard}`,
  );
}

// The system prompt must not tell the agent to do the one thing the transport
// forbids: a per-call `timeout_ms` cannot outlive the 60 s MCPL request cap.
const prompt = String(agent.systemPrompt ?? '');
check(
  'system prompt teaches job polling rather than raising a per-call timeout',
  prompt.includes('job_id'),
  'the bash tool answers long commands with a job handle; the prompt must say so',
);
check(
  'system prompt does not tell the agent to sleep-poll',
  !/raise `timeout_ms` rather than/.test(prompt),
  'that advice cannot work: the transport abandons any call at 60 s',
);

console.log(`\n${failures === 0 ? 'RECIPE OK' : `${failures} RECIPE CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
