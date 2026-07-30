# `bench/gha_results/` — the newest suite-level numbers, tracked in git

Each subdirectory holds the **most recent uncompressed suite-level results**
from a publishing GitHub Actions workflow: accuracy and perf numbers, metric
tables, and the mature reference tools' own figures where a workflow measures
against them. Focused, one-off proof runs keep their logs in workflow artifacts
instead of committing them here.

```
bench/gha_results/
  reference-quality/    quality_results.tsv, quality_results.jsonl, quality-aggregate.txt
  benchmark/            results.nightly.json
  large-scale/          per-method result JSON + the aggregate summary
  rust-test-suite/      MASTER_FAILURES.md
  python-contracts/     JUnit inventory
  fuzz/                 fuzz findings
```

Every directory also carries `_run.json`, naming the run that produced it: run
id and URL, commit, event, job status, and completion time.

## Why in git rather than only in the artifact store

Artifacts expire and are per-run, so "did this number move?" was not answerable
without downloading two runs and diffing by hand. Here `git log -p
bench/gha_results/<workflow>/` is the history of that measurement, and a regular
`git diff` shows what a change did to the numbers.

## Contract

Written by `.github/actions/publish-gha-results`, invoked with `if: always()`.

- **Failures and timeouts publish too.** A failed run's numbers are usually the
  interesting ones, and `_run.json` records `job_status` so a partial set is
  never mistaken for a clean one.
- **No results means no commit.** If a run produced nothing, the previous good
  results stay exactly as they are. An empty or missing set never overwrites
  them, so this directory is never blanked by a broken run.
- **Uncompressed only.** `.json .jsonl .ndjson .tsv .csv .txt .md .log .out
  .yaml .yml`, subject to a per-file size cap. Archives, wheels, binaries and
  images stay in the artifact store — this directory is for numbers you can read
  in a diff.
- **Each workflow owns its own subdirectory** and replaces it wholesale, so a
  file a workflow stops producing disappears instead of lingering as a stale
  number beside fresh ones. Concurrent publishers rebase and retry; they cannot
  conflict, because they never write outside their own subdirectory.

## Do not hand-edit

These files are overwritten by the next run of their workflow. Commits land as
`results(<name>): run <id> (<status>) [skip ci]`.

The `[skip ci]` is load-bearing — these commits push to `main`, and without it
the push-triggered workflows would re-run and publish again indefinitely.
`cross-check.yml` and `wheel-nightly.yml` additionally carry a `paths-ignore`
for this directory.
