# Repository agent instructions

- If pushing fails because the wrong GitHub identity is active, use `gh auth switch`.
- Do not restore files from Git or use Git to edit or revert files without approval.
- You may push without approval.
- Do not create branches or worktrees. Work in the shared checkout while preserving unrelated
  changes.
- Backwards compatibility is not a goal. Remove dead or unused code after deciding whether it
  should instead be wired into the product. Do not add tests for dead code; `#[cfg(test)]` is for
  test code only.
- Do not stop because unrelated or unexpected working-tree changes are present. Preserve them and
  continue around them.
- Avoid fallbacks and environment variables when possible.
- Never kill Claude processes. Never use Silico.
- GitHub Actions runs are shared evidence, not disposable task processes. Never cancel a
  `push`- or `schedule`-triggered run. Never cancel any run that was not created by the current
  task. A task that dispatches a workflow must record the run ID returned by GitHub; it may cancel
  only that exact `workflow_dispatch` run, and only when that run is obsolete, dead, or hung.
  Never bulk-cancel runs from a listing, and never infer ownership from a workflow name, ref, SHA,
  age, or apparent inactivity. Workflow-declared concurrency may supersede runs according to its
  checked-in policy; do not recreate that policy with `gh run cancel` or the Actions cancel API.
  If ownership is not proven, leave the run alone and report it.
