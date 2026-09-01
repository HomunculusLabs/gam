# GAM Codebase City

A procedural 3D map of the GAM repository. The city is generated from source
files, crate structure, GitHub issues and comments, commits, test failures,
GitHub Actions runs, measurement artifacts, and historical Git trees.

## Refresh the city

The collector reads the repository worktree and authenticated GitHub data, then
rebuilds the static city snapshot:

```bash
python3 scripts/collect_city_data.py
```

## Run

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
npm run build
npm test
```

The timeline is sourced from real Git trees. District placement is a weighted
graph embedding built from dependency edges, commit co-change, and shared issue
references. Measurement lag is modeled per artifact and affects only the
districts that artifact measures.
