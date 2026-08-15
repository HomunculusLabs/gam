<!--
This page renders the repository-root CHANGELOG-ARCHIVE.md verbatim — the
already-released half of the changelog, split off when CHANGELOG.md crossed the
10,000-line tracked-file limit (issue #780). CHANGELOG.md remains the single
source of truth for unreleased work; this file is the continuation, not a
duplicate. The include is resolved by pymdownx.snippets with check_paths on, so
`mkdocs build --strict` fails if CHANGELOG-ARCHIVE.md is moved or removed.
-->

--8<-- "CHANGELOG-ARCHIVE.md"
