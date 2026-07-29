# Issue #2502 measurement harness (Qwen3.5-4B-Base layer-16 manifold SAE campaign)

Python side of the overcomplete manifold-SAE campaign. The Rust side is
`crates/gam-sae/examples/support_fit_dump.rs` (fit/dump harness with the
opt-in tokens: `fista`, `unroll`, `price`, `vark`).

- `harvest.py` — residual-stream harvest (wikitext-103, document split,
  train-only PCA-128 chart). Parameterized for row count; reuse the frozen
  chart (`lift.npy`, `c0.npy`) for comparability across harvests.
- `splice_paired.py` — causal delta-CE: replace the in-chart component at
  held-out positions, teacher-forced next-token CE vs clean; identity
  round-trip is an exact-zero control, mean-floor is the destroyed-info
  control. OMP top-k scoring for flat-SAE baselines.
- `surrogate_gen.py` — surrogate reading: sampled continuations after the
  model reads a prefix through each dictionary (KV-cache-baked splice).
- `steer_grid.py` / `steer_text.py` — steering selectivity (atom-chord vs
  equal-norm random vs best-aligned flat feature) and steered generations.
- `kappa_census.py` — shattered-manifold witness statistics (kappa = m4/m2^2
  joint amplitude law + phase spread) over co-active pairs of a flat SAE,
  with a covariance-matched Gaussian null through the same encoder
  (run with argument `null`). NOTE: pair screen only; a nonnegative-gate
  dictionary can shatter a circle into up to FOUR rectified half-atoms, so
  an antipodal/quad-aware pass is the planned upgrade.
- `fits_gallery.py` / `trio_figs.py` / `night_figs.py` — figures: fitted
  curves against their partial-residual data, loop occupancy walks, ladder
  dynamics, EV-vs-causal frontier.

Campaign results live on GitHub issue #2502.
