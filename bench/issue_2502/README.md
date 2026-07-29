# #2502 measurement and figure scripts

Each script here answers one question and prints the control alongside the
result. They are recorded because several of today's retractions came from
figures and statistics that looked right and were not.

## Interpretability

- `atom326_occupancy.py` — phase occupancy of a periodic atom. Atom 326 fills
  **10 of 36 bins (28% of its circle)**, 89% of its mass inside 40 degrees,
  largest empty gap **272 degrees**. The closed topology is refuted by the
  same largest-gap test `convert_underoccupied_loops` uses, at −30546 against
  a −19.96 threshold. Min/max of the routed coordinate is a RANGE, not
  coverage — reporting `[0,1]` as "covers the period" was wrong.
- `atom326_tokens.py` — what actually ROUTES to an atom, against a corpus
  baseline. Atom 326: **40.8% punctuation/markup vs 22.4% baseline**.
- `atom326_context.py` — script of the surrounding context. Atom 326 and the
  corpus are both **100% Latin-dominant**, so multilinguality is not testable
  on this corpus in either direction.
- `coord_meaning.py` — what varies ALONG the coordinate. Atom 326 runs
  function words → punctuation → **hyphen 40%** → digits/commas → function
  words: a punctuation-density axis, symmetric at both ends.
- `why_cjk.py` — why lensed labels come out CJK. Qwen's vocabulary is
  **26.5% CJK**, and RANDOM directions through `lm_head` produce the same
  tokens the figures labelled (`traduz`, `Topl`, `tasa`, `ɔn`). Logit-lens
  labels on a weakly-constrained direction carry no information about the
  atom.

## Reconstruction

- `random_floor.py` — the floor nobody had measured. Sampling random DATA
  ROWS as atoms scores **0.8120** held-out at K=11010; ours fitted 0.8538.
  A floor must be the cheapest thing that could REPLACE the method, not the
  cheapest thing imaginable — isotropic Gaussian (0.6080) is the wrong
  control and flatters the fitter by 0.20.
- `gen_gap.py` — train vs held-out for both dictionaries. Ours 0.8610/0.8538,
  steelman 0.8940/0.8831 at half the parameters: we **underfit**, and their
  generalisation gap is larger than ours.
- `split_atoms.py` — is the affine form the tax? Splitting 11010 affine atoms
  into 22020 pure directions at identical budget LOSES 0.0017. `b0` alone
  (0.8362) beats `b1` alone (0.8186): `b0` is the direction, `b1` the
  correction.
- `reclaim_headroom.py` — dead-atom reclamation priced on a frozen
  dictionary: **+0.0024** with no refit. In the fitter it measured +0.0001.
  A proxy earns trust on the axis it was checked on and no further.

## Figures

- `figure_labels.py` — CJK rendering and invisible-token display. matplotlib
  has no per-glyph fallback here, so a label containing CJK is drawn entirely
  in Noto Sans CJK; invisible tokens print as `[zwsp]`/`[sp]` instead of
  `repr()`'s escape codes.
- `sphere_atlas_bilinear.py` — token placement on a decoded surface. The
  previous version truncated fractional lattice indices, snapping every token
  to a mesh vertex; the most-used atom put 32,001 tokens on 1049 vertices and
  the resulting grid of clusters was pure quantisation.
