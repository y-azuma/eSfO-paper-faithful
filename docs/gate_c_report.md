# Gate C report: direct orbit factor and gauge-aware recovery

Status: **local implementation and synthetic verification complete; independent
cross-review pending**.

Gate C implements only the paper's orbit-constrained reprojection stage under
the frozen, result-independent conventions in
[`geometry_conventions.md`](geometry_conventions.md). No TOPSPIN scene or
public-data performance result was used to select a factor, coordinate,
tolerance, solver, or observation.

## Implemented reference path

- one two-pixel direct `gtsam.CustomFactor` per frozen D-009 Eq. (4)
  observation;
- variable/key order `point, c, r, f, R0, n_raw, u_raw`;
- additive Euclidean coordinates for points, `c`, raw `n`, and raw `u`;
- unconstrained real `r` and `f`, including evaluation and reporting of
  negative values;
- GTSAM `Rot3` right-local coordinates for `R0`;
- in-factor normalization of `n` and plane projection plus normalization of
  `u`, followed by `v=n cross u`;
- the invariant-derived orbit pose frozen before implementation;
- predicted-minus-observed fixed-PINHOLE residuals, isotropic `sigma=1 px`,
  squared Gaussian error, no robust loss, and no prior;
- analytic Jacobians for all seven variable blocks;
- direct GTSAM `4.2` default Levenberg--Marquardt, with no exception recovery,
  solver fallback, or retry.

The observation builder keeps every original fixed-undistorted Eq. (4)
observation for each bijectively initialized source track. This includes
unregistered logical windows and observations absent from the selected
COLMAP point track. It does not fabricate empty track/window pairs or select
factors by depth, residual, ground truth, or outcome. Split, merged, and
uninitialized track diagnostics are exact and may overlap.

The extracted COLMAP model also preserves the complete selected-model
`points3D` observation map as an immutable value. Consequently, D-009's
retained/absent-in-COLMAP diagnostics can be reproduced directly from the
stored model object without reparsing a mutable text export.

## Jacobian and gauge verification

The central-difference step and tolerance were frozen before public-data use:

```text
h_additive = eps_float64^(1/3) * max(1,abs(value))
h_Rot3     = eps_float64^(1/3) radians
atol       = 2e-7
rtol       = 2e-7
```

All seven analytic blocks agree with these central differences on the fixed
nondegenerate factor fixture. An independent random stress audit also found
the same derivatives, with the expected larger absolute finite-difference
error only in deliberately ill-scaled long-time/high-projection cases.

For 12 points observed at 9 times, the stacked residual Jacobian has shape
`216 x 50`. Its float64 rank is `40`, so its nullity is exactly `10`. The ten
explicitly constructed directions are:

- global translation: 3;
- global rotation: 3;
- monocular scale: 1;
- raw-normal magnitude: 1;
- raw-`u` magnitude: 1;
- raw-`u` component parallel to the raw normal: 1.

The maximum absolute entry of `J G` for those directions is below `2e-12`,
and the first identifiable singular value is more than `1e10` times the
largest numerical null singular value. This confirms the expected gauge; it
does not make the raw parameter vector identifiable.

## Gauge-aware synthetic recovery

A deterministic, noise-free 12-point by 9-view problem contains 108 direct
factors. The initialization perturbs every point and every global variable
block without adding a prior. Default GTSAM 4.2 Levenberg--Marquardt reduces
the graph error from approximately `203.28` to below `1e-20`.

Recovered points and sampled orbit centres are aligned to their generating
geometry by one proper Sim(3), then checked jointly. On the local reference
environment the aligned coordinates, radius, frequency, axis, in-plane
direction, and `R0` agree to approximately `1e-12`--`1e-11`. These are
synthetic self-consistency checks, not claims about TOPSPIN or the authors'
hidden outputs.

## Test and environment result

On 2026-11-24, Python 3.10.6 with `numpy==1.26.4`, `scipy==1.13.0`, and
`gtsam==4.2` reports:

- 13 pure orbit-factor tests;
- 12 GTSAM adapter, observation-scope, recovery, and gauge tests;
- 24 Gate B geometry tests after the temporal-sign B1 correction;
- `135 passed` for the complete synthetic suite;
- statement coverage `85%`;
- four existing warnings inside frozen `hdbscan==0.8.33` caused by a later
  NumPy deprecation;
- successful `compileall`, `pip check`, and `git diff --check`.

The CI workflow installs the pinned GTSAM extra and executes the same
synthetic suite on Python 3.10. Commit
`1bccdd693dcdf229a0db03825ff04b03e8b3a505` passed GitHub Actions run
`45368753199`, job `138357216972`, on Ubuntu 24.04 with Python 3.10.14. An
earlier runs exposed only over-tight exact-invariance test tolerances: the
largest observed Ubuntu round-off was `1.14e-13`, while individual assertions
allowed `5e-14` or `6e-14`. Those test-only tolerances were normalized to
`2e-13`; the factor, Jacobian, solver, and recovery implementation were
unchanged.

The exact same commit was then checked out independently on the RTX 4070
workstation at
`/home/azumayudai/Desktop/ws_snn/eSfO-paper-faithful`. Ubuntu 24.04.4,
Python 3.10.14, NumPy 1.26.4, SciPy 1.13.0, and GTSAM 4.2 again reported
`135 passed` (four known HDBSCAN warnings), plus successful `compileall`,
`pip check`, and `git diff --check`. This synthetic suite is CPU-only; the
presence of the RTX 4070 is not evidence of GPU validation or performance.

These cross-platform results still require independent Claude Code review
before Gate C is called mutually approved.

## Evidence boundary and next gate

Passing Gate C establishes that the frozen factor equations, derivatives,
GTSAM wiring, observation membership, and gauge-aware comparison are
internally consistent. It does not establish that the unpublished author code
used these missing conventions or that the method reproduces paper behavior
on TOPSPIN.

After cross-review, Gate D may integrate the complete synthetic pipeline,
stage-boundary provenance and resume logic, fail-closed unavailable metrics,
and the preregistered scale/resource measurements. Public TOPSPIN execution
remains prohibited until Gates A--D and both reviewers approve it.
