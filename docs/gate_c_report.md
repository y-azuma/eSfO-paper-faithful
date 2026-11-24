# Gate C report: direct orbit factor and gauge-aware recovery

Status: **final Gate C approval recorded. Factor mathematics, Jacobians, GTSAM
binding, provenance, gauge, Sim(3) recovery, and C1 cheirality handling are
independently approved; Gate D is authorized and in progress**.

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

For a non-positive camera depth, the factor follows pinned GTSAM 4.2
`GenericProjectionFactor` with `throwCheirality=false`: it remains in the
graph, returns the constant residual `(2*fx,2*fx)`, and returns zero for all
Jacobian blocks. This is ordinary factor evaluation, not a solver retry or
factor-pruning recovery path.

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

## Cheirality behavior and reporting boundary

Raising an exception inside a Python `CustomFactor` aborts the complete GTSAM
LM call, including when only a rejected trial step crosses depth zero. The C1
correction therefore mirrors the
[primary GTSAM 4.2 implementation](https://github.com/borglab/gtsam/blob/4f66a491ffc83cf092d0d818b11dc35135521612/gtsam/slam/ProjectionFactor.h#L153-L163).
A focused test drives a real LM
trial through non-positive depth and verifies that optimization is not
interrupted.

An auditable binding associates every graph factor with its keys and
measurement. The diagnostic optimizer records the number of non-positive
depth factors at the initial and final values; binding count, order, and keys
must match the graph. The integrated Gate D reference will fail closed if the
final count is nonzero. In that case reprojection metrics are unavailable,
and the constant penalty's pseudo-pixel is not included in Fig. 10-style
pixel-residual outputs.

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
- 14 GTSAM adapter, cheirality, observation-scope, recovery, and gauge tests;
- 26 Gate B geometry tests after the B1/B1-prime corrections;
- `139 passed` for the complete synthetic suite;
- statement coverage `84%`;
- four existing warnings inside frozen `hdbscan==0.8.33` caused by a later
  NumPy deprecation;
- successful `compileall`, `pip check`, and `git diff --check`.

The corrected implementation commit
`6640dad46f51825fa311b72d44aceccc4bf0e673` passed GitHub Actions run
`45372999990`, job `138371780290`, on Ubuntu 24.04 / Python 3.10.14 with all
`139` tests. This is the authoritative B1-prime/C1 evidence submitted for and
approved in the focused re-review. Earlier commit `1bccdd6` and run
`45368753199` establish only the pre-correction core. A previous CPU-only rerun
on the 4070
workstation is historical corroboration and is not relied on for the corrected
result.

Claude Code independently re-derived all seven Jacobian blocks (maximum
relative discrepancy `6.5e-10`), reproduced nullity ten, and recovered a
perturbed synthetic problem from graph error `6.9e4` to `1.2e-24` with
joint-Sim(3) RMSE `2.8e-16`. It approved D-009 and the Gate C core. Its C1
finding and B1-prime Gate B finding were narrow post-review corrections, not
a withdrawal of those approved results. In the focused re-review of final
commit `9313c6da48399432e06cc080f07f832574176aae`, Claude Code confirmed the
exact GTSAM 4.2 `(2*fx,2*fx)` residual, zero Jacobians, and
`throwCheirality=false`. In 40 deterministic LM stress perturbations none
aborted; 19/40 ended with a behind-camera factor and correctly retained the
fail-closed final-depth failure. C1 and Gate C received final approval.

## Evidence boundary and next gate

Passing Gate C establishes that the frozen factor equations, derivatives,
GTSAM wiring, observation membership, and gauge-aware comparison are
internally consistent. It does not establish that the unpublished author code
used these missing conventions or that the method reproduces paper behavior
on TOPSPIN.

Gate D's stage-boundary provenance, resume, fail-closed metric, and resource
instrumentation are authorized and in progress. Integrated synthetic evidence
must use the focused-reapproved B1-prime/C1 commit. Public TOPSPIN execution
remains prohibited until Gate D is complete and both reviewers approve it.
