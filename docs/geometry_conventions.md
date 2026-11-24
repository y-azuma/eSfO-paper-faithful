# Geometry conventions for the paper-faithful reference path

Status: **Gate A approved; Gate C core approved; B1-prime and C1 corrections
implemented and awaiting focused re-review**.

This document freezes the geometric conventions needed to implement eSfO.
It separates statements printed in the paper from conventions that are needed
because the printed equations are incomplete or mutually ambiguous.  None of
the inferred choices below was selected using TOPSPIN performance.

The notation `A <- B` means that a transform maps coordinates expressed in
frame `B` into coordinates expressed in frame `A`.  Column vectors and active
rotations are used throughout.

## Evidence labels

- **Paper-reported**: stated in the paper text, figures, or displayed equations.
- **Derived invariant**: algebraically required for the paper's stated camera
  centre or relative motion, but not written as an implementation rule.
- **Frozen interpretation**: a deterministic convention needed for an
  underspecified or inconsistent part of the paper.
- **Sensitivity only**: a plausible alternative excluded from the reference
  result unless it is reported separately.

## Plane and circle initialization: Eqs. (6)--(8)

### What the paper reports

The paper defines the mean camera centre

```text
t_c = mean_k(C_k)
```

and the centred camera-centre matrix whose rows are `C_k - t_c`.  It then:

1. regresses the third coordinate on the first two coordinates and a constant;
2. says that the normalized regression result gives the plane normal;
3. rotates the camera centres from the estimated plane to the `xy` plane;
4. fits the implicit circle in Eqs. (7)--(8); and
5. derives the radius, circle centre, and initial in-plane direction `u`.

Those high-level operations and the least-squares systems are
**Paper-reported**.  The following details are not implementable literally:

- If `z = a x + b y + c_0`, the coefficient vector `(a,b,c_0)` is not a
  plane normal.  The corresponding implicit plane has normal `(a,b,-1)`.
- The printed projected point uses an uncentred camera centre, whereas adding
  `t_c` back to the fitted circle centre is consistent only if the projected
  points were centred first.
- The starred second component in the printed radius expression is not a
  defined operation and is treated as a typesetting error for a square.
- The text does not specify the sign of the normal or the degenerate cases of
  the Rodrigues rotation.
- Section 3 calls `n` a unit vector and says that `u` lies in the plane and is
  orthogonal to `n`; it does **not** explicitly call `u` a unit vector.  Unit
  length is nevertheless required for `r` in Eq. (1) to remain the radius and
  is also suggested by the hatted `u` in Fig. 4.

### Frozen reference interpretation

Let `C_k` be the COLMAP camera centres in temporal order and let
`X_k = C_k - t_c`.

1. Solve the paper's ordinary least-squares system

   ```text
   [X_x  X_y  1] [a, b, c_0]^T = X_z.
   ```

   Form `n_raw = (a,b,-1)` and normalize it.  Record the design-matrix
   condition number.  A rank-deficient system or a non-finite solution is a
   reference-path failure, not permission to substitute a different fit.

2. Before choosing the sign of `n`, construct a provisional active alignment
   from the unsigned normal to `z`, project the **centred** points, and fit the
   printed implicit circle:

   ```text
   y_k = first_two_components(A_unsigned (C_k - t_c))
   [y_x  y_y  1] Theta_c = y_x^2 + y_y^2
   centre_2d = (Theta_c[0]/2, Theta_c[1]/2)
   c^w = A_unsigned^T (centre_2d[0], centre_2d[1], 0)^T + t_c.
   ```

   Reflecting the unsigned plane normal changes only the in-plane orthogonal
   coordinates; transforming the fitted centre back to world coordinates
   gives the same `c^w`. Thus the circle centre is available without choosing
   the normal sign. Rank and radius-squared failures are checked here rather
   than replaced by the sample centroid.

3. Fix the sign of `n` from temporal motion **about the fitted circle
   centre**. For

   ```text
   q_k = (I - n n^T) (C_k - c^w)
   ```

   compute

   ```text
   s = sum_k n^T (q_k x q_{k+1}).
   ```

   Flip `n` if `s < 0`, so increasing time is positive rotation about `n`.
   This agrees with Eq. (1), `v = n x u`, and the frozen positive-frequency
   convention.  If `s` is numerically zero, the sign is unobservable and the
   initialization fails explicitly.

   This sampled-chord sign rule has a Nyquist precondition that the paper does
   not state.  Once the positive FFT frequency is available, form
   `Delta theta_k = 2*pi*f*(t_{k+1}-t_k)` for every adjacent **registered**
   pair and compute the score and its tolerance using only pairs with
   `Delta theta_k < pi`.  A larger gap can make that pair's forward motion
   indistinguishable from the opposite direction, so it is excluded rather
   than allowed to vote on the sign.  The reference initializer fails with
   `temporal_sign_aliased` only if no non-aliased pair remains; it fails with
   `temporal_sign_unobservable` if the retained score is numerically zero. It
   records the total, retained, and excluded pair counts, maximum phase gap,
   and the `c^w` used for the score. It does not choose pairs from
   reconstruction quality. Referencing the chord to the sample centroid is
   explicitly excluded: with sparse registered views the centroid can lie on
   the opposite side of an otherwise identifiable chord.

4. Construct `A = Rod(n,z)`, the active rotation mapping the now-signed `n` to
   `z=(0,0,1)`.  For neither-parallel case use

   ```text
   axis  = normalize(n x z)
   angle = atan2(||n x z||, n^T z).
   ```

   Define `eps_parallel = 32 * eps_float64`.  Treat `||n x z|| <=
   eps_parallel` as the parallel/antiparallel branch.  If `n^T z >= 0`, use
   the identity.  Otherwise choose the least-aligned Cartesian basis vector
   `e` (ties resolve `x`, then `y`, then `z`), form
   `axis=normalize(e-(e^T n)n)`, and use a pi rotation about that axis.  The
   projection makes the axis exactly orthogonal to `n`; the tolerance and tie
   rule are **Frozen interpretations**.

   Re-express the circle with this signed alignment using

   ```text
   y_k = first_two_components(A (C_k - t_c)).
   ```

   Fit the paper's implicit circle

   ```text
   [y_x  y_y  1] Theta_c = y_x^2 + y_y^2.
   ```

   Then

   ```text
   centre_2d = (Theta_c[0]/2, Theta_c[1]/2)
   r^2       = Theta_c[2] + ||centre_2d||^2
   c^w       = A^T (centre_2d[0], centre_2d[1], 0)^T + t_c.
   ```

   A negative `r^2` beyond a documented floating-point tolerance is a
   reference-path failure. The resulting world `c^w` must agree, to the
   documented float64 scale tolerance, with the sign-reference centre from
   step 2.

5. Let `k*` be the earliest-time registered view in the largest retained
   COLMAP model.  Form the first radial direction from that same centre,
   project it onto the fitted plane, and normalize:

   ```text
   u_raw = (I - n n^T) (C_{k*} - c^w)
   u     = normalize(u_raw)
   v     = normalize(n x u).
   ```

   Projecting before normalization is a **Frozen interpretation** required
   because the COLMAP centre need not lie exactly on the fitted plane.  A
   near-zero `u_raw` is a reported degeneracy.

An SVD/total-least-squares plane fit is geometrically attractive but is not
the regression displayed in Eq. (6); it is therefore **Sensitivity only**.

### Frozen numerical-degeneracy conventions

The following float64 rules are fixed before any public-data execution. They
make failure decisions reproducible; they are not fitted method parameters.

- Every displayed least-squares system uses explicit
  `rcond = eps_float64 * max(number_of_rows, number_of_columns)` and must have
  full column rank.
- The temporal sign is unobservable when
  `abs(s) <= 64*eps*sum_k(||q_k|| ||q_{k+1}||)`.
- A fitted `r^2` in the interval `[-tol_r2,0)` is clamped to zero only for
  arithmetic roundoff, where
  `tol_r2 = 64*eps*max(max_k ||y_k||^2, ||centre_2d||^2, |Theta_c[2]|)`.
  A zero radius still fails because its radial direction is unobservable.
- The earliest-view radial direction is unobservable when
  `||u_raw|| <= 64*eps*max(r,max_k ||C_k-c||)`.
- Input and constructed rotation matrices are accepted only when both the
  infinity-norm orthogonality error and determinant error are at most
  `256*eps`. Inputs are rejected rather than individually projected to
  `SO(3)`.
- Registered-view times must be strictly increasing. Duplicate timestamps are
  a reference-path failure rather than an unstated view-ID tie rule.
- The temporal-sign score and tolerance use only adjacent registered-view
  pairs whose positive-frequency phase gap is strictly less than `pi`.
  Excluded-pair count and maximum phase gap are recorded.  If no such pair
  remains, temporal direction is rejected as aliased. This is an
  identifiability guard on the sampled-chord rule, not a fitted threshold.
- Every temporal chord is referenced to the fitted circle centre `c^w`, not
  the sample centroid `t_c`. Circle fitting precedes sign selection; its world
  centre is invariant to the provisional plane-normal sign.
- The chordal `R_0` mean is unobservable when its smallest singular value is
  no larger than `eps*3*sigma_max`. If the determinant correction is negative,
  it is also unobservable when the two smallest singular values differ by no
  more than that tolerance, because there is no unique singular direction to
  flip. No arbitrary SVD completion is accepted.

The paper's ordinary `z`-on-`x,y` regression is not equivariant under every
noisy global rotation and cannot represent a vertical plane. Exact
equivariance is therefore required only for noise-free transformed fixtures
whose displayed regression remains full rank. A transform producing a
vertical/rank-deficient regression must fail; the reference path must not
silently replace Eq. (6) with total least squares.

## Orbit pose convention: Eqs. (1) and (11)--(14)

### Why the printed composition is ambiguous

Equation (11) says that the product in Eqs. (12)--(14) maps a world point into
the camera at time `t_k`.  The paper does not explicitly define the direction
of every superscript/subscript transform.  With the usual world-to-camera
homogeneous-transform convention, the translation printed in Eq. (12),
combined with the two zero-translation rotations in Eqs. (13)--(14), does not
recover Eq. (1)'s camera centre: a world-to-camera matrix `[R,t]` has centre
`-R^T t`, not `t`.  The initialization of `R_0` is also not stated.

Consequently, the displayed Eqs. (11)--(14) are recorded as **Ambiguous**, not
as an exact executable composition.  The reference implementation is derived
from the following three result-independent invariants:

1. the camera centre must equal Eq. (1);
2. camera-to-world relative orientation over `Delta t` must rotate about `n`
   by `2 pi f Delta t`; and
3. a constant choice of orbit-frame or camera-frame axes must be absorbable in
   the constant rotation `R_0`.

### Frozen transform directions and formula

The paper does not state an absolute phase origin.  Freeze `k*` as the
earliest-time registered view in the largest retained COLMAP model and set
`t_ref=t_{k*}` in seconds.  The factor graph treats `t_ref` as a constant, not
an optimized variable.  This uses the same centre that defines `u`, so zero
phase has its plane-projected radial direction; for a noise-free centre on the
fitted circle, the initialized orbit passes `C_{k*}` exactly.  Reading the
displayed window indices literally could instead assume that window 1 is
registered and `t_ref=0`; when the first retained view is later, that
paper-unstated assumption introduces phase `2 pi f t_{k*}`.  It is recorded
as an alternative interpretation, not the reference path.

Define the right-handed orbit basis and phase rotation

```text
B        = [u  v  n] in SO(3), where v = n x u
tau_k    = t_k - t_ref, in seconds
theta_k  = 2 pi f tau_k
S(theta) = Rot_z(theta) in the orbit basis.
```

The camera centre is the **Paper-reported** Eq. (1), written as

```text
C_w(t_k) = c^w + B [r cos(theta_k), r sin(theta_k), 0]^T.
```

Freeze `R_0` as the constant rotation from the phase-zero orbit frame into the
camera frame.  The reference world-to-camera pose is

```text
R_cw(t_k) = R_0 S(-theta_k) B^T
t_cw(t_k) = -R_cw(t_k) C_w(t_k)
T_{c(t_k) <- w} = [R_cw(t_k), t_cw(t_k); 0, 1].
```

Equivalently, the camera-to-world orientation is

```text
R_wc(t_k) = B S(theta_k) R_0^T.
```

This formula is a **Derived invariant / Frozen interpretation**.  It is not
claimed to be the literal matrix product printed in Eqs. (11)--(14).  It
ensures

```text
-R_cw(t_k)^T t_cw(t_k) = C_w(t_k)
R_wc(t_j) R_wc(t_i)^T = Rot(n, theta_j - theta_i).
```

Any constant alternative assignment of orbit-frame axes is acceptable only
if it is shown algebraically to be absorbed by `R_0` and passes the same
projection-equivalence tests.  It must not be chosen from TOPSPIN accuracy.

### Frozen initialization of `R_0`

For each registered COLMAP world-to-camera rotation `Rhat_cw,k`, the model
implies a constant candidate

```text
R_0,k = Rhat_cw,k B S(theta_k).
```

Initialize `R_0` by the chordal mean followed by projection to `SO(3)`:

```text
M = sum_k w_k R_0,k / sum_k w_k
U, _, V^T = svd(M)
R_0 = U diag(1,1,det(U V^T)) V^T.
```

The reference weights are uniform over registered views.  This is a
**Frozen interpretation** because the paper provides no `R_0` initialization
rule. The reference initializer requires the positive FFT frequency; later
orbit equations can still represent and report a negative optimized
frequency. Every `theta_k` here uses the same frozen `t_ref=t_{k*}`. A geodesic
mean or a look-at initialization is **Sensitivity only**.

### Frozen factor-observation scope

Let `O` contain every fixed-undistorted Eq. (4) observation emitted by the
front end, and let `M` be the selected COLMAP model's bijective mapping from a
source track to an initialized 3-D point.  The reference factor set is

```text
F = {(p, k, z_p,k) in O : p is in domain(M)}.
```

Factors are ordered by `(track_id, window_index)`.  Membership in the set of
registered COLMAP views is used to initialize the orbit geometry, but does
not filter `F`: the analytic orbit gives a camera pose for every observation
time.  Likewise, an original Eq. (4) observation for an accepted source track
remains in `F` when that image feature is absent from the selected COLMAP
point track.  Geometric verification and reconstruction initialize the
point; they do not rewrite the measured source track.

An empty track/window pair has no Eq. (4) measurement and therefore no
factor. Split, merged, and uninitialized source tracks have no unique point
initialization and are excluded as whole tracks.  Factor membership is fixed
before optimization and cannot depend on residual, depth, ground truth, or
scene outcome. All measurements use the same fixed undistortion and camera
matrix as COLMAP input.

This rule is an **Ambiguous / Frozen interpretation** of Eq. (5)'s schematic
double sum, Fig. 4, and Sec. 5.2's instruction to project points to cameras
where they are visible. It is not claimed to reproduce unpublished author
code. A COLMAP-inlier-only factor graph is a sensitivity variant.

### Frozen direct-factor coordinates and residual

Each observation contributes one two-dimensional residual with the variable
blocks ordered as

```text
(x_p, c, r, f, R_0, n_raw, u_raw).
```

The point `x_p`, centre `c`, raw normal `n_raw`, and raw in-plane direction
`u_raw` use additive Euclidean coordinates; `r` and `f` are unconstrained
real scalars; and `R_0` uses the GTSAM `Rot3` right-local perturbation
`R_0 Exp(delta)`. Inside every factor, form

```text
n = n_raw / ||n_raw||
u = normalize((I - n n^T) u_raw)
v = n x u
B = [u v n].
```

Zero `n_raw`, or a projected `u_raw` whose norm is at most
`64*eps*max(1,||u_raw||)`, is a factor-evaluation failure. The camera pose is
the frozen orbit pose above. With the fixed zero-skew `PINHOLE` calibration,
the residual is

```text
e_p,k = project(K, R_cw(t_k) (x_p - C_w(t_k))) - z_p,k.
```

Thus the sign is predicted minus observed. At a non-positive camera depth the
factor is **not** removed and the optimizer is **not** interrupted. Following
the pinned GTSAM 4.2 `GenericProjectionFactor` default, it returns the constant
residual `(2*fx,2*fx)` and zero Jacobians for all seven variable blocks. This
lets LM reject or move through an invalid trial step. Initial and final
non-positive-depth factor counts are recorded. If any remain at the final
state, the reference scene fails closed and its reprojection metric is
unavailable; the constant penalty is never presented as a predicted pixel or
mixed into the Fig. 10-style pixel-residual distribution. This convention is
audited against
[GTSAM commit `4f66a491...`, `ProjectionFactor.h` lines 153--163](https://github.com/borglab/gtsam/blob/4f66a491ffc83cf092d0d818b11dc35135521612/gtsam/slam/ProjectionFactor.h#L153-L163).

Negative optimized `r` or `f` values are evaluated and reported rather than
hidden with a positivity transform. Every factor uses isotropic
`sigma=1 px`, squared Gaussian error, no robust loss, and no prior.

### Frozen derivative and gauge conventions

The direct GTSAM factor supplies analytic Jacobians for all seven blocks in
the order above. Central finite differences use

```text
h_i = eps_float64^(1/3) * max(1, |value_i|)
```

for additive coordinates and `h_R=eps_float64^(1/3)` radians for each
right-local `Rot3` coordinate. On the deterministic nondegenerate Gate C
fixture, each analytic block must agree with its central difference under
`atol=2e-7` and `rtol=2e-7`. These rules were fixed from float64
roundoff/truncation balance on synthetic geometry before public-data
execution; they are not TOPSPIN-tuned tolerances.

The reprojection system is not expected to identify one raw parameter vector.
Its continuous gauge contains the global similarity transform (three
translations, three rotations, and one scale), the magnitude of `n_raw`, the
magnitude of the projected `u_raw`, and the component of `u_raw` parallel to
`n_raw`: ten local null directions in total. The discrete
`(n,f)->(-n,-f)` representation is additionally projection-equivalent after
the corresponding `R_0` basis update. Gate C synthetic recovery therefore
judges final residuals and Sim(3)-aligned geometry, records Jacobian rank and
the known null directions, and does not add a gauge-fixing prior. If default
GTSAM 4.2 Levenberg--Marquardt raises an indeterminate-system exception, the
reference run fails without retrying another solver.

## Required pre-data tests

All tests below use synthetic inputs and must pass before a TOPSPIN scene is
used.  Passing them validates implementation consistency, not paper accuracy.

### Plane and circle tests

- Recover a known non-axis-aligned plane and circle from exact centres.
- Verify `n^T u = 0`, unit norms, `v = n x u`, and the positive temporal sign.
- Use a sparse-view fixture where the sample centroid gives the wrong chord
  sign; verify the circle-centre rule before and after a world rigid transform.
- Verify recovery after an arbitrary global rotation and translation.
- Exercise `n` parallel, near-parallel, antiparallel, and near-antiparallel to
  `z`, including the Cartesian-axis tie rule.
- Reject rank-deficient plane fits, collinear circle points, negative `r^2`,
  and an unobservable first radial direction.

### Analytic orbit tests

For `f > 0`, express these times relative to the frozen `t_ref` and verify the
Eq. (1) centres:

```text
C(t_ref)          = c + r u
C(t_ref+1/(4f))   = c + r v
C(t_ref+1/(2f))   = c - r u.
```

At every tested time, recover the camera centre from the homogeneous pose and
compare it with Eq. (1).  For pairs of times, compare the relative
camera-to-world orientation with `Rot(n,2 pi f Delta t)`.
For the initialization fixture, verify that the analytic orbit passes the
earliest registered centre `C_{k*}` exactly.  Shift every absolute time and
`t_ref` by the same constant and verify unchanged centres, poses, and
projections.

### Transform and projection tests

- A point transformed world-to-camera and back with the inverse pose must
  round-trip to floating-point tolerance.
- Homogeneous projection and the explicit `K(R_cw x_w + t_cw)` expression
  must agree.
- A noise-free point visible at multiple analytic poses must have zero
  reprojection residual at the generating parameters.
- A mapped source track must contribute all original Eq. (4) observations in
  deterministic order, including observations in an unregistered logical
  window and observations absent from the selected COLMAP point track.
- Empty track/window pairs and non-bijective or uninitialized tracks must not
  contribute factors; factor counts must not depend on synthetic residuals.
- Drive a real LM trial through non-positive depth; verify the constant GTSAM
  penalty, zero Jacobians, no optimizer abort, auditable initial/final counts,
  and fail-closed final reporting.

### Gauge and sign tests

- Apply one global rigid transform to camera centres, points, `c`, `n`, and
  `u`; after the corresponding update of `R_0`, image projections must be
  unchanged.
- Verify the algebraic sign-equivalent representation
  `(n,f) -> (-n,-f)` while updating `v=n x u`; it must produce the same centres
  and be canonicalized to the frozen positive-frequency / temporal-sign
  convention.
- Verify that a constant change of orbit-frame axes is absorbed by `R_0` and
  leaves projections unchanged.

### Jacobian tests

Every orbit-factor Jacobian used by the optimizer must be compared against a
central finite difference on a deterministic, nondegenerate fixture.  Gate C
must freeze the perturbation coordinates, finite-difference step, absolute and
relative tolerances, and scaling for `r`, `f`, `c`, `n`, `u`, `R_0`, and the
3D point before accepting the direct-GTSAM factor.  Tolerances may not be
selected from TOPSPIN behavior.

If direct GTSAM raises `IndeterminantLinearSystemException` (or an equivalent
singular-system failure), reference evaluation reports a failure.  It must not
add a prior, change the 14-parameter representation, or retry with a different
solver in response.

## Reporting rule

Outputs must preserve both the paper equations and the frozen executable
convention in their provenance.  A result from this convention is a
reproduction attempt under an audited interpretation; it is not evidence that
the authors used the same missing conventions.  Any alternate plane fit,
pose composition, `R_0` initialization, or sign rule belongs to a separately
labelled sensitivity or experimental path.
