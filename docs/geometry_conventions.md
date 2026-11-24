# Geometry conventions for the paper-faithful reference path

Status: **Gate 0 revision; no implementation-fidelity claim yet**.

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

2. Fix the sign of `n` from temporal motion.  For plane-projected centred
   vectors `q_k = (I - n n^T) X_k`, compute

   ```text
   s = sum_k n^T (q_k x q_{k+1}).
   ```

   Flip `n` if `s < 0`, so increasing time is positive rotation about `n`.
   This agrees with Eq. (1), `v = n x u`, and the frozen positive-frequency
   convention.  If `s` is numerically zero, the sign is unobservable and the
   initialization fails explicitly.

3. Construct `A = Rod(n,z)`, the active rotation mapping `n` to
   `z=(0,0,1)`.  For neither-parallel case use

   ```text
   axis  = normalize(n x z)
   angle = atan2(||n x z||, n^T z).
   ```

   If `n` is parallel to `z`, use the identity.  If `n` is antiparallel to
   `z`, use a pi rotation about the deterministic Cartesian axis least aligned
   with `n`.  These branch rules are **Frozen interpretations**.

4. Project the **centred** points as

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
   reference-path failure.

5. Form the first radial direction from the first registered camera centre,
   project it onto the fitted plane, and normalize:

   ```text
   u_raw = (I - n n^T) (C_1 - c^w)
   u     = normalize(u_raw)
   v     = normalize(n x u).
   ```

   Projecting before normalization is a **Frozen interpretation** required
   because the COLMAP centre need not lie exactly on the fitted plane.  A
   near-zero `u_raw` is a reported degeneracy.

An SVD/total-least-squares plane fit is geometrically attractive but is not
the regression displayed in Eq. (6); it is therefore **Sensitivity only**.

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

Define the right-handed orbit basis and phase rotation

```text
B        = [u  v  n] in SO(3), where v = n x u
theta_k  = 2 pi f (t_k - t_ref)
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
rule.  A geodesic mean or a look-at initialization is **Sensitivity only**.

## Required pre-data tests

All tests below use synthetic inputs and must pass before a TOPSPIN scene is
used.  Passing them validates implementation consistency, not paper accuracy.

### Plane and circle tests

- Recover a known non-axis-aligned plane and circle from exact centres.
- Verify `n^T u = 0`, unit norms, `v = n x u`, and the positive temporal sign.
- Verify recovery after an arbitrary global rotation and translation.
- Exercise `n` parallel and antiparallel to `z`.
- Reject rank-deficient plane fits, collinear circle points, negative `r^2`,
  and an unobservable first radial direction.

### Analytic orbit tests

For `f > 0` and `t_ref=0`, verify the Eq. (1) centres:

```text
C(0)       = c + r u
C(1/(4f))  = c + r v
C(1/(2f))  = c - r u.
```

At every tested time, recover the camera centre from the homogeneous pose and
compare it with Eq. (1).  For pairs of times, compare the relative
camera-to-world orientation with `Rot(n,2 pi f Delta t)`.

### Transform and projection tests

- A point transformed world-to-camera and back with the inverse pose must
  round-trip to floating-point tolerance.
- Homogeneous projection and the explicit `K(R_cw x_w + t_cw)` expression
  must agree.
- A noise-free point visible at multiple analytic poses must have zero
  reprojection residual at the generating parameters.

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

## Reporting rule

Outputs must preserve both the paper equations and the frozen executable
convention in their provenance.  A result from this convention is a
reproduction attempt under an audited interpretation; it is not evidence that
the authors used the same missing conventions.  Any alternate plane fit,
pose composition, `R_0` initialization, or sign rule belongs to a separately
labelled sensitivity or experimental path.
