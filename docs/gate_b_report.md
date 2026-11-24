# Gate B report: initialization components

Status: **local implementation, synthetic unit tests, and pinned COLMAP 3.9.1
sidecar integration complete; independent review pending**.

Gate B implements the paper's COLMAP preparation, Eqs. (6)--(10)
initialization, and the frozen conventions needed to make those operations
executable. No TOPSPIN scene, public-data performance, numeric speed label, or
CAD result was used to choose or validate an interpretation.

## Implemented reference components

### FFT frequency initialization

- accepts only the raw `EventSequence`, not corners or tracks;
- uses exact integer-microsecond 20 ms boundaries;
- implements `f=1..floor(T/delta_tf)-1` and `(f*dt,(f+1)*dt]`;
- fills empty displayed windows with the unweighted mean of nonempty-window
  means;
- applies a rectangular `numpy.fft.rfft` with no detrending, taper, padding,
  interpolation, range prior, or harmonic correction;
- excludes DC, includes Nyquist, and resolves exact ties toward the lower
  frequency;
- records all omitted/used event counts, the complete spectrum, and the
  zero-positive-power diagnostic.

### Plane, circle, orbit geometry, and `R_0`

- derives COLMAP camera centres as `C_w=-R_cw^T t_cw`;
- implements the printed ordinary `z=ax+by+c` regression and
  `n proportional to (a,b,-1)` rather than substituting total least squares;
- fixes the normal sign from temporal motion and explicitly rejects an
  unobservable sign;
- rejects registered-view sampling whose positive-frequency phase gaps reach
  `pi`, where the temporal sign is Nyquist-aliased;
- implements all general/parallel/antiparallel Rodrigues branches;
- fits the circle from centred, plane-aligned camera centres;
- defines zero phase and `u` from the earliest registered view;
- implements the invariant-derived pose convention and uniform chordal
  `R_0` initialization;
- rejects both rank-deficient and non-unique `R_0` means rather than accepting
  an SVD-basis-dependent answer.

The exact float64 rank and degeneracy rules are frozen in
[`geometry_conventions.md`](geometry_conventions.md). They are reproducibility
rules, not fitted thresholds.

### COLMAP 3.9.1 preparation and audit

- strictly parses the nine-value calibration and specifies
  `opencv-python-headless==4.9.0.80` for
  `undistortPoints(...,R=None,P=K)`;
- preserves empty logical windows, sorts feature indices by source track, and
  streams every shared-track image pair;
- checks the generated correspondence total against
  `sum_p choose(track_length_p,2)`;
- writes one fixed known `PINHOLE` camera and two-column float32 keypoints to a
  database created by the pinned binary;
- uses the COLMAP 3.9.1 route
  `matches_importer --match_type raw`, which performs geometric verification,
  never `--match_type inliers`;
- records CPU matching, seed zero, all three disabled intrinsic-refinement
  flags, and `Mapper.extract_colors=0`;
- validates raw-match and fixed-camera database postconditions, including the
  complete COLMAP 3.9.1 two-view defaults, configuration/inlier-count
  consistency, unique verified rows, and at least one mapper-usable pair;
- selects the largest model by registered images, then points, then lowest
  numeric directory ID;
- binds each converted text model to its selected numeric directory and
  requires exact bidirectional agreement between `images.txt` feature point
  IDs and `points3D.txt` tracks;
- excludes and counts non-bijective source-track/COLMAP-point mappings rather
  than choosing a favorable reconstruction.

This repository does not implement a substitute SfM algorithm. The local
tests validate preparation, database semantics, commands, and postconditions.
Execution of the complete synthetic fixture through the exact COLMAP 3.9.1
sidecar passed as recorded below. This validates plumbing and conventions, not
paper-level TOPSPIN accuracy.

## Synthetic verification

The current Gate B additions contain 50 focused tests:

- 14 FFT fixtures for boundaries, floor, empty fill, DC, power, exact ties,
  Nyquist, harmonic non-correction, raw-event use, invariance, immutability,
  provenance, and failures;
- 21 geometry fixtures for exact non-axis orbits, temporal sign and aliasing,
  all Rodrigues branches and threshold sides, phase/time/frame invariance,
  analytic centres,
  pose/projection identities, sign equivalence, chordal reflection correction,
  degeneracies, and immutable diagnostics;
- 15 COLMAP-preparation fixtures for calibration, undistortion call semantics,
  logical views, all-pairs counts, importer text, pair IDs, database blobs,
  fixed `K`, raw/verified postconditions, commands, model selection, and
  source-track/point mapping.

Combined with Gate A and provenance tests, both the local environment and the
4070 Ubuntu host report `107 passed`, statement coverage `84%`, with four
warnings originating inside the frozen `hdbscan==0.8.33` dependency.

### Exact sidecar integration

The CPU sidecar was built from upstream COLMAP tag commit
`e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68` on the pinned Ubuntu 22.04 base.
Its image ID is
`sha256:05e37ea64e73868a25ee09c32f61e060ab2ba3c51f283a01913f5bec3c10ef49`
and `/usr/local/bin/colmap` has SHA-256
`94e23e91028950818ff9fe7fa5b3dbead7bd2e0ba3d3ef165eb533e47e9f846b`.
The binary identifies itself as COLMAP 3.9.1 at commit `e9903641` without
CUDA. Compiler, linked-library, package, command, source-tree, and fixture
identities are retained in the integration manifest.

The generated fixture has 12 well-constrained views, 2 views with only 14
shared tracks, one empty logical view, 200 non-coplanar core points, and one
singleton track. It produces exactly 13,550 raw correspondences. Two fresh,
independent executions produced the same single model:

- 12 registered and 3 unregistered logical views;
- 200 reconstructed, bijectively mapped tracks;
- camera-centre RMSE `1.1763799837836353e-06` after one similarity alignment;
- 3D-point RMSE `6.029560659026428e-07` under that same alignment;
- maximum orientation error `3.0392524681048417e-07 rad`;
- reprojection RMSE `1.6141309267270156e-05 px`;
- zero difference between the two aligned camera and point reconstructions.

The installed distribution was recorded as
`opencv-python-headless==4.9.0.80`; its `cv2` runtime reports OpenCV 4.9.0.
It produced zero error for the zero-distortion case and
`2.0128254618612118e-10 px` maximum error for the fixed five-coefficient
round trip, below the preregistered `1e-6 px` tolerance. The complete synthetic
databases, models, logs, and manifest are stored under
`artifacts/gate_b/colmap_synthetic/`. The manifest records the complete
effective COLMAP plan and individual size/SHA-256 identities for all 34
non-manifest outputs; its own SHA-256 is
`76293f55a4b3d8feea1dc255212bbb7e5809c609042f0682f4e27ee0189e9fce`.

## Required completion before Gate C

1. Obtain independent Claude Code review of the implementation, tests,
   frozen numerical rules, and the sidecar result.

Until that review passes, this report makes no Gate B approval claim and Gate
C must not begin.
