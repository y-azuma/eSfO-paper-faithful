# Paper-faithful reproduction decision log

Status: **Gate A approved; D-009 and the Gate C core approved; B1-prime and
C1 corrections implemented and awaiting focused re-review**
Scope: eSfO reproduction, not a method-improvement project.

Unclosed non-blocking findings and their mandatory closure gates are tracked
in [`open_review_items.md`](open_review_items.md).

## D-001 - Reproduction boundary

- Implement the paper and audited primary citations as literally as available
  evidence permits.
- Do not select an interpretation by TOPSPIN accuracy.
- Put harmonic correction, learned features, tuned time scaling, robust losses,
  defensive recovery, and alternative association/optimization schemes in a
  mechanically separate experimental namespace.
- A poor result is not permission to modify the reference path.

## D-002 - Evidence hierarchy

Resolve ambiguities in this order:

1. displayed equation or explicit paper prose;
2. paper figure or table;
3. audited primary citation;
4. established implementation convention;
5. a simple deterministic convention frozen before public-data execution.

Performance is not an admissible tie-break.

## D-003 - Frozen primary interpretations accepted for conditional Gate 0

These are result-independent interpretations. Paper-reported parts and
inferred parts must remain distinguishable in code, provenance, and reports.

1. **Frontend time coordinate.** Convert public TOPSPIN timestamps as
   `t_ms = (t_us - t_first_us) / 1000`. Use this same coordinate for Eq. (2),
   HDBSCAN, and Eq. (3). Use seconds only for physical frequency/orbit
   equations. This is Ambiguous, but milliseconds are the most paper-derived
   choice because Table 2 gives both time windows in milliseconds and
   `phi=30` then has the same order as `delta_t=30 ms`. Seconds and raw
   microseconds are preregistered sensitivity variants only. In the independent
   eFAST implementation, initialize unseen polarity-specific SAE cells to
   negative infinity after this relative-time conversion. This preserves the
   published strict newest-arc ordering and is equivalent to the cited
   upstream's zero initialization when its input timestamps are positive;
   it is a frozen convention, not a tuned improvement.
2. **Eq. (2).** Use Euclidean distance in `(t_ms,x,y)` with inclusive radius
   `lambda=7`. The neighbor set is the raw event set `E`; the query is an
   eFAST corner and only same-polarity neighbors are counted. The polarity-wise
   mean is over corner queries across the whole sequence. Including the query
   itself is recorded but is not a consequential branch because it shifts all
   scores and their mean equally. Corner-only neighbors and spatial-only
   distance remain sensitivity variants.
3. **HDBSCAN.** Use the McInnes `hdbscan` package, pinned to `0.8.33` (the
   latest non-release-candidate stable version before arXiv v1 on
   2024-05-10), with
   `min_cluster_size=10`, `min_samples=None` (therefore 10),
   `cluster_selection_epsilon=5`, Euclidean distance,
   `approx_min_span_tree=True`, `cluster_selection_method="eom"`,
   `alpha=1.0`, `allow_single_cluster=False`, `algorithm="best"`, and
   `core_dist_n_jobs=4`, together with the otherwise inherited 0.8.33
   defaults `max_cluster_size=0` and `p=None`. Discard noise label `-1`.
   Record all effective
   values, and require repeated identical synthetic input to produce identical
   labels before accepting the dependency in Gate A.
   Reference mode operates on the complete filtered-corner sequence; it may
   not silently chunk, subsample, or use approximate external clustering.
4. **Eq. (3) association.** Compute each original HDBSCAN cluster's head and
   tail descriptor once. From every cluster tail, create at most one directed
   edge to the nearest distinct head satisfying `t_head >= t_tail` and the
   paper's strict distance gate `< phi`. Break equal distances by earlier head
   time and then smaller stable cluster ID. Merge weakly connected components
   once. Sequential descriptor recomputation and one-to-one matching are not
   reference behavior. Report, without changing the merge, every merged track
   containing original-cluster time spans whose closed intervals intersect;
   this diagnostic exposes possible mixed observations caused by the frozen
   weak-component interpretation.
5. **Eq. (4) track windows.** Shift the first timestamp to zero and implement
   the displayed indices literally: `k=1..floor(T/delta_t)-1`, window
   `(k*delta_t,(k+1)*delta_t]`, observation time `t_k=k*delta_t`. Thus the
   first 30 ms are unused. An empty window produces no observation, not an
   interpolated point.
6. **COLMAP.** Pin COLMAP `3.9.1`, the latest non-release-candidate stable
   version before arXiv v1 on 2024-05-10. Undistort Eq. (4) mean pixel
   locations with each
   scene's `calib.txt`, then use one fixed known `PINHOLE` camera. Insert
   correspondences for every pair of windows observing the same track, run
   COLMAP's default geometric verification and incremental mapper, and keep
   the largest model. Use only its registered camera poses and bijectively
   initialized source tracks for the plane/circle/`u`/`R0` initialization.
   Unregistered windows do not by themselves remove an existing Eq. (4)
   observation from the later orbit factor graph; the separate, frozen
   factor-observation rule is D-009.
   Use the original `K` as the new camera matrix after undistortion and set
   `ba_refine_focal_length=false`, `ba_refine_principal_point=false`, and
   `ba_refine_extra_params=false`. Record every effective option and failure.
   Ignoring distortion, only
   adjacent-window pairs, and bypassing verification are sensitivity variants.
7. **Plane/circle initialization.** Follow
   [`geometry_conventions.md`](geometry_conventions.md): interpret the printed
   plane regression as `z=ax+by+c` and set `n proportional to (a,b,-1)`; choose
   its sign from positive temporal rotation; use centered camera centers for
   the 2-D circle fit; treat the starred square in Eq. (8) as a typesetting
   error; project the first radial vector into the plane and normalize it.
8. **FFT initialization.** Use all events, not only corners. Follow the paper's
   `f=1..floor(T/delta_tf)-1` windows; compute mean x per nonempty window;
   fill empty windows with the mean of all nonempty window values; use a
   rectangular real DFT, no zero padding, omit DC from maximization, choose the
   maximum positive-frequency power and break ties toward lower frequency.
   Do not interpolate peaks, apply a range prior, or correct harmonics.
9. **Orbit factor graph.** Treat Eqs. (11)-(14) as Ambiguous rather than exact
   and use the invariant-derived convention in
   [`geometry_conventions.md`](geometry_conventions.md). Variables are 3-D
   points and the paper's 14 orbit parameters: `c,r,f,R0,n,u`. Represent
   `c,n,u` in `R^3`, `r,f` in `R`, and `R0` in `SO(3)`; inside each factor,
   normalize `n` and orthogonalize/normalize `u`. Use direct GTSAM 4.2,
   Levenberg-Marquardt defaults, isotropic `sigma=1 px`, squared residuals,
   no robust loss, no priors, and no positivity transform. Initialize `R0`
   by an SO(3)-projected chordal mean of the COLMAP rotations after removing
   the orbit term. Report negative `r` or `f`; do not hide them with a
   constraint. Any alternate manifold or solver requires a preregistered
   numerical-equivalence gate.
10. **Orbit phase origin.** Let `k*` be the earliest-time registered view in
    the largest retained COLMAP model. Set `t_ref=t_{k*}` in seconds,
    `tau_k=t_k-t_ref`, and `theta_k=2*pi*f*tau_k`; `t_ref` is a fixed gauge
    convention, not an optimized variable. Define `u` from the same `C_{k*}`
    so zero phase is aligned to that centre's plane-projected radial direction
    (and the initial orbit passes it exactly in the noise-free on-circle
    fixture). The paper's literal
    indexing could instead imply that window 1 is registered and `t_ref=0`;
    that unstated assumption differs by a small phase when the first retained
    view is later, and is not the frozen reference interpretation.
11. **Unavailable metrics.** Without independently authenticated numeric
    speed labels or CAD/alignment assets, frequency MAE/RMSE and CAD RMSE in
    millimetres must fail closed. Axis output remains Qualitative because the
    paper itself makes only a qualitative claim. Within-speed consistency and
    values digitized from Fig. 8a are respectively `consistency` and
    `figure-proxy`, never accuracy.

These interpretations remain frozen after implementation begins unless
stronger source evidence or a reproducible code defect is recorded and
cross-reviewed. Public-data performance is not admissible evidence for a
change.

## D-004 - Public-data evidence boundary

- Public TOPSPIN has already been inspected in earlier work. It is not held
  out and results are exploratory/retrospective.
- Numeric frequency GT, authors' CAD, CAD alignment, and axis GT are currently
  unavailable. Corresponding paper metrics must fail closed.
- Fast/Medium/Slow consistency is not accuracy.
- Values digitized from paper plots are figure proxies, not authoritative GT.

## D-005 - Review gates

1. **Gate 0:** sources, matrix, and primary interpretations mutually reviewed.
2. **Gate A:** front-end component tests on synthetic fixtures only.
3. **Gate B:** COLMAP, plane/circle, and FFT initialization tests.
4. **Gate C:** direct-GTSAM factor and gauge-aware synthetic recovery.
5. **Gate D:** integrated synthetic run, provenance, checkpoint/resume, and
   fail-closed unavailable metrics. Before public data, measure all-pairs
   correspondence count, whole-stream HDBSCAN memory, and runtime on a
   deterministic synthetic problem sized to exercise the expected `O(K^2)`
   correspondence risk.
6. **Gate E:** fixed implementation-review scenes, then one all-72 run only
   after Claude Code and Codex approve A-D.

## D-006 - Source versions and data rights

- Method citations use the CVF final PDF, SHA-256
  `70ac861187cb75d840e40ef0f67c5392cf9587d15f33f5ea42022b4df9df6f21`.
- Claude Code independently reviewed arXiv v1, SHA-256
  `7ad59e3495d37a0772e8d2274db9202a0c1601f1434779aaa4ad3b826d6caa00`.
  Codex performed the normalized full-text comparison and found identical
  method text, Eqs. (1)-(14), tables, and references; the only residual tokens
  were arXiv page numbers 2-10. Both identities remain in provenance.
- Unless a paper-specific source states otherwise, dependency selection uses
  one cutoff: the latest stable non-release-candidate version available before
  arXiv v1 on 2024-05-10. This yields `hdbscan==0.8.33`, COLMAP `3.9.1`, and
  GTSAM `4.2`; it does not claim that the authors used those builds.
- TOPSPIN concept DOI is `10.5281/zenodo.10884693`; version DOI is
  `10.5281/zenodo.10884694`. The official API reports license ID
  `cc-by-4.0`; redistribution of derivatives requires attribution.

## D-007 - Preregistered public-data review scope

The fixed implementation-review scenes are selected before any eSfO result is
observed and span all speed categories, three objects, and three viewpoints:

1. `camera-diagonal-slow`;
2. `hubble-perpendicular-med`;
3. `tdrs-sideon-fast`.

They are diagnostic, not a development set and not a paper-comparison sample.
No interpretation or threshold may be chosen from their performance. Reference
mode uses each complete scene. A per-scene wall-clock guard of 24 hours and a
26 GiB process/container RAM cap on the 4070 host are resource limits, not
method parameters; timeout or OOM is reported as failure and remains in the
denominator. Scene checkpoints are allowed only at stage boundaries with
exact input/config/code fingerprints. After Gates A-D and mutual review, all
72 public scenes are run once; adaptive retries or selective exclusion are
forbidden.

## D-008 - Gate B executable conventions frozen before implementation review

- FFT initialization uses exact integer-microsecond 20 ms boundaries from the
  raw `EventSequence`, so window membership cannot depend on the one-ulp
  float conversion discussed in the Gate A review. Nyquist is a positive
  candidate. If every positive-frequency power is exactly zero, the exact-tie
  rule returns the lowest positive bin and records
  `zero_positive_power=true`; adding a confidence threshold would be a new
  paper-unstated rule.
- The plane/circle implementation uses the float64 rank and degeneracy rules
  recorded in [`geometry_conventions.md`](geometry_conventions.md). In
  particular, it does not replace the printed ordinary regression with TLS
  when a world-frame transform makes the regression vertical.
- A largest COLMAP model is ordered first by registered image count, then by
  reconstructed point count, then by the lowest numeric model-directory ID.
  A source track mapping to multiple reconstructed points, or multiple source
  tracks mapping to one point, is not resolved by selecting the most
  convenient point: every non-bijective mapping is excluded and counted.
- For COLMAP 3.9.1, raw all-pairs correspondences are imported through
  `matches_importer --match_type raw`; this is the version-correct route that
  invokes its geometric verification. `--match_type inliers` is prohibited
  because it bypasses verification. The mapper uses an empty image directory
  with `Mapper.extract_colors=0`, since no intensity image is part of an event
  window, and all three intrinsic-refinement switches remain false.
- The temporal normal-sign rule is sampled from registered camera centres
  after the positive FFT frequency is available. Adjacent registered-view
  pairs with a phase gap of at least `pi` are excluded from the score and
  counted; the maximum gap is recorded. The initialization fails as
  `temporal_sign_aliased` only when no non-aliased pair remains, and fails as
  `temporal_sign_unobservable` when the retained score is numerically zero.
  This is a result-independent identifiability rule and not a quality-based
  pair selection.

These choices were made from versioned source behavior, algebra, and synthetic
identifiability before TOPSPIN execution. They may be changed only for a
reproducible code defect or stronger primary evidence, never for performance.

## D-009 - Gate C factor-observation scope frozen after Gate B review

The paper states that every Eq. (4) feature observation is connected to the
global orbit parameters and its 3-D point, and that points are projected to
cameras where they are visible. Its Eq. (5) double sum is schematic: Eq. (4)
does not define a measurement for an empty track/window pair. Before Gate C
implementation and without inspecting TOPSPIN performance, freeze the
executable observation set as follows.

- Let `O` be the original, fixed-undistorted Eq. (4) observations and let `M`
  be the bijective source-track-to-COLMAP-point initialization from the
  selected model. Create exactly one orbit factor for each observation in
  `O` whose source track lies in the domain of `M`, ordered by
  `(track_id, window_index)`.
- For a track in `M`, retain every one of its original Eq. (4) observations,
  including observations in logical windows that COLMAP did not register and
  observations not retained in the selected model's `points3D` track. The
  orbit model defines a pose at every `t_k`; COLMAP registration and geometric
  verification initialize the model but do not redefine the measured track.
- Do not fabricate a factor for an empty track/window pair. Exclude an entire
  split, merged, or uninitialized source track because it lacks a unique 3-D
  point initialization. Do not prune factors by reprojection residual, depth,
  ground truth, scene outcome, or public-data performance.
- Use the same fixed undistortion and `K` convention used for COLMAP input.
  Record total factors, factors in registered and unregistered windows,
  factors present and absent in the selected COLMAP point tracks, missing
  track/window pairs, and every excluded-track category. Split, merged, and
  uninitialized categories are recorded independently and are not forced to
  be disjoint.

This is an **Ambiguous / Frozen interpretation**, not a claim about the
authors' unpublished implementation. It is the most literal reconciliation
of Sec. 4.3, Eq. (5), Fig. 4, and Sec. 5.2's phrase "cameras where they are
visible". A COLMAP-inlier-only graph is a separately labelled sensitivity
variant and cannot replace the reference path based on its result.

## D-010 - Gate C local coordinates, derivatives, and gauge

Before any public-data optimization, freeze the direct-factor convention as
follows.

- Residual order is predicted fixed-PINHOLE pixel minus the undistorted Eq.
  (4) observation. Variable/key order is point, `c`, `r`, `f`, `R0`, raw `n`,
  raw `u`.
- Point, `c`, raw `n`, and raw `u` use additive `R^3` coordinates; `r` and
  `f` are additive real scalars; `R0` uses GTSAM `Rot3` right-local
  coordinates. Each factor normalizes raw `n` and projects then normalizes raw
  `u` before constructing `v=n cross u`.
- Supply analytic Jacobians. Compare them with central differences using
  `eps_float64^(1/3)*max(1,abs(value))` for additive coordinates and
  `eps_float64^(1/3)` radians for each right-local rotation coordinate. The
  preregistered synthetic tolerance is `atol=rtol=2e-7`.
- A degenerate normalized basis is a reported factor failure. A non-positive
  depth keeps the factor and returns GTSAM 4.2's default constant
  `(2*fx,2*fx)` penalty with zero Jacobians, so an invalid LM trial does not
  abort the complete optimization. Record initial/final counts; a nonzero
  final count makes reference metrics unavailable. Run default GTSAM 4.2
  Levenberg--Marquardt once, without priors, robust loss, solver fallback, or
  retry.
- Judge synthetic recovery by reprojection and one joint proper-Sim(3)
  alignment of reconstructed points and orbit centres, not raw parameter
  equality. Record the ten expected continuous null directions: global
  translation (3), rotation (3), scale (1), raw-normal magnitude (1),
  projected raw-`u` magnitude (1), and the raw-`u` component parallel to the
  raw normal (1).

These rules are implementation conventions required by omissions in the
paper. They were fixed from algebra and deterministic synthetic geometry,
without TOPSPIN execution or performance-based selection.

## D-011 - Post-review temporal-centre and cheirality corrections

Independent review approved D-009 and the Gate C factor mathematics,
Jacobians, GTSAM binding, provenance, ten-dimensional gauge, and joint-Sim(3)
recovery test, but found two execution defects in frozen conventions. Before
integrated Gate D evidence and without public-data execution, freeze their
corrections as follows.

- **B1-prime:** fit the unsigned plane and implicit circle before choosing the
  normal sign. Use the resulting sign-independent world circle centre `c^w`
  in `q_k=(I-nn^T)(C_k-c^w)`, then apply the existing phase-gap `<pi` pair
  rule. Never substitute the sample centroid. Re-expressing the circle after
  the sign choice must recover the same world centre within the documented
  float64 scale tolerance.
- **C1:** follow GTSAM 4.2 `GenericProjectionFactor` with
  `throwCheirality=false`: for depth `<=0`, retain the factor, return residual
  `(2*fx,2*fx)`, and return zero for all seven Jacobian blocks. Record initial
  and final penalty counts. If any remain after the single default-LM run,
  fail the reference scene and mark reprojection metrics unavailable; never
  mix the penalty's pseudo-pixel into Fig. 10-style residuals.

B1-prime follows the circular orbit in Eq. (1); C1 follows pinned GTSAM commit
`4f66a491ffc83cf092d0d818b11dc35135521612`,
`gtsam/slam/ProjectionFactor.h` lines 153--163. Neither rule was chosen from a
TOPSPIN result.

## Change log

| Date | Decision | Evidence | Agreed by |
|---|---|---|---|
| 2026-11-24 | Repository and Gate 0 draft created; no implementation or result accepted yet. | CVPR paper and public-artifact audit | Codex draft; Claude review pending |
| 2026-11-24 | Gate 0 draft rejected with seven documentation blockers; primary interpretations revised without running code or TOPSPIN. | Claude Code review of `7e498ee`; arXiv v1/CVF comparison; Zenodo API | Codex revision; Claude rereview pending |
| 2026-11-24 | Gate 0 conditionally approved and Gate A authorized; phase origin and seven non-blocking audit details frozen before Gate B. | Claude Code review of `6826b9f` | Claude Code and Codex |
| 2026-11-24 | Gate A audit exposed an unstated SAE-initialization convention. Negative infinity was frozen because relative time starts at zero and every unseen cell must be older than every observed event. No TOPSPIN or other performance result was run or used; cross-review remains pending. | Static comparison with cited eFAST revision; synthetic time-shift and initialization fixtures | Codex; Claude Code review pending |
| 2026-11-24 | Gate A approved with no blocking findings and Gate B authorized. Negative-infinity SAE initialization was mutually accepted after an independent differential check reported zero disagreements on 171,761 synthetic events. The literal time equation was changed from multiplication to division to remove a one-ulp implementation/documentation mismatch; no public data was used. | Claude Code review of `d57e63f` and `dffc27b`; local synthetic regression tests | Claude Code and Codex |
| 2026-11-24 | Gate B implementation and exact COLMAP 3.9.1 two-run synthetic integration completed. Parser/configuration guards and the temporal-sign alias failure were added from result-independent audits; no TOPSPIN scene or performance result was used. Gate B remains unapproved until independent review. | 107 synthetic unit tests; `artifacts/gate_b/colmap_synthetic/manifest.json` | Codex implementation; Claude Code review pending |
| 2026-11-24 | Gate B review exposed an ambiguous factor-observation scope. Before Gate C and without public-data execution, D-009 froze the literal Eq. (4) observation set for every bijectively initialized track; COLMAP registration is used for initialization, not residual-based observation selection. | Sec. 4.3 Eq. (4), Sec. 5.1 Eq. (5), Sec. 5.2 and Fig. 4; Claude Code Gate B review | Codex interpretation; Claude Code confirmation pending |
| 2026-11-24 | Gate B received conditional approval. The sole blocking finding B1 was addressed by excluding only FFT-phase-aliased adjacent registered-view pairs from the temporal-sign score, failing only when none remain or the retained score is unobservable, and recording exclusion diagnostics. | Claude Code Gate B review; focused synthetic fixtures | Claude Code finding; Codex implementation; re-review pending |
| 2026-11-24 | Gate C factor coordinates, residual sign, finite-difference rule, and ten-dimensional gauge were frozen before public-data execution. | Orbit algebra; GTSAM 4.2 local-coordinate API; deterministic synthetic rank analysis | Codex implementation; Claude Code review pending |
| 2026-11-24 | Gate C direct factors, analytic Jacobians, D-009 observation provenance, prior-free GTSAM recovery, and the ten-dimensional gauge passed the complete synthetic suite. Cross-platform exact-invariance assertions were normalized from `5e-14`/`6e-14` to `2e-13`; no method or public-data result changed. This is intermediate core evidence, superseded for final Gate C status by the B1-prime/C1 correction evidence. | 135 synthetic tests; implementation commit `1bccdd693dcdf229a0db03825ff04b03e8b3a505`; GitHub Actions run `45368753199` | Codex implementation and cross-platform verification; Claude Code core review pending at that point |
| 2026-11-24 | Independent review approved D-009 and the Gate C core, then identified B1-prime (centroid-referenced sparse chord) and C1 (CustomFactor cheirality exception). D-011 froze circle-centre sign scoring and the pinned GTSAM constant-penalty behavior before public-data execution. | Claude Code review of `bb1efce`; 899 noise-free orbit audit; GTSAM 4.2 primary source | Claude Code findings; Codex corrections; focused re-review pending |
| 2026-11-24 | B1-prime and C1 were implemented with deterministic sparse-view, transformed-view, exact-penalty, zero-Jacobian, diagnostic-binding, and real negative-depth LM-trial fixtures. The complete corrected synthetic suite passed locally and on pinned Ubuntu CI; no TOPSPIN scene was executed. | 139 tests; implementation commit `6640dad46f51825fa311b72d44aceccc4bf0e673`; GitHub Actions run `45372999990`, job `138371780290` | Codex implementation and independent audit; Claude Code focused re-review pending |

