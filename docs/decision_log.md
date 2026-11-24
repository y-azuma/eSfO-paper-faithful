# Paper-faithful reproduction decision log

Status: **Gate A mutually approved; Gate B may proceed**
Scope: eSfO reproduction, not a method-improvement project.

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
   COLMAP's default geometric verification and incremental mapper, keep the
   largest model, and exclude unregistered windows/tracks from later stages.
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
- The temporal normal-sign rule is sampled from registered camera centres.
  After the positive FFT frequency is available, any adjacent registered-view
  phase gap of at least `pi` is rejected as `temporal_sign_aliased`; otherwise
  an aliased direction could be hidden by the later `R_0` initialization.

These choices were made from versioned source behavior, algebra, and synthetic
identifiability before TOPSPIN execution. They may be changed only for a
reproducible code defect or stronger primary evidence, never for performance.

## Change log

| Date | Decision | Evidence | Agreed by |
|---|---|---|---|
| 2026-11-24 | Repository and Gate 0 draft created; no implementation or result accepted yet. | CVPR paper and public-artifact audit | Codex draft; Claude review pending |
| 2026-11-24 | Gate 0 draft rejected with seven documentation blockers; primary interpretations revised without running code or TOPSPIN. | Claude Code review of `7e498ee`; arXiv v1/CVF comparison; Zenodo API | Codex revision; Claude rereview pending |
| 2026-11-24 | Gate 0 conditionally approved and Gate A authorized; phase origin and seven non-blocking audit details frozen before Gate B. | Claude Code review of `6826b9f` | Claude Code and Codex |
| 2026-11-24 | Gate A audit exposed an unstated SAE-initialization convention. Negative infinity was frozen because relative time starts at zero and every unseen cell must be older than every observed event. No TOPSPIN or other performance result was run or used; cross-review remains pending. | Static comparison with cited eFAST revision; synthetic time-shift and initialization fixtures | Codex; Claude Code review pending |
| 2026-11-24 | Gate A approved with no blocking findings and Gate B authorized. Negative-infinity SAE initialization was mutually accepted after an independent differential check reported zero disagreements on 171,761 synthetic events. The literal time equation was changed from multiplication to division to remove a one-ulp implementation/documentation mismatch; no public data was used. | Claude Code review of `d57e63f` and `dffc27b`; local synthetic regression tests | Claude Code and Codex |
| 2026-11-24 | Gate B implementation and exact COLMAP 3.9.1 two-run synthetic integration completed. Parser/configuration guards and the temporal-sign alias failure were added from result-independent audits; no TOPSPIN scene or performance result was used. Gate B remains unapproved until independent review. | 107 synthetic unit tests; `artifacts/gate_b/colmap_synthetic/manifest.json` | Codex implementation; Claude Code review pending |

