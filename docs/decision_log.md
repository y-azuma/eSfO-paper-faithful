# Paper-faithful reproduction decision log

Status: **Gate 0 revision, addressing Claude Code review of `7e498ee`**
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

## D-003 - Frozen primary interpretations proposed for Gate 0 approval

These are result-independent interpretations. Paper-reported parts and
inferred parts must remain distinguishable in code, provenance, and reports.

1. **Frontend time coordinate.** Convert public TOPSPIN timestamps as
   `t_ms = (t_us - t_first_us) / 1000`. Use this same coordinate for Eq. (2),
   HDBSCAN, and Eq. (3). Use seconds only for physical frequency/orbit
   equations. This is Ambiguous, but milliseconds are the most paper-derived
   choice because Table 2 gives both time windows in milliseconds and
   `phi=30` then has the same order as `delta_t=30 ms`. Seconds and raw
   microseconds are preregistered sensitivity variants only.
2. **Eq. (2).** Use Euclidean distance in `(t_ms,x,y)` with inclusive radius
   `lambda=7`. The neighbor set is the raw event set `E`; the query is an
   eFAST corner and only same-polarity neighbors are counted. The polarity-wise
   mean is over corner queries across the whole sequence. Including the query
   itself is recorded but is not a consequential branch because it shifts all
   scores and their mean equally. Corner-only neighbors and spatial-only
   distance remain sensitivity variants.
3. **HDBSCAN.** Use the McInnes `hdbscan` package, pinned to `0.8.33` (the
   final release available before the paper's submission period), with
   `min_cluster_size=10`, `min_samples=None` (therefore 10),
   `cluster_selection_epsilon=5`, Euclidean distance,
   `approx_min_span_tree=True`, `cluster_selection_method="eom"`,
   `alpha=1.0`, and `allow_single_cluster=False`. Discard noise label `-1`.
   Reference mode operates on the complete filtered-corner sequence; it may
   not silently chunk, subsample, or use approximate external clustering.
4. **Eq. (3) association.** Compute each original HDBSCAN cluster's head and
   tail descriptor once. From every cluster tail, create at most one directed
   edge to the nearest distinct head satisfying `t_head >= t_tail` and the
   paper's strict distance gate `< phi`. Break equal distances by earlier head
   time and then smaller stable cluster ID. Merge weakly connected components
   once. Sequential descriptor recomputation and one-to-one matching are not
   reference behavior.
5. **Eq. (4) track windows.** Shift the first timestamp to zero and implement
   the displayed indices literally: `k=1..floor(T/delta_t)-1`, window
   `(k*delta_t,(k+1)*delta_t]`, observation time `t_k=k*delta_t`. Thus the
   first 30 ms are unused. An empty window produces no observation, not an
   interpolated point.
6. **COLMAP.** Pin COLMAP `3.9.1`, the final release available before the
   paper's publication. Undistort Eq. (4) mean pixel locations with each
   scene's `calib.txt`, then use one fixed known `PINHOLE` camera. Insert
   correspondences for every pair of windows observing the same track, run
   COLMAP's default geometric verification and incremental mapper, keep the
   largest model, and exclude unregistered windows/tracks from later stages.
   Record every effective option and failure. Ignoring distortion, only
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
10. **Unavailable metrics.** Without independently authenticated numeric
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
   fail-closed unavailable metrics.
6. **Gate E:** fixed implementation-review scenes, then one all-72 run only
   after Claude Code and Codex approve A-D.

## D-006 - Source versions and data rights

- Method citations use the CVF final PDF, SHA-256
  `70ac861187cb75d840e40ef0f67c5392cf9587d15f33f5ea42022b4df9df6f21`.
- Claude Code independently reviewed arXiv v1, SHA-256
  `7ad59e3495d37a0772e8d2274db9202a0c1601f1434779aaa4ad3b826d6caa00`.
  A normalized full-text comparison found identical method text, Eqs. (1)-(14),
  tables, and references; the only residual tokens were arXiv page numbers
  2-10. Both identities remain in provenance.
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

## Change log

| Date | Decision | Evidence | Agreed by |
|---|---|---|---|
| 2026-11-24 | Repository and Gate 0 draft created; no implementation or result accepted yet. | CVPR paper and public-artifact audit | Codex draft; Claude review pending |
| 2026-11-24 | Gate 0 draft rejected with seven documentation blockers; primary interpretations revised without running code or TOPSPIN. | Claude Code review of `7e498ee`; arXiv v1/CVF comparison; Zenodo API | Codex revision; Claude rereview pending |

