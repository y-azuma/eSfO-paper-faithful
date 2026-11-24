# Frozen evaluation protocol (draft)

## Objectives

The evaluation asks whether the frozen primary interpretation executes and
whether the paper's publicly auditable claims can be reproduced. It does not
optimize the method.

All unresolved independent-review obligations are tracked in
[`open_review_items.md`](open_review_items.md); no public-data run may bypass
that register.

## Stages

### Synthetic component and integration checks

- Use generated geometry/events only to verify equations, conventions,
  Jacobians, and software plumbing.
- Synthetic success is not evidence of TOPSPIN performance.
- Freeze all thresholds before executing public-data review scenes.

### Front-end benchmark

- Audit the Event Camera Dataset and cited eCDT evaluation protocol first.
- Record exact sequences, intervals, ground-truth poses, tracker versions,
  outlier rules, and aggregation.
- If any consequential condition is unavailable, label Tables 3-4 comparison
  Ambiguous or Unavailable rather than silently substituting a new protocol.

### TOPSPIN

- Verify all archives against Zenodo sizes and MD5 values.
- Record source URL, record version, acquisition time, archive hash, scene
  hash, calibration, event count, duration, and timestamp unit.
- Use only the preregistered three scenes in D-007 for implementation review:
  `camera-diagonal-slow`, `hubble-perpendicular-med`, and
  `tdrs-sideon-fast`. They are diagnostic, never a tuning set.
- Run all 72 scenes once after Gates 0-D and both reviewers approve.
- Include every failure in the denominator and record the first failing stage.
- Do not change reference settings after observing results.
- If every positive-frequency FFT power is exactly zero, record
  `fft_frequency_unobservable` and stop that reference scene. Do not substitute
  the lowest positive bin or apply a post-hoc confidence threshold.
- Use each complete scene. Do not silently truncate, chunk, subsample, or
  approximate whole-stream HDBSCAN.
- A 24-hour per-scene wall-clock guard and a 26 GiB process/container RAM cap
  on the 4070 host are resource limits. Timeout/OOM is a reported failure, not
  grounds for a parameter change or exclusion.
- Resume only at completed stage boundaries whose input, config, dependency,
  and source fingerprints match exactly.
- Before any public-data execution, Gate D must record all-pairs
  correspondence count, peak memory and runtime for whole-stream HDBSCAN, and
  peak memory and runtime for COLMAP correspondence preparation on a
  deterministic synthetic scale fixture. The COLMAP evidence must distinguish
  auxiliary all-pairs materialization/raw-SQLite insertion from an actual run
  of the pinned COLMAP 3.9.1 `matches_importer --match_type raw`; only the
  latter demonstrates geometric verification. Accepted reference evidence
  requires Linux `/proc` process-tree measurements for both the host workers
  and the importer inside the sidecar. Mapper runtime is outside this N6
  measurement. No measurement permits external chunking, subsampling, or
  approximation.
- The accepted full-scale N6 fixture is frozen at 168,960 HDBSCAN input
  events in 32 deterministic separated clusters and 128 COLMAP views by 512
  complete tracks. This produces all `128 choose 2` image pairs and exactly
  4,161,536 raw correspondence rows. Sample each complete Linux `/proc`
  process tree every 10 ms, including the in-container importer tree. These
  dimensions and the sampling interval cannot be changed after observing the
  accepted run; a changed scale is separately labelled evidence and cannot
  replace it.
- N6 reference eligibility additionally requires a clean recorded Git commit,
  the pinned sidecar image/revision/version/binary SHA-256, exact fixture and
  output hashes, and successful raw/verified database validation. The Gate D
  report must record host CPU model, physical and logical core counts, total
  RAM, OS/kernel, and separate wall-time/peak-RSS results for HDBSCAN,
  auxiliary raw-SQLite preparation, and the actual pinned importer.
- Require the preregistered semantic fixture hashes: integrated synthetic
  `499c760fa35a0a1ec00f824065337115eec421f54c8cf144809d47e7fb05566f`,
  N6 HDBSCAN
  `6e9bc65b0375506f96e69eb766c4b1384fca6f4e417e837f38445dfc85ee9263`,
  and N6 COLMAP
  `f9dc9f77434273f8470a42988e815cfd032238d9d2cedb3ee35eb15bffdd1df6`.
  Their identity schema quantizes floating values to `1e-9` only when
  serializing the cross-platform semantic hash. All method execution retains
  the original unquantized arrays; semantic hashing must not alter eFAST,
  HDBSCAN, COLMAP, FFT, geometry, GTSAM, or numerical tolerances.
- Record effective `hdbscan` values including `algorithm`,
  `core_dist_n_jobs`, `max_cluster_size`, and `p`, and verify repeated
  identical synthetic input produces identical labels.
- Classify a COLMAP failure as exactly one of: method outcome failure (the
  pinned method runs but yields no usable model), validator/postcondition
  failure (output is malformed or inconsistent with the frozen contract), or
  execution/environment failure (the pinned method cannot execute, including
  dependency/version and subprocess failures). Timeout and OOM remain
  separately named cross-cutting resource failures. The category is for
  reporting only and never authorizes retry, fallback, or selective exclusion.
- Report merged-track and cluster-pair counts whose original HDBSCAN time
  spans intersect. This is a diagnostic of the frozen weak-component rule,
  not a basis for filtering or changing association.
- If direct GTSAM reports an indeterminate/singular linear system, count a
  reference-path failure. Do not add a prior or change solver/variables as a
  recovery action.
- Before orbit optimization, freeze and record the D-009 factor set: total
  factors; factors in registered and unregistered logical windows; factors
  present and absent from selected-model point tracks; missing track/window
  pairs; and split, merged, or uninitialized tracks excluded for lack of a
  unique point initialization. Record exclusion categories independently
  because split and merged categories can overlap. Do not prune by residual,
  depth, or outcome.
- Record non-positive-depth factor counts at the initial and final optimizer
  values. During LM, retain such a factor with the pinned GTSAM 4.2 constant
  `(2*fx,2*fx)` residual and zero Jacobians; never abort the whole run merely
  because a trial step crosses depth zero. If the final count is nonzero,
  record a reference-path failure and mark all reprojection metrics
  unavailable. Do not treat the constant penalty as a predicted pixel or mix
  it into Fig. 10-style residual distributions.

## Permitted public-TOPSPIN outputs

- processing success/failure and failure reason;
- corner, cluster, track, registered-camera, and reconstructed-point counts;
- orbit-factor counts with the D-009 registered/unregistered,
  selected-track present/absent, and excluded-track breakdowns;
- internal reprojection residuals before and after orbit refinement;
- circular-trajectory diagnostics;
- within-category frequency consistency, explicitly not accuracy;
- track-age distributions corresponding to the internal claim in Fig. 8b;
- COLMAP-versus-eSfO reprojection residual distributions corresponding to the
  internal claim in Fig. 10;
- qualitative orbit, sparse point cloud, and screw-line figures;
- runtime, peak RAM, dependency versions, input hashes, and Git identity.

## Unavailable outputs

Unless independently authenticated assets are supplied, reference evaluation
must refuse to compute:

- frequency MAE/RMSE against numeric turntable GT;
- CAD ICP RMSE in millimetres;
- numeric rotation-axis angular error (the paper makes only a qualitative
  axis claim and public TOPSPIN has no authenticated axis ground truth);
- exact numerical reproduction of plot-only paper figures.

## Reporting language

If the frozen implementation does not reproduce paper behavior, report a
**reproduction failure of the frozen primary interpretation**. Do not call it
a refutation of the paper. If required artifacts are missing, report the claim
as **Unavailable**, not failed.

All transformed or redistributed TOPSPIN-derived artifacts must retain
attribution required by the official `CC-BY-4.0` record license. Raw archives
are not committed to this code repository.

