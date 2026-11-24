# Frozen evaluation protocol (draft)

## Objectives

The evaluation asks whether the frozen primary interpretation executes and
whether the paper's publicly auditable claims can be reproduced. It does not
optimize the method.

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
- Use each complete scene. Do not silently truncate, chunk, subsample, or
  approximate whole-stream HDBSCAN.
- A 24-hour per-scene wall-clock guard and a 26 GiB process/container RAM cap
  on the 4070 host are resource limits. Timeout/OOM is a reported failure, not
  grounds for a parameter change or exclusion.
- Resume only at completed stage boundaries whose input, config, dependency,
  and source fingerprints match exactly.
- Before any public-data execution, Gate D must record all-pairs
  correspondence count, peak memory and runtime for whole-stream HDBSCAN, and
  peak memory and runtime for COLMAP injection on a deterministic synthetic
  scale fixture. This measurement does not permit chunking or approximation.
- Record effective `hdbscan` values including `algorithm`,
  `core_dist_n_jobs`, `max_cluster_size`, and `p`, and verify repeated
  identical synthetic input produces identical labels.
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

