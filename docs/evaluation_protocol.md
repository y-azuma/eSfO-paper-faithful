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
- Use a fixed small scene set for implementation review only.
- Run all 72 scenes once after Gates 0-D and both reviewers approve.
- Include every failure in the denominator and record the first failing stage.
- Do not change reference settings after observing results.

## Permitted public-TOPSPIN outputs

- processing success/failure and failure reason;
- corner, cluster, track, registered-camera, and reconstructed-point counts;
- internal reprojection residuals before and after orbit refinement;
- circular-trajectory diagnostics;
- within-category frequency consistency, explicitly not accuracy;
- qualitative orbit, sparse point cloud, and screw-line figures;
- runtime, peak RAM, dependency versions, input hashes, and Git identity.

## Unavailable outputs

Unless independently authenticated assets are supplied, reference evaluation
must refuse to compute:

- frequency MAE/RMSE against numeric turntable GT;
- CAD ICP RMSE in millimetres;
- rotation-axis angular error;
- exact numerical reproduction of plot-only paper figures.

## Reporting language

If the frozen implementation does not reproduce paper behavior, report a
**reproduction failure of the frozen primary interpretation**. Do not call it
a refutation of the paper. If required artifacts are missing, report the claim
as **Unavailable**, not failed.

