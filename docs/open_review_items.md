# Open independent-review items

This register prevents a non-blocking review finding from being mistaken for
an accepted implementation detail.  No item may be closed from TOPSPIN
performance.  Closure requires the listed result-independent evidence and an
update to the decision log.

## Focused re-review queue

| ID | Finding | Current state | Required closure |
|---|---|---|---|
| B1-prime | The temporal-sign chord was referenced to the sample centroid, which can lie on the wrong side of a sparse chord. | Implemented: fit the circle before choosing the plane-normal sign and use `q_k=(I-nn^T)(C_k-c^w)`; deterministic sparse-view and transformed-view fixtures and Ubuntu CI run `45372999990` pass. | Focused Claude Code confirmation before integrated Gate D evidence. |
| C1 | Raising on a non-positive trial depth aborts the complete GTSAM LM call. | Implemented: preserve the factor and return GTSAM 4.2's default `(2*fx,2*fx)` residual with zero Jacobians; record initial/final counts; real negative-depth LM-trial fixture and Ubuntu CI run `45372999990` pass. | Focused Claude Code confirmation before integrated Gate D evidence. |

## Non-blocking items required before public-data execution

| ID | Origin | Current state | Closure gate | Required evidence |
|---|---|---|---|---|
| N1 | Gate B review | An all-zero positive-frequency FFT spectrum currently selects the lowest positive bin and records `zero_positive_power=true`. This remains a frozen but questioned convention. | Gate D | Decide, without public data, whether exact zero power must fail as `fft_frequency_unobservable`; update D-008, metadata, focused tests, and reports. If retained, document the consistency argument and required Gate E reporting. |
| N3 | Gate B review | COLMAP two-view postconditions are tested, but their pinned upstream source locations are not cited and validator failures are not yet a separate report category. | Gate D | Audit commit `e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68`, add immutable source permalinks for the enum/default/inlier rules, and distinguish validator failure from method failure. |
| N4 | Gate A/B review | Claude Code independently compared 171,761 eFAST decisions with the cited implementation and found zero differences, but that differential oracle is not in the suite. | Gate D | Add a legally and reproducibly isolated, pinned differential test without silently copying GPL code into the reference package; document its license boundary. |
| N5 | Gate B/C review | Component diagnostics exist for overlapping source clusters, temporal phase gaps, mapping exclusions, unregistered windows, D-009 membership, and cheirality. They are not yet aggregated in one stage manifest. | Gate D instrumentation; Gate E reporting | Aggregate all counts without using them for filtering; add registered-window ratio and initial/final non-positive-depth counts; report every scene in Gate E. |
| N6 | Gate B review | The scale/resource benchmark is preregistered but not yet executed. | Gate D | On a deterministic synthetic scale fixture, record all-pairs correspondence rows plus runtime and peak RAM for whole-stream HDBSCAN and COLMAP injection. Chunking or approximation is not permitted. |

Gate E and every TOPSPIN run remain blocked until the applicable Gate D
closures and both reviewers' approval are recorded.
