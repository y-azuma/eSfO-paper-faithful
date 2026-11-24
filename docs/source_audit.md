# Primary-source and public-artifact audit

Status: living source audit through Gate D implementation, 2026-11-24.

## Paper identity

- Ethan Elms, Yasir Latif, Tae Ha Park, and Tat-Jun Chin,
  *Event-based Structure-from-Orbit*, CVPR 2024, pp. 19541-19550.
- Official CVF PDF:
  <https://openaccess.thecvf.com/content/CVPR2024/papers/Elms_Event-based_Structure-from-Orbit_CVPR_2024_paper.pdf>
- Audited CVF PDF: 10 pages; SHA-256
  `70ac861187cb75d840e40ef0f67c5392cf9587d15f33f5ea42022b4df9df6f21`.
- arXiv record: <https://arxiv.org/abs/2405.06216>.
- Audited arXiv v1 PDF: 10 pages; SHA-256
  `7ad59e3495d37a0772e8d2274db9202a0c1601f1434779aaa4ad3b826d6caa00`.
- Official arXiv source archive: <https://export.arxiv.org/e-print/2405.06216>;
  SHA-256
  `b65ac9cffdb71ee28df9ad4345046be54eedcfe73325ed197f54cb113bd6d293`.
- `main.tex` in that archive has SHA-256
  `f8a46454572ee71b3a77e1af633f3d384668990475913fcfa72108b3a605375f`.
  The archive contains the paper TeX, bibliography, and figures, but no source
  code, numerical result tables, plotting scripts, or supplementary document.

The CVF PDF and arXiv v1 PDF are different files, so every review records which
version it quoted. For the Gate 0 cross-review, Claude Code quoted the attached
arXiv v1 PDF. Codex compared both complete PDFs after raw text extraction,
Unicode NFKC normalization, and removal of the CVF watermark, arXiv header, and
CVF page labels. The only residual tokens were arXiv page numbers `2` through
`10`; the method prose, Eqs. (1)-(14), tables, and references were otherwise
identical. The arXiv v1 PDF is therefore sufficient for method review, while
both hashes remain part of the provenance record.

Evidence in this repository is classified as **Exact**, **Ambiguous**,
**Unavailable**, **Exploratory**, or **Experimental**. A public artifact's
absence is recorded as Unavailable; it is never filled with an inferred value.

## Public artifacts

- TOPSPIN version record: <https://zenodo.org/records/10884694>, DOI
  `10.5281/zenodo.10884694`; paper-cited concept DOI
  `10.5281/zenodo.10884693`.
- The official record contains 72 `.tar.gz` archives, exactly matching six
  objects by three speed categories by four viewpoints, with total compressed
  size `10,123,279,125` bytes. Each archive contains only `events.csv`,
  `calib.txt`, and `meta.json`; timestamps are documented in microseconds.
  Acquisition must verify the record's per-file sizes and MD5 values.
- The official Zenodo API reports `metadata.license.id = cc-by-4.0`. Its
  separate `metadata.rights` field is null; that null is not an absence of a
  license. TOPSPIN and any distributed derivatives must retain the attribution
  required by CC BY 4.0. The repository does not redistribute the archives.
- Paper-listed code URL: <https://github.com/0thane/eSfO>.
  It returned HTTP 404 during the 2026-11-24 audit. Wayback captures at
  `20240513055605` and `20240513073400` both show repository ID `778107231`
  and the message "This repository is empty." No files, commits, branches, or
  releases were recoverable. The auditable evidence therefore shows an empty
  public repository at those two publication-time snapshots and current
  unavailability; it does not prove that code was never uploaded transiently
  outside the captured dates.
- The paper refers to supplementary material, but the current CVF record has
  no supplement link and the expected CVF supplement URL returned HTTP 404.
  The official arXiv source has only a commented `% \\input{sec/X_suppl}` and
  no corresponding source. Wayback has no preserved successful response for
  the expected CVF supplement URL. This establishes current unavailability
  only.

The public TOPSPIN description lists, for each of 72 scenes, `events.csv`,
`calib.txt`, and `meta.json`. It does not authenticate the numeric turntable
frequencies, authors' CAD models, manual CAD alignments, rotation-axis ground
truth, or per-scene author estimates used by the paper.

## Audited cited implementations

### eFAST

- The cited official implementation candidate is
  <https://github.com/uzh-rpg/rpg_corner_events>.
- Its latest commit before eSfO publication is
  `e6c88625479a62e649ff391e2489ef6470955411` (2019-09-18).
- It is GPL-licensed ROS/C++ research code. The eSfO paper does not pin this
  revision or establish that the authors used this exact build.

Using that pinned upstream implementation under its license is auditable, but
still **Ambiguous** with respect to the authors' actual build. An independent
criterion reimplementation must also remain Ambiguous and must not claim
bitwise identity without author code or golden outputs.

For N4, a legally isolated audit compiles the pinned GPL implementation only
from an external checkout and commits only deterministic generator metadata
and its boolean decision-mask hash. The test covers 13,000 event decisions,
of which 1,000 are positive corner decisions. The installed reference package
does not contain or import GPL implementation source. This committed oracle is
separate from Claude Code's earlier 171,761-decision independent scratchpad
comparison; the smaller reproducible fixture neither replaces that audit nor
retroactively packages it.

### HDBSCAN

The paper cites McInnes et al. but names no implementation or version. Table 2
states minimum cluster size `minPts=10` and selection epsilon `5`. Mapping
these words to a particular API (for example `min_cluster_size=10`,
`cluster_selection_epsilon=5`, and an unspecified `min_samples`) and scaling
the `(t,x,y)` coordinates are **Ambiguous** decisions that must be frozen
before public-data execution. The frozen reference selects
`hdbscan==0.8.33`, the latest stable non-RC release available before arXiv v1
on 2024-05-10. This common version cutoff is an audited interpretation, not
evidence of the authors' version.

### COLMAP

The paper cites COLMAP but gives no version, camera model, database-injection
procedure, mapper or bundle-adjustment options, fixed-intrinsics policy, or
failure/retry behavior. TOPSPIN calibration contains distortion coefficients
while the paper equations use a pinhole `K`; the undistortion stage is not
specified. These choices remain **Ambiguous**, not tunable by TOPSPIN results.
The frozen reference selects COLMAP `3.9.1`, the latest stable non-RC release
available before arXiv v1 on 2024-05-10; this is not a claim about the authors'
build. Undistortion's new camera matrix and bundle-adjustment refinement flags
are also absent from the paper and therefore frozen interpretations.

The COLMAP behavior used by the reference path is pinned to immutable commit
`e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68` at these primary-source ranges:

- [two-view configuration enum (`scene/two_view_geometry.h`, lines 40--58)](https://github.com/colmap/colmap/blob/e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68/src/colmap/scene/two_view_geometry.h#L40-L58);
- [two-view estimator defaults (`estimators/two_view_geometry.h`, lines 42--106)](https://github.com/colmap/colmap/blob/e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68/src/colmap/estimators/two_view_geometry.h#L42-L106);
- [two-view method selection (`estimators/two_view_geometry.cc`, lines 420--554)](https://github.com/colmap/colmap/blob/e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68/src/colmap/estimators/two_view_geometry.cc#L420-L554); and
- [watermark-pair exclusion while loading the database (`scene/database_cache.cc`, lines 72--87)](https://github.com/colmap/colmap/blob/e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68/src/colmap/scene/database_cache.cc#L72-L87).

N3 freezes three failure categories. A **method outcome failure** means the
pinned method executed but returned no usable result (for example no raw or
verified pair, no model, or insufficient registered views). A
**validator/postcondition failure** means an output exists but is malformed or
inconsistent with the frozen contract (for example pair/configuration/inlier
count mismatch, invalid duplicate verified matches, a changed fixed camera,
or inconsistent model tracks). An **execution/environment failure** means the
method could not execute as pinned (for example an unavailable or
version-mismatched dependency or nonzero subprocess exit). Timeout and OOM
remain separately named cross-cutting resource failures. These categories are
reporting labels only: none authorizes a retry, alternate method, fallback, or
selective exclusion.

### Factor graph / GTSAM

The paper cites factor-graph literature but gives no GTSAM version, exact
variable manifolds, factor/noise model, priors or gauge policy, optimizer,
termination criteria, or numerical safeguards. A direct, version-pinned GTSAM
path is the most auditable reference implementation, but its missing choices
remain **Ambiguous**. A custom solver can only be an equivalence-qualified
alternative on a preregistered deterministic problem; it cannot silently
replace the reference path.

The printed Eq. (5) sums over all windows and points, while Eq. (4) defines no
measurement for an empty track/window pair. Sec. 5.2 instead refers to every
observation and to cameras where a point is visible. The frozen reference
therefore interprets visibility as the existence of an original Eq. (4)
observation. For each source track with a unique COLMAP point initialization,
all such fixed-undistorted observations enter the orbit graph, even when the
window is unregistered or the feature is absent from the selected COLMAP
point track. No missing measurement is fabricated, and non-bijective or
uninitialized tracks are excluded. This remains **Ambiguous**, is frozen
before Gate C, and cannot be changed based on public-data residuals.

The frozen reference selects GTSAM `4.2`, the latest stable non-RC release
available before arXiv v1 on 2024-05-10. The same cutoff selects
`hdbscan==0.8.33` and COLMAP `3.9.1`; none of these choices is evidence of the
authors' environment.

GTSAM 4.2's primary
[`GenericProjectionFactor` source](https://github.com/borglab/gtsam/blob/4f66a491ffc83cf092d0d818b11dc35135521612/gtsam/slam/ProjectionFactor.h#L153-L163)
sets projection Jacobians to zero and returns
`Vector2::Constant(2.0*K_->fx())` when cheirality fails and
`throwCheirality=false` (the default). The frozen custom orbit factor mirrors
that behavior so a temporary invalid LM trial is rejected or traversed rather
than aborting the whole optimizer. This is an audited library convention,
not a claim that the eSfO authors documented it.

## Directly auditable paper details and evidence boundaries

| Item | Location | Auditable statement | Evidence boundary |
|---|---|---|---|
| Orbit translation | Sec. 3, Eq. (1) | Circular camera center parameterized by `r`, `f`, `u`, `v=n x u`, and `c^w`. | **Exact equation**; coordinate convention remains explicit. |
| Density score | Sec. 4.1, Eq. (2) | Count same-polarity neighbors within distance `lambda`, divide by `lambda`, and reject below the mean for that polarity. | **Ambiguous**: the distance domain and whether `E_i` contains raw events or corner events are not stated consistently. |
| HDBSCAN | Sec. 4.2 | Cluster remaining corners in spatio-temporal space while disregarding polarity. | Mechanism and tabulated parameters are **Exact**; library version and time scaling are **Ambiguous**. |
| Head-tail association | Sec. 4.2, Eq. (3) | Mean first/last `N_sigma` `(t,x,y)` descriptors and nearest forward hemisphere gate. | Mechanism is stated; merge conflicts, ties, and whether descriptors are recomputed are **Ambiguous**. |
| Track extraction | Sec. 4.3, Eq. (4) | Mean cluster pixel location in the displayed `k=1...K` temporal windows. | Displayed indexing is **Exact**; fractional terminal and empty windows are **Ambiguous**. |
| SfM initialization | Sec. 5.1, Eq. (5) | Use track observations as camera/point correspondences in COLMAP. | High level is stated; version, distortion, correspondence injection, validation, and model selection are **Ambiguous**. |
| Plane/circle initialization | Sec. 5.1, Eqs. (6)-(8) | Fit a plane and then a circle to COLMAP camera centers. | **Ambiguous**: printed algebra does not by itself yield the stated normal and omits centering, sign, projection, normalization, and degeneracy conventions. |
| Frequency initialization | Sec. 5.1, Eqs. (9)-(10) | FFT of windowed mean x coordinate of all events; select the highest-power frequency. | High level is stated; empty windows and FFT edge conventions are **Ambiguous**. |
| Orbit refinement | Sec. 5.2, Eqs. (11)-(14) | Replace free camera poses by orbit-governed poses in reprojection-error optimization. | **Ambiguous**: literal transform composition does not recover Eq. (1), and variable manifolds, factor/noise model, gauge, and solver are not stated. |
| `R0` initialization | Sec. 5 | No initialization rule is given. | **Not stated**; a result-independent rule must be frozen as an interpretation. |
| Constants | Sec. 7.1, Table 2 | `lambda=7`, `minPts=10`, epsilon `5`, `phi=30`, `N_sigma=5`, `delta_t=30 ms`, `delta_tf=20 ms`. | Values are **Exact**; API and unit mappings can remain **Ambiguous**. |
| Dataset | Sec. 6, Table 1 | Six objects, three speed categories, four viewpoints: 72 scenes. | **Exact** and matched by the official record. |

## Source limitations

The paper does not uniquely specify the time-to-pixel scale in `(t,x,y)`, the
distance domain of Eq. (2), exact eFAST implementation, HDBSCAN API mapping,
cluster-merge conflict policy, COLMAP version/settings, FFT preprocessing,
factor-graph variables/priors/noise/solver settings, CAD alignment protocol,
absolute orbit phase origin, or several aggregation conventions. These remain
Ambiguous unless an audited
primary citation resolves them.

In the frozen reference, `t_ref` is the time in seconds of the earliest
registered view in the largest retained COLMAP model, the same view used to
define `u`. The paper does not state this rule. It is a deterministic phase
gauge chosen to make the initialized orbit pass that camera centre, not a
paper-reported value or a choice made from TOPSPIN performance.

For the frozen primary interpretation, frontend time is expressed as
`t_ms = (t_us - t_1) / 1000` in Eq. (2), HDBSCAN, and Eq. (3). This remains an
**Ambiguous** inference, but it follows the paper's dimensions: both explicit
time-window constants are in milliseconds, and `phi=30` combines time with
pixel distances. Raw microseconds would make the temporal gate roughly
30 microseconds, while seconds would make it roughly 30 seconds.

Sec. 4 describes offline HDBSCAN over the filtered corner stream and reports no
chunking, subsampling, approximation, or temporal segmentation. Whole-stream
clustering is therefore the reference interpretation. Any resource-motivated
partitioning is a consequential paper-unreported change and must remain
**Experimental** unless it passes a separately preregistered equivalence test;
it cannot silently replace the reference path.

The TOPSPIN artifacts also lack authoritative numeric fast/medium/slow
frequencies, camera-to-turntable extrinsics or rotation-axis ground truth,
authors' CAD meshes and manual transforms, feature tracks, COLMAP
initialization, and per-scene expected outputs. Frequency MAE/RMSE, axis-angle
error, and CAD RMSE in millimetres are therefore **Unavailable** from the
public release. Category consistency, reconstruction diagnostics, and
qualitative plots may be **Exploratory**, but they are not substitutes for
those unavailable metrics.

