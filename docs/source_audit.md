# Primary-source and public-artifact audit

Status: preimplementation audit, 2026-11-24.

## Paper identity

- Ethan Elms, Yasir Latif, Tae Ha Park, and Tat-Jun Chin,
  *Event-based Structure-from-Orbit*, CVPR 2024, pp. 19541-19550.
- Official CVF PDF:
  <https://openaccess.thecvf.com/content/CVPR2024/papers/Elms_Event-based_Structure-from-Orbit_CVPR_2024_paper.pdf>
- Audited CVF PDF: 10 pages; SHA-256
  `70ac861187cb75d840e40ef0f67c5392cf9587d15f33f5ea42022b4df9df6f21`.
- arXiv record: <https://arxiv.org/abs/2405.06216>.
- Official arXiv source archive: <https://export.arxiv.org/e-print/2405.06216>;
  SHA-256
  `b65ac9cffdb71ee28df9ad4345046be54eedcfe73325ed197f54cb113bd6d293`.
- `main.tex` in that archive has SHA-256
  `f8a46454572ee71b3a77e1af633f3d384668990475913fcfa72108b3a605375f`.
  The archive contains the paper TeX, bibliography, and figures, but no source
  code, numerical result tables, plotting scripts, or supplementary document.

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
- The Zenodo metadata exposes no explicit rights/license field. Downloading
  the public record does not authorize this repository to redistribute the
  data or to assert a license that the record does not state.
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

### HDBSCAN

The paper cites McInnes et al. but names no implementation or version. Table 2
states minimum cluster size `minPts=10` and selection epsilon `5`. Mapping
these words to a particular API (for example `min_cluster_size=10`,
`cluster_selection_epsilon=5`, and an unspecified `min_samples`) and scaling
the `(t,x,y)` coordinates are **Ambiguous** decisions that must be frozen
before public-data execution.

### COLMAP

The paper cites COLMAP but gives no version, camera model, database-injection
procedure, mapper or bundle-adjustment options, fixed-intrinsics policy, or
failure/retry behavior. TOPSPIN calibration contains distortion coefficients
while the paper equations use a pinhole `K`; the undistortion stage is not
specified. These choices remain **Ambiguous**, not tunable by TOPSPIN results.

### Factor graph / GTSAM

The paper cites factor-graph literature but gives no GTSAM version, exact
variable manifolds, factor/noise model, priors or gauge policy, optimizer,
termination criteria, or numerical safeguards. A direct, version-pinned GTSAM
path is the most auditable reference implementation, but its missing choices
remain **Ambiguous**. A custom solver can only be an equivalence-qualified
alternative on a preregistered deterministic problem; it cannot silently
replace the reference path.

## Directly auditable paper details

| Item | Location | Auditable statement |
|---|---|---|
| Orbit translation | Sec. 3, Eq. (1) | Circular camera center parameterized by `r`, `f`, `u`, `v=n x u`, and `c^w`. |
| Density score | Sec. 4.1, Eq. (2) | Count same-polarity neighbors within distance `lambda`, divide by `lambda`, and reject below the mean for that polarity. |
| HDBSCAN | Sec. 4.2 | Cluster remaining corners in spatio-temporal space while disregarding polarity. |
| Head-tail association | Sec. 4.2, Eq. (3) | Mean first/last `N_sigma` `(t,x,y)` descriptors and nearest forward hemisphere gate. |
| Track extraction | Sec. 4.3, Eq. (4) | Mean cluster pixel location in fixed temporal windows. |
| SfM initialization | Sec. 5.1, Eq. (5) | Use track observations as camera/point correspondences in COLMAP. |
| Plane/circle initialization | Sec. 5.1, Eqs. (6)-(8) | Fit a plane and then a circle to COLMAP camera centers. |
| Frequency initialization | Sec. 5.1, Eqs. (9)-(10) | FFT of windowed mean event x coordinate; select the highest-power frequency. |
| Orbit refinement | Sec. 5.2, Eqs. (11)-(14) | Replace free camera poses by orbit-governed poses in reprojection-error optimization. |
| Constants | Sec. 7.1, Table 2 | `lambda=7`, `minPts=10`, epsilon `5`, `phi=30`, `N_sigma=5`, `delta_t=30 ms`, `delta_tf=20 ms`. |
| Dataset | Sec. 6, Table 1 | Six objects, three speed categories, four viewpoints: 72 scenes. |

## Source limitations

The paper does not uniquely specify the time-to-pixel scale in `(t,x,y)`, the
distance domain of Eq. (2), exact eFAST implementation, HDBSCAN API mapping,
cluster-merge conflict policy, COLMAP version/settings, FFT preprocessing,
factor-graph variables/priors/noise/solver settings, CAD alignment protocol,
or several aggregation conventions. These remain Ambiguous unless an audited
primary citation resolves them.

The TOPSPIN artifacts also lack authoritative numeric fast/medium/slow
frequencies, camera-to-turntable extrinsics or rotation-axis ground truth,
authors' CAD meshes and manual transforms, feature tracks, COLMAP
initialization, and per-scene expected outputs. Frequency MAE/RMSE, axis-angle
error, and CAD RMSE in millimetres are therefore **Unavailable** from the
public release. Category consistency, reconstruction diagnostics, and
qualitative plots may be **Exploratory**, but they are not substitutes for
those unavailable metrics.

