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

## Public artifacts

- TOPSPIN record: <https://zenodo.org/records/10884694>.
- Paper-listed code URL: <https://github.com/0thane/eSfO>.
  It returned HTTP 404 during the 2026-11-24 audit. This establishes current
  unavailability, not that code was never public.
- The paper refers to supplementary material, but the current CVF record has
  no supplement link and the expected CVF supplement URL returned HTTP 404.
  This establishes current unavailability only.

The public TOPSPIN description lists, for each of 72 scenes, `events.csv`,
`calib.txt`, and `meta.json`. It does not authenticate the numeric turntable
frequencies, authors' CAD models, manual CAD alignments, rotation-axis ground
truth, or per-scene author estimates used by the paper.

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

