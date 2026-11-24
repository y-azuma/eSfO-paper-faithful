# Paper-to-code traceability matrix

Status: **Gate 0 draft; no implementation-fidelity claim yet**.

| Paper item | Required behavior | Evidence class | Frozen primary interpretation (draft) | Validation |
|---|---|---|---|---|
| Sec. 3, Eq. (1) | Fixed-axis constant-frequency circular orbit | Exact equation | Implement the displayed equation with an explicit right-handed frame convention | Analytic poses at 0, quarter, and half period |
| Sec. 4.1 | eFAST corner events from SAE | Ambiguous implementation fidelity | Implement the published eFAST arc criterion; do not claim bitwise identity without an authoritative revision/golden output | Arc, polarity, timestamp-order fixtures |
| Sec. 4.1, Eq. (2) | Same-polarity local density; reject below same-polarity mean | Exact equation / Ambiguous distance | Spatial `(x,y)` Euclidean radius `lambda=7`; raw events are neighbors; query event is included | Boundary and polarity fixtures |
| Sec. 4.2 | Polarity-agnostic spatio-temporal HDBSCAN | Exact mechanism / Ambiguous scale | Timestamp converted to seconds with multiplier `1`; `min_cluster_size=10`, `cluster_selection_epsilon=5`, `min_samples=None` | Deterministic synthetic clusters; no TOPSPIN tuning |
| Sec. 4.2, Eq. (3) | Mean head/tail descriptors and nearest forward gate | Exact mechanism / Ambiguous conflicts | `N_sigma=5`, `phi=30`; deterministic tail-time/distance/ID order; iteratively merge nearest forward candidates | Direction, gate, tie, transitive-merge fixtures |
| Sec. 4.3, Eq. (4) | Mean pixel location in `delta_t=30 ms` windows | Exact mechanism / Ambiguous boundaries | Shift time by first timestamp and implement the displayed `k=1...K` windows literally | Off-by-one and empty-window fixtures |
| Sec. 5.1, Eq. (5) | COLMAP SfM from known tracks | Exact high level / Ambiguous configuration | Pinned COLMAP; fixed known PINHOLE intrinsics; record every non-paper default | Noise-free known-track reconstruction |
| Sec. 5.1, Eq. (6) | Plane from COLMAP camera centers | Exact printed algebra / Ambiguous sign | Literal `z=ax+by+c` least squares and deterministic right-handed normal sign | Known-plane and sign fixtures |
| Sec. 5.1, Eqs. (7)-(8) | Circle fit and derive `r,c^w,u` | Exact printed algebra / Ambiguous normalization | Use standard implicit-circle algebra; normalize `u` because Sec. 3 defines it as a unit vector; treat the starred square as a typesetting error | Known-circle fixtures |
| Sec. 5.1, Eqs. (9)-(10) | FFT of 20 ms mean-x signal; maximum-power frequency | Exact high level / Ambiguous preprocessing | Raw window means, rectangular window, `rfft`, omit DC from maximization, no zero padding, interpolation, search prior, or harmonic correction | Exact-bin and harmonic behavior fixtures |
| Sec. 5.2, Eqs. (11)-(14) | Orbit-constrained reprojection factors | Exact equations / Ambiguous optimization | Direct GTSAM path; plain Gaussian pixel residual; no unreported robust loss; all conventions and necessary gauge priors explicit | Jacobians, zero-residual synthetic recovery, gauge-aware comparison |
| Sec. 7.2, Tables 3-4 | Front-end benchmark on Event Camera Dataset | Ambiguous protocol | Separate benchmark; do not claim reproduction until cited protocol and assets are audited | Dataset/protocol provenance gate |
| Sec. 7.3, Fig. 8 | Frequency by categorical speed | Unavailable authoritative metric | Report same-speed consistency and figure-proxy separately; never call either GT accuracy | Fail closed without authenticated numeric GT |
| Sec. 7.3, Fig. 9 | CAD registration RMSE | Unavailable | Do not compute paper-comparison RMSE without authors' CAD and alignment provenance | Fail closed without authenticated CAD |
| Sec. 7.3, Fig. 11 | Screw-line projection | Qualitative | Render only; do not invent angular-error GT | Visual artifact with provenance |

No row may be promoted from Ambiguous to Exact because it performs well.

