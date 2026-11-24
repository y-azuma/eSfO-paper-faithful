# eSfO paper-faithful reproduction

Independent reproduction of **Event-based Structure-from-Orbit** (Elms,
Latif, Park, and Chin, CVPR 2024).

## Status

**Gate 0: evidence and interpretation review.** The implementation is not yet
claimed complete and no public-TOPSPIN performance result is accepted as a
paper reproduction result. The authors' paper-listed code URL and the
supplementary material referenced by the paper are currently unavailable.

This repository separates five evidence classes:

| Class | Meaning |
|---|---|
| **Exact** | The paper or an audited primary citation specifies enough detail to implement the operation. |
| **Ambiguous** | A consequential convention or value is missing; a result-independent primary interpretation is frozen and documented. |
| **Unavailable** | Public artifacts do not contain the assets needed for the claimed metric. |
| **Exploratory** | Diagnostics on already-inspected public data; not confirmatory evidence. |
| **Experimental** | Paper-unreported improvements, excluded from the reference path and paper-comparison outputs. |

The governing documents are:

- [`docs/source_audit.md`](docs/source_audit.md)
- [`docs/traceability_matrix.md`](docs/traceability_matrix.md)
- [`docs/decision_log.md`](docs/decision_log.md)
- [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md)

No interpretation may be changed because it improves TOPSPIN performance.

## Intended reference pipeline

1. eFAST corner-event detection.
2. Same-polarity local-density filtering (paper Eq. 2).
3. Polarity-agnostic spatio-temporal HDBSCAN.
4. Forward head-tail cluster association (paper Eq. 3).
5. Windowed feature-track extraction (paper Eq. 4).
6. COLMAP initialization from known feature correspondences (paper Eq. 5).
7. Plane and circle fitting to COLMAP camera centers (paper Eqs. 6-8).
8. FFT frequency initialization from the mean event x coordinate (paper Eqs. 9-10).
9. Orbit-constrained factor-graph refinement (paper Eqs. 11-14).

The implementation will be added only after the preimplementation decisions
have been independently reviewed by Codex and Claude Code.

