# eSfO paper-faithful reproduction

Independent reproduction of **Event-based Structure-from-Orbit** (Elms,
Latif, Park, and Chin, CVPR 2024).

## Status

**Gate D: integrated synthetic evidence and instrumentation in progress.**
Gates A--C are mutually approved after independent Codex/Claude Code review.
Claude Code's focused re-review of final commit
`9313c6da48399432e06cc080f07f832574176aae` approved the sparse-view
temporal-sign correction (B1-prime) and LM trial-step cheirality correction
(C1), gave final approval to Gates B and C, and authorized Gate D. The Sec. 4
front end, Gate B initialization, and Gate C direct-GTSAM factor are
implemented and tested only on synthetic fixtures; no public-TOPSPIN
performance result is accepted as a paper reproduction result. Gate D's
result-independent N1, N3, and N4 closures are recorded in the governing
documents, while N5/N6 and the integrated Gate D evidence remain unfinished.
Public TOPSPIN execution is still prohibited.
The authors' paper-listed code URL and the supplementary material referenced
by the paper are currently unavailable.

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
- [`docs/geometry_conventions.md`](docs/geometry_conventions.md)
- [`docs/gate_a_report.md`](docs/gate_a_report.md)
- [`docs/gate_b_report.md`](docs/gate_b_report.md)
- [`docs/gate_c_report.md`](docs/gate_c_report.md)
- [`docs/open_review_items.md`](docs/open_review_items.md)

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

The first Gate 0 review of commit `7e498ee` requested documentation changes.
Claude Code conditionally approved the revised interpretation at `6826b9f`
and authorized Gate A. It subsequently approved the `d57e63f` convention
follow-up and the `dffc27b` Gate A implementation with no blocking findings,
authorizing Gate B. Public-data execution remains blocked by the later gates.

