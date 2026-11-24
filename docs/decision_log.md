# Paper-faithful reproduction decision log

Status: **Gate 0 draft, awaiting Claude Code and Codex agreement**  
Scope: eSfO reproduction, not a method-improvement project.

## D-001 - Reproduction boundary

- Implement the paper and audited primary citations as literally as available
  evidence permits.
- Do not select an interpretation by TOPSPIN accuracy.
- Put harmonic correction, learned features, tuned time scaling, robust losses,
  defensive recovery, and alternative association/optimization schemes in a
  mechanically separate experimental namespace.
- A poor result is not permission to modify the reference path.

## D-002 - Evidence hierarchy

Resolve ambiguities in this order:

1. displayed equation or explicit paper prose;
2. paper figure or table;
3. audited primary citation;
4. established implementation convention;
5. a simple deterministic convention frozen before public-data execution.

Performance is not an admissible tie-break.

## D-003 - Current primary interpretations requiring cross-review

1. Eq. (2) uses spatial `(x,y)` distance with the query event included.
2. Spatio-temporal coordinates use seconds with multiplier `1`.
3. HDBSCAN maps `minPts=10` to `min_cluster_size=10` and leaves
   `min_samples=None`.
4. Head-tail merge uses deterministic sequential nearest-forward association
   until no merge remains.
5. Eq. (4) follows the displayed `k=1...K` windows literally after subtracting
   the first timestamp, even though this omits the first 30 ms.
6. COLMAP uses a pinned version, fixed known PINHOLE intrinsics, and recorded
   default mapper/BA settings.
7. Plane/circle algebra is literal; normal sign is deterministic; `u` is
   normalized to honor its Sec. 3 unit-vector definition.
8. FFT uses raw 20 ms means, a rectangular window, real FFT, DC excluded only
   from peak selection, no zero padding/interpolation/range prior/harmonic fix.
9. The reference optimizer uses direct GTSAM with plain Gaussian pixel
   residuals. Any necessary gauge prior is explicit and reported; an alternate
   solver requires predeclared numerical-equivalence thresholds.

These are interpretations, not paper-reported facts. They remain frozen after
implementation begins unless stronger source evidence or a reproducible code
defect is recorded and cross-reviewed.

## D-004 - Public-data evidence boundary

- Public TOPSPIN has already been inspected in earlier work. It is not held
  out and results are exploratory/retrospective.
- Numeric frequency GT, authors' CAD, CAD alignment, and axis GT are currently
  unavailable. Corresponding paper metrics must fail closed.
- Fast/Medium/Slow consistency is not accuracy.
- Values digitized from paper plots are figure proxies, not authoritative GT.

## D-005 - Review gates

1. **Gate 0:** sources, matrix, and primary interpretations mutually reviewed.
2. **Gate A:** front-end component tests on synthetic fixtures only.
3. **Gate B:** COLMAP, plane/circle, and FFT initialization tests.
4. **Gate C:** direct-GTSAM factor and gauge-aware synthetic recovery.
5. **Gate D:** integrated synthetic run, provenance, checkpoint/resume, and
   fail-closed unavailable metrics.
6. **Gate E:** fixed implementation-review scenes, then one all-72 run only
   after Claude Code and Codex approve A-D.

## Change log

| Date | Decision | Evidence | Agreed by |
|---|---|---|---|
| 2026-11-24 | Repository and Gate 0 draft created; no implementation or result accepted yet. | CVPR paper and public-artifact audit | Codex draft; Claude review pending |

