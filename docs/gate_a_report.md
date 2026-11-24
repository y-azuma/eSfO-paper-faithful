# Gate A report: synthetic front-end verification

Status: **implemented locally; independent review pending**.

Gate A verifies only the Sec. 4 front end on deterministic synthetic inputs.
It does not use TOPSPIN, estimate paper performance, or select an
interpretation from public-data results.

## Implemented reference path

- strict TOPSPIN-style event schema with `0/1` polarity and stable ordering;
- relative frontend time `t_ms=(t_us-t_first_us)/1000`;
- independent eFAST SAE implementation using both published rings and strict
  newest-arc tests (inner lengths 3--6, outer lengths 4--8), with unseen cells
  frozen to negative infinity under the relative-time convention;
- paper Eq. (2) using raw, same-polarity neighbors in `(t_ms,x,y)`, inclusive
  radius 7, and a whole-sequence mean per polarity;
- McInnes `hdbscan==0.8.33` on the complete filtered-corner sequence with all
  frozen effective options recorded and noise label `-1` excluded;
- one-shot cluster descriptors, one nearest forward edge per tail, strict
  `phi=30`, deterministic tie-breaking, and one weak-component merge;
- literal Eq. (4) indexing and half-open/closed windows, with no fabricated
  observation for empty track/window pairs;
- content hashes, configuration/source/input fingerprints, and atomic JSON
  stage-boundary writes.

The eFAST implementation follows the published criterion.  Its ring order,
arc ranges, strict comparison, polarity separation, update order, and border
rule were statically compared with the cited upstream revision
`e6c88625479a62e649ff391e2489ef6470955411`.  GPL source was not copied, no
upstream executable was used as a golden oracle, and bitwise identity with the
unavailable eSfO author build is not claimed.

## Synthetic verification

The frozen Python 3.10 test suite contains 55 tests covering:

- timestamp units, stable equal-timestamp ordering, sensor bounds, and
  fail-closed polarity validation;
- inner/outer eFAST arcs, wraparound, strict timestamp separation, polarity
  separation, SAE initialization, and border behavior;
- Eq. (2) radius boundary, raw-event versus corner queries, time/polarity, and
  global per-polarity thresholds;
- real `hdbscan==0.8.33` parameter metadata, noise handling, and repeated-run
  determinism;
- Eq. (3) direction, strict gate, distance/time/ID ties, fan-in, cycles, and
  input-order-invariant components;
- Eq. (4) first-window omission, boundaries, floor rule, and empty windows;
- configuration separation and provenance/content sensitivity.

Local result on 2026-11-24: `55 passed`; statement coverage `83%`.  Four
warnings originate inside `hdbscan==0.8.33` from a NumPy API deprecated after
that frozen dependency was released; they do not change the test result.

The local reference environment is Python 3.10.6 with `numpy==1.26.4`,
`scipy==1.13.0`, `hdbscan==0.8.33`, `scikit-learn==1.4.2`,
`joblib==1.4.2`, and `threadpoolctl==3.5.0`.  `pip check` reports no broken
requirements.  CI is a pinned dependency smoke test on Ubuntu 24.04; it is
not yet the bit-for-bit container image required for final archival
reproduction.

## Evidence boundary and next gate

Passing Gate A establishes internal conformance to the frozen interpretation,
not accuracy or agreement with hidden author outputs.  Gate B will implement
and test COLMAP initialization, plane/circle fitting, and FFT initialization.
It must not begin until Claude Code has checked the `t_ref` follow-up diff and
reviewed this Gate A implementation.
