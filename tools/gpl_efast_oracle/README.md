# Isolated GPL eFAST differential oracle

The reference Python package does **not** contain or import GPL source code.
This directory documents an optional audit which compiles the cited upstream
implementation from an external checkout at commit
`e6c88625479a62e649ff391e2489ef6470955411` and compares its boolean decisions
with the independent implementation in `esfo/efast.py`.

The upstream repository is GPL-licensed:
<https://github.com/uzh-rpg/rpg_corner_events>.  Obtain it separately, retain
its license, and do not redistribute the temporary oracle binary as part of
this package.  The regeneration script verifies the pinned Git commit and
SHA-256 identities before compiling the external file.  Locally generated
ROS/message/matrix stubs only expose the `FastDetector::isFeature` criterion;
they contain no copied upstream algorithm.

```bash
python scripts/regenerate_efast_oracle.py \
  --upstream-dir /path/to/rpg_corner_events \
  --output /tmp/efast_upstream_oracle.json
```

The committed JSON contains only deterministic input-generator metadata and
the resulting boolean decision mask/hash.  Normal CI regenerates the inputs,
runs the independent Python detector, and checks that mask.  Recreating the
golden mask requires the external GPL checkout and a C++17 compiler; it is not
part of the installed wheel or normal test dependency graph.
