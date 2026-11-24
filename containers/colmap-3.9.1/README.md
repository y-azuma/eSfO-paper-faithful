# COLMAP 3.9.1 CPU sidecar

This Ubuntu 22.04 image builds the exact upstream COLMAP 3.9.1 tag commit used by the
frozen eSfO reproduction protocol:

```text
e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68
```

CUDA, the GUI, OpenGL, and upstream tests are disabled because Gate B injects
known event-track keypoints and uses only geometric verification plus the
incremental mapper. Disabling them is an execution-environment choice, not a
method change.

Build on the 4070 Ubuntu host from the repository root:

```bash
docker build \
  --file containers/colmap-3.9.1/Dockerfile \
  --tag esfo-colmap:3.9.1-cpu .
```

The Gate B manifest must record the resulting image ID/repository digest,
`sha256sum /usr/local/bin/colmap`, `ldd /usr/local/bin/colmap`, compiler and
linked dependency versions. The image is not accepted merely because its
version string contains `3.9.1`; the integration test also checks database,
geometric-verification, fixed-intrinsics, and mapper postconditions.
