FROM ubuntu:22.04@sha256:b000e69a50ff940390233025e7a337fb8469b509e2334fa4d018ac30f3e18cdb

ARG DEBIAN_FRONTEND=noninteractive
ARG COLMAP_COMMIT=e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libcgal-dev \
        libceres-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libgl1-mesa-dev \
        libglew-dev \
        libgoogle-glog-dev \
        libmetis-dev \
        libsqlite3-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/colmap/colmap.git /opt/colmap-src \
    && git -C /opt/colmap-src checkout --detach "${COLMAP_COMMIT}" \
    && test "$(git -C /opt/colmap-src rev-parse HEAD)" = "${COLMAP_COMMIT}"

RUN cmake -S /opt/colmap-src -B /opt/colmap-build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCUDA_ENABLED=OFF \
        -DGUI_ENABLED=OFF \
        -DOPENGL_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
    && cmake --build /opt/colmap-build --parallel 4 \
    && cmake --install /opt/colmap-build \
    && strip /usr/local/bin/colmap \
    && /usr/local/bin/colmap --help >/tmp/colmap-help.txt

LABEL org.opencontainers.image.title="eSfO COLMAP 3.9.1 CPU sidecar" \
      org.opencontainers.image.source="https://github.com/y-azuma/eSfO-paper-faithful" \
      org.opencontainers.image.version="3.9.1" \
      org.opencontainers.image.revision="e99036415ec0cf0f75c1d0b8d60fdd91af0d6c68"

ENTRYPOINT ["/usr/local/bin/colmap"]
