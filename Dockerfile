# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS wheel

WORKDIR /src
COPY pyproject.toml README.rst LICENSE MANIFEST.in ./
COPY batbot ./batbot
# hadolint ignore=DL3013
RUN python -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    ORT_DISABLE_TELEMETRY=1 \
    BATBOT_CACHE_DIR=/tmp/batbot-cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# WAV decoding, OpenMP, cuDNN's zlib dependency, and signal forwarding.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libsndfile1 tini zlib1g \
 && rm -rf /var/lib/apt/lists/*

COPY --from=wheel /wheels /wheels
# hadolint ignore=DL3013
RUN python -m pip install --no-cache-dir /wheels/*.whl \
 && rm -rf /wheels \
 && groupadd --gid 10001 batbot \
 && useradd --create-home --uid 10001 --gid 10001 batbot \
 && mkdir -p /work /home/batbot/.cache \
 && chown -R batbot:batbot /work /home/batbot

COPY scripts/smoke_install.py /usr/local/share/batbot/smoke_install.py
WORKDIR /work
ENTRYPOINT ["/usr/bin/tini", "--", "batbot"]
CMD ["--help"]

# Runtime extras supply CUDA/cuDNN libraries matching the ORT build.
# The host supplies the NVIDIA driver through the NVIDIA Container Toolkit.
FROM base AS gpu
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
# hadolint ignore=DL3013
RUN python -m pip install --no-cache-dir "batbot[gpu]" \
 && python -m pip check
USER 10001:10001

# CPU is the default image, including on Apple Silicon Docker hosts.
FROM base AS cpu
# hadolint ignore=DL3013
RUN python -m pip install --no-cache-dir "batbot[cpu]" \
 && python -m pip check
USER 10001:10001
