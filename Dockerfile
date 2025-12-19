FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV GRADIO_SERVER_NAME=0.0.0.0

ENV GRADIO_SERVER_PORT=7860

# Install apt packages
# hadolint ignore=DL3008
RUN set -ex \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ \
        git \
        build-essential \
        ffmpeg \
        libsm6 \
        libtirpc-dev \
        libxext6 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
 && rm -rf /var/cache/apt \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./ /code

RUN python3 -m venv /venv

# hadolint ignore=DL3003,DL3013
RUN /venv/bin/pip install --no-cache-dir -r requirements/runtime.txt \
 && /venv/bin/pip install --no-cache-dir -r requirements/optional.txt \
 && cd tpl/pyastar2d/ \
 && /venv/bin/pip install --no-cache-dir -e . \
 && cd ../.. \
 && /venv/bin/pip install --no-cache-dir -e . \
 && if [ "$(uname -m)" != "aarch64" ] \
       ; then \
       /venv/bin/pip uninstall -y onnxruntime \
       /venv/bin/pip install --no-cache-dir onnxruntime-gpu \
       ; fi

CMD [".", "/venv/bin/activate", "&&", "exec", "python", "app.py"]
