# syntax=docker/dockerfile:1
#
# Image that serves one model, baked in at build time and loaded offline.
#
#   docker build -t review-classifier \
#     --build-arg MODEL_ID=<hf-user>/<repo> --build-arg MODEL_REVISION=<commit> \
#     --secret id=hf_token,env=HF_TOKEN .          # the secret only for private models
#
# Stages: builder (Python environment) -> model (fetch + verify the model)
# -> runtime (slim, non-root, no network needed, no build tools, no token).

ARG PYTHON_IMAGE=python:3.12-slim

# ---- builder: the Python environment -----------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU-only PyTorch: the default wheel bundles CUDA and is several GB larger.
# Its own layer, so code changes do not reinstall it.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url "${TORCH_INDEX_URL}"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[hf,db]"

# ---- model: fetch the model and prove it loads offline -----------------------------
FROM builder AS model

# MODEL_ID is a Hugging Face Hub id, or a directory under models/ in the build
# context (it is copied to /opt/models). Leave MODEL_ID empty for the default model.
ARG MODEL_BACKEND=hf
ARG MODEL_ID=""
ARG MODEL_REVISION=""
ENV HF_HOME=/opt/hf
COPY models/ /opt/models/
# The Hugging Face token is mounted only for this step and is not stored in any layer.
# The service runs as a non-root user, so the model files are made world-readable,
# whatever permissions they had on the machine that built the image.
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p /opt/hf \
    && if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi \
    && MODEL_BACKEND="${MODEL_BACKEND}" MODEL_ID="${MODEL_ID}" MODEL_REVISION="${MODEL_REVISION}" \
       python -m review_classifier.prefetch \
    && chmod -R a+rX /opt/hf /opt/models

# ---- runtime ------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG MODEL_BACKEND=hf
ARG MODEL_ID=""
ARG MODEL_REVISION=""

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=model /opt/hf /opt/hf
COPY --from=model /opt/models /opt/models
# The release's deployment files: the production Compose file and the monitoring
# configuration. The AWS instance copies them out of the image it deploys, so they
# always match the code they run with (see infra/templates/start.sh).
COPY deploy/compose.yaml /opt/release/compose.yaml
COPY monitoring/prometheus /opt/release/monitoring/prometheus
COPY monitoring/grafana/provisioning /opt/release/monitoring/grafana/provisioning
COPY monitoring/grafana/dashboards /opt/release/monitoring/grafana/dashboards

# The image serves exactly the model it was built with, with the network off.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_BACKEND=${MODEL_BACKEND} \
    MODEL_ID=${MODEL_ID} \
    MODEL_REVISION=${MODEL_REVISION}

USER app
WORKDIR /home/app
EXPOSE 8000

# Ready = model loaded. The database is not part of readiness (see /ready).
HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/ready\", timeout=2)"]

CMD ["python", "-m", "review_classifier"]
