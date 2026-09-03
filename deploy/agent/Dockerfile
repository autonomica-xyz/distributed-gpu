FROM nvidia/cuda@sha256:f8ef28f579ea42a44b415d2c5d46f788e6a9b395c6c83f2929416e1fc192c143

# Replace these with your own public controller URL and the digest of the
# distributed-gpu wheel you want agents to install.
ARG DGPU_WHEEL_URL=https://gpu-control.example.com/bootstrap/distributed_gpu-0.1.0-py3-none-any.whl
ARG DGPU_WHEEL_SHA256=replace-me
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && if [ "$DGPU_WHEEL_SHA256" = "replace-me" ]; then \
         echo "ERROR: set DGPU_WHEEL_SHA256 to the SHA-256 of your bootstrap wheel" >&2; \
         exit 1; \
       fi \
    && curl --fail --silent --show-error --location "$DGPU_WHEEL_URL" \
      -o /tmp/distributed_gpu-0.1.0-py3-none-any.whl \
    && echo "$DGPU_WHEEL_SHA256  /tmp/distributed_gpu-0.1.0-py3-none-any.whl" \
      | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir \
      /tmp/distributed_gpu-0.1.0-py3-none-any.whl \
    && rm -f /tmp/distributed_gpu-0.1.0-py3-none-any.whl \
    && rm -rf /var/lib/apt/lists/*

LABEL dev.distributed-gpu.dgpu.wheel-sha256="$DGPU_WHEEL_SHA256"
WORKDIR /work
ENTRYPOINT ["dgpu", "agent"]
