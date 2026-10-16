FROM libretranslate/libretranslate@sha256:726d14b752c6a9c5463f4e2b8f3aab5e78602b6eb4b341a86b8b18c8a2da0052

# LibreTranslate's CUDA image can fall back to a large CPU ONNX thread pool
# for every MiniSBD model. Bound sentence splitting without changing
# CTranslate2's CUDA translation path.
COPY sitecustomize.py /opt/sparehour-python/sitecustomize.py
ENV PYTHONPATH=/opt/sparehour-python \
    SPAREHOUR_MINISBD_THREADS=1
