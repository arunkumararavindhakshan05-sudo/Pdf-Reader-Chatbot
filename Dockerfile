# ---------------------------------------------------------------------------
# PDF Reader Chatbot - production image for Azure Container Apps
#
# Two-stage build. The builder stage installs dependencies and pre-downloads
# the sentence-transformer model; the runtime stage copies only the finished
# virtualenv and model cache, so none of pip's build tooling ships to prod.
# ---------------------------------------------------------------------------

# =========================== Stage 1: builder ==============================
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Model cache location. Set in BOTH stages so the path the model is written to
# at build time is the same path it is read from at runtime.
ENV HF_HOME=/opt/hf-cache

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch, installed FIRST and from PyTorch's own index.
# The default PyPI torch wheel bundles NVIDIA CUDA libraries (~2.5 GB) that are
# dead weight on a CPU-only container. Installing it up front means the later
# `pip install -r requirements.txt` sees torch already satisfied and will not
# pull the CUDA build in as a sentence-transformers dependency.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4,<3.0"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image.
# Without this the container downloads ~90 MB from Hugging Face on first use,
# which turns every cold start into a slow, network-dependent, failure-prone
# operation. Baking it makes cold starts fast and the container fully offline.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('embedding model cached')"

# Drop pip/setuptools caches and compiled test fixtures that add size but no value
RUN find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    find /opt/venv -type d -name 'tests' -prune -exec rm -rf {} + && \
    rm -rf /root/.cache

# =========================== Stage 2: runtime ==============================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    # Never reach out to Hugging Face at runtime. If the bake above silently
    # failed, we want a loud error at startup instead of a surprise network
    # call from inside a locked-down container.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Keep torch from spawning one thread per host core. Azure Container Apps
    # gives us a fraction of a CPU; oversubscribing causes thrash, not speed.
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false \
    STREAMLIT_SERVER_PORT=8000

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

# Run as a non-root user. If the app is ever compromised through a malicious
# PDF, the attacker lands as an unprivileged user, not as root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser .streamlit ./.streamlit

RUN chown -R appuser:appuser /opt/hf-cache
USER appuser

EXPOSE 8000

# Streamlit exposes a built-in health endpoint. Azure Container Apps uses its
# own probes, but this keeps `docker run` locally honest too.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/_stcore/health',timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
