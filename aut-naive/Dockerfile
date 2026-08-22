# aut-naive container. DESIGN.md 1.4. Implemented in STEP 4, then FROZEN.
# Own Dockerfile on purpose: no shared build context with the harness.
#
# Build context is this directory only, which is why the policy text lives in
# ./corpus/ as a committed snapshot rather than being read from ../policies/.
# That snapshot IS the freeze: a merchant who edits a policy without redeploying
# their agent has an agent answering from stale text, and the harness should be
# able to catch that rather than paper over it. Re-freeze (rebuild + new tag)
# when the agent is meant to see new policy text.
#
# Nothing here references the harness. If this file ever needs a path outside
# this directory, commitment C3 has been compromised.

# Digest-pinned, not just tagged. `python:3.12-slim` is a moving tag - it is rebuilt for
# every patch release and every base-OS security update - so `FROM python:3.12-slim`
# alone means the frozen agent's interpreter and system libraries can change underneath a
# fixed git SHA. Same class of leak as an unpinned dependency, one layer lower.
# Digest resolved from the 2026-08-22 build. To move it deliberately:
#   docker pull python:3.12-slim   (prints the new Digest:), then rebuild and re-freeze.
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

# HF_HUB_CACHE is set explicitly and to the same directory as the other two. Without it
# the two conventions disagree: `sentence-transformers` treats SENTENCE_TRANSFORMERS_HOME
# as the cache *root* (so weights land in /opt/models/models--BAAI--...), while
# huggingface_hub derives its cache from HF_HOME by appending /hub (so it looks in
# /opt/models/hub and finds nothing). That happens to work while sentence-transformers is
# the only downloader, and fails closed under HF_HUB_OFFLINE=1 the moment anything reaches
# huggingface_hub directly. One root, three variables, no ambiguity.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    HF_HUB_CACHE=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HUB_OFFLINE=1

# libgomp1 is required by the faiss-cpu wheel; without it `import faiss` fails at
# runtime with a missing libgomp.so.1 rather than at build time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, installed first and explicitly. The default index serves CUDA
# wheels on Linux, which would add several gigabytes this agent never uses.
#
# Pinned here rather than in requirements.txt because the resolved build is
# 2.13.0+cpu, and that local-version segment exists only on this index - PyPI
# would 404 on it. `==2.13.0` matches `2.13.0+cpu` (PEP 440: a specifier with no
# local segment matches any local version), so this stays a one-line pin.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the frozen agent needs no network at
# runtime and cannot silently pick up a different revision on first request.
# HF_HUB_OFFLINE is set above, so this layer is the only chance to fetch it.
#
# `revision=` is the point of this layer, not an ornament. Without it the model id is a
# moving reference and a rebuild can fetch different weights at the same git SHA - the one
# form of drift commitment C3 cannot see. The value must stay equal to
# retrieval.MODEL_REVISION, which is what the runtime asks for; test_aut_freeze.py fails
# the build discipline if the two ever disagree.
RUN HF_HUB_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5', \
revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')"

# Injected by scripts/freeze_aut.py. Reported at /health and on every /chat reply.
#
# DELIBERATELY THE LAST THING BEFORE THE SOURCE COPY. These values change on every freeze,
# and Docker invalidates every layer below a changed ENV - so putting this block near the
# top would rebuild apt, CPU torch, the pinned pip install and the 130MB model bake (~230s)
# each time an agent is re-frozen. Down here, a freeze rebuilds only the cheap tail. That
# matters for more than convenience: freezing is the step that cuts an irreversible tag, and
# a slow build makes it likelier someone shortcuts the verification around it.
ARG AUT_COMMIT_SHA=""
ARG AUT_REPO_HEAD=""
ARG AUT_GIT_TAG=""
ARG AUT_FROZEN_AT=""
ENV AUT_COMMIT_SHA=${AUT_COMMIT_SHA} \
    AUT_REPO_HEAD=${AUT_REPO_HEAD} \
    AUT_GIT_TAG=${AUT_GIT_TAG} \
    AUT_FROZEN_AT=${AUT_FROZEN_AT}

COPY corpus/ ./corpus/
COPY chunker.py retrieval.py prompts.py backends.py app.py ./

EXPOSE 8000

# /health builds the index, so a passing healthcheck means the agent can answer.
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
  CMD python -c "\
import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=10).status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
