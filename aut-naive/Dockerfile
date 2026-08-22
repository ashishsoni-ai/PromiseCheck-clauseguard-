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

FROM python:3.12-slim

# Injected by scripts/freeze_aut.py. Reported at /health and on every /chat reply.
ARG AUT_COMMIT_SHA=""
ARG AUT_REPO_HEAD=""
ARG AUT_GIT_TAG=""
ARG AUT_FROZEN_AT=""
ENV AUT_COMMIT_SHA=${AUT_COMMIT_SHA} \
    AUT_REPO_HEAD=${AUT_REPO_HEAD} \
    AUT_GIT_TAG=${AUT_GIT_TAG} \
    AUT_FROZEN_AT=${AUT_FROZEN_AT}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
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
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the frozen agent needs no network at
# runtime and cannot silently pick up a different revision on first request.
# HF_HUB_OFFLINE is set above, so this layer is the only chance to fetch it.
RUN HF_HUB_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY corpus/ ./corpus/
COPY chunker.py retrieval.py prompts.py backends.py app.py ./

EXPOSE 8000

# /health builds the index, so a passing healthcheck means the agent can answer.
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
  CMD python -c "\
import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=10).status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
