# aut-strong container. DESIGN.md 1.4. Scaffolded in STEP 1, frozen in STEP 5.
# Own Dockerfile on purpose: no shared build context with the harness, and no
# shared build context with aut-naive either. These are two deployables that
# happen to be compared, not one agent with a flag.
#
# Build context is this directory only, which is why the policy text lives in
# ./corpus/ as a committed snapshot rather than being read from ../policies/ or
# from ../aut-naive/corpus/. DESIGN.md 1.4 says "same corpus", and the way that
# is guaranteed here is byte equality of two independent snapshots, asserted by
# a test - not by one directory reaching into the other. A symlink or a ../ path
# would make the two agents share a failure and would compromise C3.
#
# Nothing here references the harness. If this file ever needs a path outside
# this directory, commitment C3 has been compromised.

# Digest-pinned, not just tagged, for the reason spelled out in aut-naive's
# Dockerfile: `python:3.12-slim` is rebuilt for every patch release, so the tag
# alone lets the interpreter move underneath a fixed git SHA.
#
# DELIBERATELY THE SAME DIGEST AS aut-naive. The comparison in DESIGN.md 1.4 is
# between a naive prompt/retrieval stack and a defensive one; if the two agents
# also sat on different interpreters or different system libraries, a difference
# in over-promise rate would have a second candidate explanation and the slide
# would not survive a question about it. Everything that is not the independent
# variable is held equal on purpose.
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

# One cache root, three variables - see aut-naive's Dockerfile for why HF_HUB_CACHE
# has to be set explicitly rather than derived. This image bakes two models instead
# of one, which makes the ambiguity worse rather than better: a CrossEncoder reaches
# huggingface_hub by a different path than SentenceTransformer does, so the two
# conventions disagreeing would fail closed at runtime on the reranker only.
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

# CPU-only torch, installed first and explicitly, at the same pin as aut-naive.
# See aut-naive's Dockerfile for why the pin lives here and not in requirements.txt
# (the resolved build is 2.13.0+cpu, a local version segment that exists only on
# this index).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

# requirements.txt is byte-identical to aut-naive's, and a test asserts that.
# Deliberate: those pins were harvested by `pip freeze` from a real build of that
# image rather than hand-written, so copying the file preserves that provenance
# while hand-editing it would quietly destroy it. The reranker needs no new
# package - CrossEncoder ships inside the already-pinned sentence-transformers -
# so there was nothing to add, and the stub's unpinned `litellm` line was dropped
# rather than kept: this agent talks to its provider through the same clients
# aut-naive uses, so a second HTTP abstraction would be one more difference
# between the two builds with nothing to show for it.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The reranker revision, resolved 2026-08-28 against the live hub.
#
# aut-naive pins its one model with a literal `revision=` because without it the
# model id is a moving reference and a rebuild can fetch different weights at the
# same git SHA - the one form of drift commitment C3 cannot see. aut-strong has
# the same obligation for both of its models. This value came from
#   HfApi().model_info("BAAI/bge-reranker-base").sha
# and must stay equal to retrieval.RERANKER_REVISION.
#
# It is an ARG with a default rather than a literal so the guard below can see it,
# but the default is the thing to edit if it ever moves deliberately. Do not supply
# it with --build-arg: `python scripts/freeze_aut.py aut-strong --tag aut-strong-v1
# --build` forwards only the four freeze args, so a revision passed on the command
# line would vanish the moment the freeze is reproduced from the recorded arguments
# and the rebuild would silently disagree with the record.
#
# The guard rejects anything that is not a 40-character hex object id, which is
# what catches the failure that matters: `--build-arg RERANKER_REVISION=main` is a
# perfectly valid hub reference, builds cleanly, and produces an image whose weights
# depend on the day it was built. A branch name is not a pin. Failing closed here
# is the difference between a freeze record that attests to specific weights and
# one that attests to a name.
ARG RERANKER_REVISION=2cfc18c9415c912f9d8155881c133215df768a70
RUN case "$RERANKER_REVISION" in \
      *[!0-9a-f]* | "" ) BAD=1 ;; \
      * ) [ ${#RERANKER_REVISION} -eq 40 ] && BAD=0 || BAD=1 ;; \
    esac; \
    if [ "$BAD" = "1" ]; then \
      echo "ERROR: RERANKER_REVISION='$RERANKER_REVISION' is not a 40-char hex commit id." >&2; \
      echo "aut-strong bakes BAAI/bge-reranker-base and will not fetch a moving" >&2; \
      echo "reference such as a branch or tag name. Resolve the revision sha, edit" >&2; \
      echo "the ARG default in this Dockerfile, and keep retrieval.RERANKER_REVISION" >&2; \
      echo "equal to it." >&2; \
      exit 1; \
    fi

# Bake both models so the frozen agent needs no network at runtime and cannot
# silently pick up a different revision on first request. HF_HUB_OFFLINE is set
# above, so this layer is the only chance to fetch them.
#
# The embedding model and its revision are deliberately identical to aut-naive's:
# DESIGN.md 1.4 changes k and adds reranking, it does not change the embedder, so
# holding it equal keeps retrieval depth the independent variable. Both revisions
# must stay equal to their retrieval.py constants; test_aut_freeze.py fails the
# build discipline if any of them disagree.
RUN HF_HUB_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5', \
revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'); \
CrossEncoder('BAAI/bge-reranker-base', \
revision='${RERANKER_REVISION}')"

# Injected by scripts/freeze_aut.py. Reported at /health and on every /chat reply.
#
# DELIBERATELY THE LAST THING BEFORE THE SOURCE COPY, for the reason aut-naive
# gives: Docker invalidates every layer below a changed ENV, and this image's
# model bake is larger than aut-naive's because it fetches two models. Putting
# this block any higher would re-run apt, CPU torch, the pinned pip install and
# both model downloads on every re-freeze - and freezing is the step that cuts an
# irreversible tag, so a slow build is a build someone shortcuts the checks around.
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
# start-period is longer than aut-naive's: this agent loads a cross-encoder as well
# as an embedder before it can serve, and a healthcheck that expires during a cold
# start reports a broken agent instead of a slow one.
HEALTHCHECK --interval=30s --timeout=15s --start-period=150s --retries=3 \
  CMD python -c "\
import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=10).status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
