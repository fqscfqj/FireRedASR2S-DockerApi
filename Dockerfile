FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

ARG FIRERED_REPO_URL=https://github.com/FireRedTeam/FireRedASR2S
ARG FIRERED_REPO_DIR=/opt/FireRedASR2S
RUN git clone --depth 1 "${FIRERED_REPO_URL}" "${FIRERED_REPO_DIR}"

ENV FIRERED_REPO_DIR=${FIRERED_REPO_DIR} \
    AUTO_CLONE_FIRERED=0 \
    MODEL_PATH=/models \
    VRAM_TTL=300 \
    MODEL_DOWNLOAD_MODE=lazy \
    LOG_LEVEL=info

COPY app /app/app
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]

