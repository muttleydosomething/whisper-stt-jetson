#!/bin/bash
# Start Whisper STT server with GPU acceleration on Jetson Orin Nano
#
# Fixes applied:
# 1. Mount host Jetson-native cuBLAS (container ships SBSA builds)
#    whisper.cpp only needs cuBLAS — no cuDNN or cuFFT required
#
# Note: Check your host library versions with:
#   ls -la /usr/local/cuda/targets/aarch64-linux/lib/libcublas*
# and update the version numbers below if they differ.
#
# IMPORTANT: If running alongside Chatterbox TTS, use borg-ai-services.sh
# instead — it handles the correct startup order (Chatterbox must load first).

CUBLAS_HOST=/usr/local/cuda/targets/aarch64-linux/lib
CUBLAS_CTR=/usr/local/cuda/targets/sbsa-linux/lib

docker run -d --name whisper-stt \
  --runtime nvidia --network host \
  --shm-size=64m \
  -v whisper-models:/app/models \
  -v ${CUBLAS_HOST}/libcublas.so.12.6.1.4:${CUBLAS_CTR}/libcublas.so.12.6.4.1:ro \
  -v ${CUBLAS_HOST}/libcublasLt.so.12.6.1.4:${CUBLAS_CTR}/libcublasLt.so.12.6.4.1:ro \
  whisper-stt-jetson
