# ============================================================
# Whisper STT Server for NVIDIA Jetson Orin Nano
# JetPack 6.x (L4T r36.x), aarch64, CUDA 12.6
#
# Multi-stage build:
#   Stage 1: Compile whisper.cpp with CUDA (devel image)
#   Stage 2: Minimal runtime image with just the binary + libs
#
# Build:  docker build -t whisper-stt-jetson .
# Run:    see start-gpu.sh
# ============================================================

# --- Stage 1: Build ---
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone whisper.cpp
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git .

# Make CUDA driver stubs visible to linker (no GPU driver during Docker build).
# Need both .so and .so.1 — the stub has SONAME libcuda.so.1, which downstream
# executables look for when linking against libggml-cuda.so
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/lib/libcuda.so.1 && \
    ln -s /usr/local/lib/libcuda.so.1 /usr/local/lib/libcuda.so && \
    ldconfig

# Build with CUDA for Orin (compute capability 8.7)
# Use -j2 to avoid OOM on 8GB Jetson during compilation
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j2

# Download the base.en model (142 MiB)
RUN bash ./models/download-ggml-model.sh base.en

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled binaries from builder
COPY --from=builder /build/build/bin/whisper-server /app/whisper-server
COPY --from=builder /build/build/bin/whisper-cli /app/whisper-cli
COPY --from=builder /build/build/bin/whisper-quantize /app/whisper-quantize

# Copy shared libs built by whisper.cpp
COPY --from=builder /build/build/src/libwhisper.so* /usr/local/lib/
COPY --from=builder /build/build/ggml/src/libggml*.so* /usr/local/lib/
COPY --from=builder /build/build/ggml/src/ggml-cuda/libggml-cuda.so* /usr/local/lib/
RUN ldconfig

# Copy the model
COPY --from=builder /build/models/ggml-base.en.bin /app/models/ggml-base.en.bin

# Copy model download script for adding models later
COPY --from=builder /build/models/download-ggml-model.sh /app/models/

EXPOSE 8005

# Default: run whisper-server with OpenAI-compatible endpoint
CMD ["/app/whisper-server", \
     "--model", "/app/models/ggml-base.en.bin", \
     "--host", "0.0.0.0", \
     "--port", "8005", \
     "--inference-path", "/v1/audio/transcriptions", \
     "--convert", \
     "--print-progress"]
