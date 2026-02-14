# Whisper STT on NVIDIA Jetson

**GPU-accelerated speech-to-text for Jetson Orin Nano using whisper.cpp.**

Real-time speech transcription running entirely on a $249 edge AI device, with an OpenAI-compatible API. Pairs perfectly with [Chatterbox TTS](https://github.com/muttleydosomething/chatterbox-jetson-nano) for a complete voice I/O stack.

| | |
|---|---|
| **Hardware** | NVIDIA Jetson Orin Nano (8GB) |
| **JetPack** | 6.x (L4T r36.x) |
| **CUDA** | 12.6 |
| **Engine** | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (C++, no Python/PyTorch) |
| **Default model** | `base.en` (147 MiB on GPU) |
| **Image size** | 3.67 GB |
| **Inference** | ~2-4 seconds for 10s of audio (GPU) |
| **License** | Apache 2.0 |

---

## Why whisper.cpp (Not faster-whisper)

The obvious choice for Whisper on GPU would be [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with CTranslate2. Unfortunately, **CTranslate2's pip wheels for aarch64 are CPU-only** — there are no CUDA-enabled ARM64 builds. Building CTranslate2 from source on Jetson has documented failures.

whisper.cpp compiles natively with CUDA via cmake, has a **built-in OpenAI-compatible HTTP server**, and requires no Python, no PyTorch, and no dependency hell. On Jetson, simplicity wins.

---

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano (8GB recommended) running JetPack 6.x
- Docker with `nvidia-container-runtime`

### 1. Get the image

**Option A: Pull pre-built (recommended)**

```bash
docker pull ghcr.io/muttleydosomething/whisper-stt-jetson:latest
docker tag ghcr.io/muttleydosomething/whisper-stt-jetson:latest whisper-stt-jetson
```

**Option B: Build from source (~30 min)**

```bash
git clone https://github.com/muttleydosomething/whisper-stt-jetson.git
cd whisper-stt-jetson
docker build -t whisper-stt-jetson .
```

Build takes ~30 minutes with `-j2` (most of that is compiling CUDA attention kernel template instances).

### 2. Start with GPU acceleration

```bash
chmod +x start-gpu.sh
./start-gpu.sh
```

> **Note:** Check your host cuBLAS library versions first:
> ```bash
> ls -la /usr/local/cuda/targets/aarch64-linux/lib/libcublas*
> ```
> If your version numbers differ from `12.6.1.4`, update `start-gpu.sh` accordingly.

### 3. Use it

```bash
# Transcribe an audio file
curl -X POST http://localhost:8005/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=base.en"
```

Response:
```json
{"text": " And so, my fellow Americans, ask not what your country can do for you.\n Ask what you can do for your country.\n"}
```

---

## The SBSA Problem

Like all `nvidia/cuda` Docker images on Jetson, this image ships **SBSA (Server Base System Architecture) builds of cuBLAS** — compiled for server-class ARM64 GPUs, not Jetson's Orin iGPU. These silently fail at runtime.

The `start-gpu.sh` script fixes this by bind-mounting your host's Jetson-native cuBLAS over the container's SBSA version. whisper.cpp only uses cuBLAS (no cuDNN or cuFFT needed), so only two mounts are required:

```bash
-v /usr/local/cuda/targets/aarch64-linux/lib/libcublas.so.12.6.1.4:/usr/local/cuda/targets/sbsa-linux/lib/libcublas.so.12.6.4.1:ro
-v /usr/local/cuda/targets/aarch64-linux/lib/libcublasLt.so.12.6.1.4:/usr/local/cuda/targets/sbsa-linux/lib/libcublasLt.so.12.6.4.1:ro
```

For a deep dive into the SBSA issue (which also affects cuDNN and cuFFT), see the [Chatterbox TTS project](https://github.com/muttleydosomething/chatterbox-jetson-nano) where it was first discovered and documented.

---

## Build Lessons: CUDA Stubs in Docker

Building CUDA code inside Docker (where no GPU driver is available) requires linking against CUDA driver stubs. The key insight: **you must create both `libcuda.so` AND `libcuda.so.1` symlinks from the stubs directory.**

```dockerfile
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/lib/libcuda.so.1 && \
    ln -s /usr/local/lib/libcuda.so.1 /usr/local/lib/libcuda.so && \
    ldconfig
```

Just `libcuda.so` alone is **not enough**. When `libggml-cuda.so` links against the stub, it records the SONAME `libcuda.so.1` in its `DT_NEEDED` entries. Downstream executables then fail at link time with `undefined reference to 'cuGetErrorString'` because the linker looks for `libcuda.so.1` (not `libcuda.so`).

Additionally, `libggml-cuda.so` is built in `ggml/src/ggml-cuda/` — a subdirectory that generic globs like `ggml/src/libggml*.so*` won't reach. It needs a separate `COPY` line in the Dockerfile.

---

## Models

The image ships with `base.en` (142 MiB download, 147 MiB on GPU). You can download additional models at runtime:

```bash
# Download inside the running container
docker exec whisper-stt bash /app/models/download-ggml-model.sh small.en

# Restart with the new model
docker stop whisper-stt && docker rm whisper-stt
# Edit start-gpu.sh or override CMD:
docker run -d --name whisper-stt \
  --runtime nvidia --network host \
  ... \
  whisper-stt-jetson \
  /app/whisper-server --model /app/models/ggml-small.en.bin \
  --host 0.0.0.0 --port 8005 \
  --inference-path /v1/audio/transcriptions \
  --convert --print-progress
```

Models are stored in a Docker volume (`whisper-models`) and persist across container recreations.

| Model | Download | GPU Memory | Accuracy | Speed (10s audio) |
|-------|----------|------------|----------|-------------------|
| `tiny.en` | 75 MiB | ~80 MiB | Basic | ~1-2s |
| `base.en` | 142 MiB | ~320 MiB | Good | ~2-4s |
| `small.en` | 466 MiB | ~730 MiB | Very good | ~4-8s |
| `medium.en` | 1.5 GiB | ~2.1 GiB | Excellent | Tight on 8GB |

> **Memory note:** On the 8GB Orin, `base.en` and `small.en` fit comfortably alongside [Chatterbox TTS](https://github.com/muttleydosomething/chatterbox-jetson-nano). `medium.en` may not leave enough room for both.

---

## Running Alongside Chatterbox TTS

This project is designed to run alongside Chatterbox TTS on the same Jetson, creating a complete voice I/O stack:

| Port | Service | Endpoint | Function |
|------|---------|----------|----------|
| 8004 | Chatterbox TTS | `POST /v1/audio/speech` | Text to speech |
| 8005 | Whisper STT | `POST /v1/audio/transcriptions` | Speech to text |

### Startup Order Matters

**Chatterbox must load first.** It uses PyTorch which needs a large contiguous GPU allocation (~3-5 GiB). If Whisper grabs GPU memory first, Chatterbox's `cudaMalloc` will fail with `NVML_SUCCESS == r INTERNAL ASSERT FAILED`.

Recommended approach: let Chatterbox auto-start via Docker's `unless-stopped` restart policy, then start Whisper with a delay via systemd:

```ini
# /etc/systemd/system/whisper-stt.service
[Unit]
Description=Whisper STT Server (after Chatterbox TTS)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/bin/sleep 45
ExecStart=/usr/bin/docker start whisper-stt
ExecStop=/usr/bin/docker stop whisper-stt

[Install]
WantedBy=multi-user.target
```

### Memory Budget (8GB Orin)

```
Chatterbox TTS (fp16):  ~5.5 GiB (PyTorch + model + CUDA context)
Whisper base.en:        ~320 MiB (model + compute buffers)
Kernel + driver:        ~650 MiB
Total:                  ~6.5 GiB of 7.5 GiB available
```

---

## API Reference

### `POST /v1/audio/transcriptions`

OpenAI-compatible transcription endpoint.

```bash
curl -X POST http://localhost:8005/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=base.en" \
  -F "response_format=json"
```

**Supported audio formats:** WAV, MP3, M4A, FLAC, OGG, WebM (ffmpeg handles conversion).

**Response:**
```json
{"text": "The transcribed text appears here."}
```

### `POST /inference`

whisper.cpp native endpoint (also available depending on build).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `cudaMalloc failed: out of memory` | Another process is using GPU memory. Stop other containers, start Chatterbox first, then Whisper |
| `CUBLAS_STATUS_ALLOC_FAILED` | SBSA cuBLAS loaded. Use `start-gpu.sh` instead of plain `docker run` |
| `undefined reference to cuGetErrorString` (build) | Missing `libcuda.so.1` stub symlink. See CUDA Stubs section above |
| `libggml-cuda.so.0: cannot open shared object` (runtime) | Missing COPY line for `ggml/src/ggml-cuda/` in Dockerfile |
| Build OOM at ~50% | Reduce parallelism: change `-j2` to `-j1` in Dockerfile |
| Slow inference (~20-30s) | Running on CPU. Check `docker logs` for `ggml_cuda_init: found 1 CUDA devices` |
| Empty transcription | Audio might be too quiet, too short, or non-speech. Check with `whisper-cli` directly |

---

## Compatibility

Tested on:
- **Hardware:** NVIDIA Jetson Orin Nano 8GB Super
- **JetPack:** 6.2.2 (L4T 36.5.0)
- **CUDA:** 12.6, Driver 540.5.0
- **Kernel:** 5.15.185-tegra

Should work on other Jetson Orin variants (Orin NX, AGX Orin) with JetPack 6.x. The cuBLAS version numbers in `start-gpu.sh` may need adjusting for different JetPack releases.

**Not compatible with:** Jetson Nano (original), Jetson TX2, Jetson Xavier — these use older JetPack versions with different CUDA toolkits and compute capabilities.

---

## See Also

- **[Chatterbox TTS on Jetson](https://github.com/muttleydosomething/chatterbox-jetson-nano)** — GPU-accelerated text-to-speech with voice cloning. Runs alongside Whisper on the same Orin Nano for a complete voice I/O stack (TTS on port 8004, STT on port 8005).

---

## Acknowledgments

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) by Georgi Gerganov — the engine that makes this possible
- [OpenAI Whisper](https://github.com/openai/whisper) — the original model
- Built with [Claude Code](https://claude.ai/code) (Claude Opus 4.6) — the CUDA stubs discovery and SBSA workarounds were developed collaboratively between human and AI

---

## Support This Project

If this saved you time getting Whisper running on Jetson, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow?style=for-the-badge&logo=buy-me-a-coffee)](https://buymeacoffee.com/muttleydosomething)

---

## License

Copyright 2026 Simon (muttleydosomething)

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
