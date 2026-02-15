#!/bin/bash
# Borg AI Services Startup
# Starts Chatterbox TTS and Whisper STT in the correct order
# Chatterbox MUST load first — it needs the largest GPU allocation

echo "[$(date)] Starting Borg AI services..."

# Clear memory
sync
echo 3 > /proc/sys/vm/drop_caches

# Stop any running instances
docker stop whisper-stt chatterbox-tts 2>/dev/null
sleep 2

# Record start time for log filtering
START_TIME=$(date -u +%Y-%m-%dT%H:%M:%S)

# Start Chatterbox TTS (needs GPU memory first)
echo "[$(date)] Starting Chatterbox TTS..."
docker start chatterbox-tts 2>/dev/null || \
  /home/muttleydosomething/chatterbox-jetson/start-gpu.sh

# Wait for model to load on GPU (takes ~30s)
echo "[$(date)] Waiting for Chatterbox model load..."
LOADED=0
for i in $(seq 1 90); do
  RECENT_LOGS=$(docker logs --since "$START_TIME" chatterbox-tts 2>&1)
  if echo "$RECENT_LOGS" | grep -q "TTS Model loaded successfully"; then
    echo "[$(date)] Chatterbox loaded successfully"
    LOADED=1
    break
  fi
  if echo "$RECENT_LOGS" | grep -q "CRITICAL.*failed to load"; then
    echo "[$(date)] ERROR: Chatterbox failed to load, retrying..."
    docker stop chatterbox-tts 2>/dev/null
    sleep 3
    sync
    echo 3 > /proc/sys/vm/drop_caches
    START_TIME=$(date -u +%Y-%m-%dT%H:%M:%S)
    docker start chatterbox-tts
  fi
  sleep 2
done

if [ "$LOADED" -eq 0 ]; then
  echo "[$(date)] WARNING: Chatterbox may not have loaded — starting Whisper anyway"
fi

# Start Whisper STT
# Remove old container and recreate with correct model
echo "[$(date)] Starting Whisper STT..."
docker rm whisper-stt 2>/dev/null

CUBLAS_HOST=/usr/local/cuda/targets/aarch64-linux/lib
CUBLAS_CTR=/usr/local/cuda/targets/sbsa-linux/lib

docker run -d --name whisper-stt \
  --runtime nvidia --network host \
  --shm-size=64m \
  -v whisper-models:/app/models \
  -v ${CUBLAS_HOST}/libcublas.so.12.6.1.4:${CUBLAS_CTR}/libcublas.so.12.6.4.1:ro \
  -v ${CUBLAS_HOST}/libcublasLt.so.12.6.1.4:${CUBLAS_CTR}/libcublasLt.so.12.6.4.1:ro \
  whisper-stt-jetson \
  /app/whisper-server \
    --model /app/models/ggml-base.en.bin \
    --host 0.0.0.0 --port 8005 \
    --inference-path /v1/audio/transcriptions \
    --convert --print-progress

# Verify both are running
sleep 5
if docker ps | grep -q chatterbox-tts && docker ps | grep -q whisper-stt; then
  echo "[$(date)] All services started successfully"
  echo "  Port 8004: Chatterbox TTS (/v1/audio/speech)"
  echo "  Port 8005: Whisper STT   (/v1/audio/transcriptions)"
else
  echo "[$(date)] WARNING: Not all services started"
  docker ps
fi
