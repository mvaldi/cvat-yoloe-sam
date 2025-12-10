#!/bin/bash
# CVAT Startup Script with Serverless AI Models
# Usage: ./zup.sh [--host <IP>] [--no-yoloe] [--no-sam3]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Feature flags
DEPLOY_YOLOE=true
DEPLOY_SAM3=true

# Default host (can be overridden with --host)
CVAT_HOST="${CVAT_HOST:-localhost}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            CVAT_HOST="$2"
            shift 2
            ;;
        --no-yoloe)
            DEPLOY_YOLOE=false
            shift
            ;;
        --no-sam3)
            DEPLOY_SAM3=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./zup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host <IP>   Set the host IP/hostname to access CVAT (default: localhost)"
            echo "  --no-yoloe    Skip YOLOE Visual Prompt deployment"
            echo "  --no-sam3     Skip SAM3 deployment"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Starting CVAT with Serverless Functions"
echo "  Host: $CVAT_HOST"
echo "========================================="

# Start CVAT services
echo "[1/3] Starting CVAT containers..."
CVAT_HOST="$CVAT_HOST" docker compose \
    -f docker-compose.yml \
    -f docker-compose.dev.yml \
    -f components/serverless/docker-compose.serverless.yml \
    -f components/serverless/docker-compose.serverless.override.yml \
    up -d --force-recreate --build

# Wait for nuclio to be ready
echo "[2/3] Waiting for Nuclio to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:8070/api/functions > /dev/null 2>&1; then
        echo "      Nuclio is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "      Waiting for Nuclio... ($attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "      WARNING: Nuclio may not be fully ready, continuing anyway..."
fi

# Deploy YOLOE Visual Prompt
echo "[3/4] Deploying YOLOE Visual Prompt..."
if [ "$DEPLOY_YOLOE" = true ]; then
    "$SCRIPT_DIR/serverless/deploy_yoloe.sh"
else
    echo "      Skipping YOLOE deployment (--no-yoloe)"
fi

# Deploy SAM3
echo "[4/4] Deploying SAM3..."
if [ "$DEPLOY_SAM3" = true ]; then
    "$SCRIPT_DIR/serverless/deploy_gpu.sh" "$SCRIPT_DIR/serverless/pytorch/facebookresearch/sam3/"
else
    echo "      Skipping SAM3 deployment (--no-sam3)"
fi

echo ""
echo "========================================="
echo "CVAT is ready!"
echo "  - Web UI: http://$CVAT_HOST:8080"
echo "  - Nuclio: http://$CVAT_HOST:8070"
echo ""
echo "Deployed models:"
[ "$DEPLOY_YOLOE" = true ] && echo "  - YOLOE Visual Prompt (visual prompting)"
[ "$DEPLOY_SAM3" = true ] && echo "  - SAM3 (text-to-segment, detect, track)"
echo "========================================="