#!/bin/bash
# CVAT Shutdown Script
# Usage: ./zdown.sh [--clean]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Parse arguments
CLEAN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./zdown.sh [--clean]"
            echo ""
            echo "Options:"
            echo "  --clean    Remove all Nuclio functions and cached data"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "Stopping CVAT"
echo "========================================="

# Stop CVAT services
echo "[1/2] Stopping CVAT containers..."
docker compose \
    -f docker-compose.yml \
    -f docker-compose.dev.yml \
    -f components/serverless/docker-compose.serverless.yml \
    -f components/serverless/docker-compose.serverless.override.yml \
    down

# Optionally clean up Nuclio functions
if [ "$CLEAN" = true ]; then
    echo "[2/2] Cleaning up Nuclio functions..."

    # Delete Fragpunk function
    echo "      Removing YOLOv11 Fragpunk..."
    nuctl delete function pth-ultralytics-yolov11-fragpunk --platform local 2>/dev/null || true
    rm -f "$SCRIPT_DIR/serverless/pytorch/ultralytics/yolov11-fragpunk/nuclio/best.pt"

    # Delete YOLOE Visual Prompt function
    echo "      Removing YOLOE Visual Prompt..."
    nuctl delete function pth-ultralytics-yoloe-visual-prompt --platform local 2>/dev/null || true

    # Delete SAM3 function
    echo "      Removing SAM3..."
    nuctl delete function pth-facebookresearch-sam3-gpu --platform local 2>/dev/null || true

    # Restore SAM3 YAML placeholder for portability
    SAM3_YAML="$SCRIPT_DIR/serverless/pytorch/facebookresearch/sam3/nuclio/function-gpu.yaml"
    if grep -q "$HOME/.cache/huggingface" "$SAM3_YAML" 2>/dev/null; then
        echo "      Restoring SAM3 config placeholder..."
        sed -i "s|$HOME|__USER_HOME__|g" "$SAM3_YAML"
    fi

    echo "      Cleanup complete"
    echo ""
    echo "Note: VPE cache in Redis will expire automatically (30-day TTL)"
    echo "      or be cleared when Redis container is removed."
else
    echo "[2/2] Skipping cleanup (use --clean to remove Nuclio functions)"
fi

echo ""
echo "========================================="
echo "CVAT stopped"
echo "========================================="