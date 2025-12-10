#!/bin/bash
# Deploy YOLOE Visual Prompt model to CVAT serverless
# Usage: ./deploy_yoloe.sh
#
# This deploys YOLOE with Visual Prompting support:
# - Uses yoloe-11l-seg.pt for segmentation (enables OBB via post-processing)
# - Supports bbox, polygon, and OBB output types
# - Caches Visual Prompt Embeddings in Redis with 30-day TTL

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNC_DIR="$SCRIPT_DIR/pytorch/ultralytics/yoloe-visual-prompt/nuclio"

echo "========================================"
echo "YOLOE Visual Prompt CVAT Serverless Deploy"
echo "========================================"

# Check if function directory exists
if [ ! -d "$FUNC_DIR" ]; then
    echo "ERROR: Function directory not found at $FUNC_DIR"
    exit 1
fi

echo "Function dir: $FUNC_DIR"

# Create nuclio project if not exists
echo ""
echo "[1/3] Creating Nuclio project..."
nuctl create project cvat --platform local 2>/dev/null || echo "      Project already exists"

# Deploy the function (Nuclio builds the image)
echo ""
echo "[2/3] Deploying YOLOE Visual Prompt function..."
echo "      This will build the container and download the YOLOE model."
echo "      This may take several minutes on first deploy..."

nuctl deploy --project-name cvat --path "$FUNC_DIR" \
    --file "$FUNC_DIR/function-gpu.yaml" --platform local \
    --env REDIS_HOST=cvat_redis_ondisk \
    --env REDIS_PORT=6666 \
    --env VPE_TTL_DAYS=30 \
    --env MODEL_PATH=/opt/nuclio/yoloe-11l-seg.pt \
    --platform-config '{"attributes": {"network": "cvat_cvat"}}'

# Show deployed functions
echo ""
echo "[3/3] Verifying deployment..."
nuctl get function --platform local

echo ""
echo "========================================"
echo "Deployment complete!"
echo ""
echo "The function 'pth-ultralytics-yoloe-visual-prompt' is now available."
echo ""
echo "Endpoints:"
echo "  POST /generate-vpe   - Generate VPE from reference images"
echo "  POST /predict        - Detect using cached VPE"
echo "  POST /predict-batch  - Batch detection"
echo "  POST /status         - Check VPE cache status"
echo "  POST /clear          - Clear VPE cache"
echo ""
echo "Features:"
echo "  - Visual Prompting: Learn from annotated examples"
echo "  - Output types: rectangle, polygon, obb"
echo "  - Redis cache: 30-day TTL with auto-renewal"
echo "  - Max references: 50 images"
echo "========================================"
