# YOLOE Visual Prompt for CVAT

This serverless function enables **Visual Prompting** capabilities in CVAT using Ultralytics YOLOE model. Users can provide annotated reference images to detect similar objects in other frames without retraining.

## Overview

YOLOE (You Only Look Once - Everything) with Visual Prompting allows:
- **Zero-shot detection**: Detect objects based on visual examples rather than pre-trained classes
- **Flexible references**: Use 1-50 annotated frames as visual prompts
- **Multiple output types**: Bounding boxes, segmentation polygons, or OBB (Oriented Bounding Boxes)
- **Job-specific caching**: VPE (Visual Prompt Embeddings) cached per job with 30-day TTL

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CVAT UI Component                       │
│  (cvat-ui/src/components/.../yoloe-inline-panel.tsx)           │
│  Integrated as tab in AI Tools (tools-control.tsx)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Django Backend API                           │
│  (cvat/apps/lambda_manager/views_yoloe.py)                     │
│  Endpoints: /api/lambda/yoloe/*                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (Nuclio invocation)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Nuclio Serverless Function                     │
│  (serverless/pytorch/ultralytics/yoloe-visual-prompt/nuclio)   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ main.py     │  │model_handler │  │ redis_cache  │          │
│  │ HTTP routes │──│ YOLOEHandler │──│ VPECache     │          │
│  └─────────────┘  └──────────────┘  └──────────────┘          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Redis Ondisk  │
                    │  VPE Cache    │
                    │  TTL: 30 days │
                    │  Port: 6666   │
                    └───────────────┘
```

## How It Works

### 1. Generate VPE (Visual Prompt Embeddings)

When users select annotated frames and click "Generate VPE":

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATE VPE                             │
│                                                             │
│  Frame 0 ──┐                                                │
│  [head]    │                                                │
│            ├──► YOLOE ──► VPE [1, N, 512] ──► Redis Cache   │
│  Frame 4 ──┤      ▲                                         │
│  [head]    │      │                                         │
│  [body]    │   Learns what                                  │
│            │   "head" and "body"                            │
│  Frame 11 ─┘   look like                                    │
└─────────────────────────────────────────────────────────────┘
```

**Process:**
1. Collect annotations (bboxes + labels) from selected reference frames
2. For each reference image, call YOLOE with `visual_prompts` and `refer_image`
3. Extract embeddings from `model.model.pe` (shape `[1, num_classes, 512]`)
4. Average embeddings per class across all reference images
5. Cache final VPE tensor in Redis with 30-day TTL

**Note:** YOLOE only supports one reference image at a time, so we process each image separately and average the embeddings per class.

```python
# For each reference image
visual_prompts = {'bboxes': [[x1,y1,x2,y2], ...], 'cls': [0, 1, ...]}

model.predict(
    image,
    visual_prompts=visual_prompts,
    refer_image=[image],
    predictor=YOLOEVPSegPredictor,
)

# Extract and average embeddings
pe = model.model.pe.cpu()  # [1, num_classes, 512]
```

### 2. Detect Frame (Prediction with VPE)

When users click "Detect Frame #N":

```
┌─────────────────────────────────────────────────────────────┐
│                 DETECT FRAME #100                           │
│                                                             │
│  Redis Cache ──► VPE ──► YOLOE ──► Predictions              │
│                           │                                 │
│  Frame 100 ───────────────┘                                 │
│  (unannotated)                                              │
│                              ▼                              │
│                    ┌─────────────────┐                      │
│                    │ Detections:     │                      │
│                    │ - head (0.85)   │                      │
│                    │ - head (0.72)   │                      │
│                    │ - body (0.91)   │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

**Process:**
1. Load VPE from Redis cache
2. Configure model: `model.set_classes(class_names, vpe)`
3. Run prediction on target frame
4. Return detections (bboxes, labels, confidence)
5. User can preview and apply as annotations

### 3. OBB Conversion (Optional)

For Oriented Bounding Boxes output:
- Get segmentation mask from YOLOE
- Apply `cv2.minAreaRect()` to find minimum enclosing rotated rectangle
- Convert to 4 corner points

```python
def mask_to_obb(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)  # 4 points
    return box.flatten().tolist()
```

## Deployment

### Prerequisites

- CVAT running with serverless support
- GPU available (NVIDIA with CUDA)
- Redis ondisk server (port 6666)

### Deploy Function

```bash
cd /path/to/cvat

# Deploy YOLOE function only
./serverless/deploy_yoloe.sh

# Or with full stack
./zup.sh --no-fragpunk
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| REDIS_HOST | cvat_redis_ondisk | Redis server hostname |
| REDIS_PORT | 6666 | Redis server port |
| MODEL_PATH | /opt/nuclio/yoloe-11l-seg.pt | Path to YOLOE model |
| VPE_TTL_DAYS | 30 | VPE cache expiration |

## API Endpoints

All endpoints under `/api/lambda/yoloe/`:

### GET /annotated-frames
Get frames with annotations for reference selection.
```
GET /api/lambda/yoloe/annotated-frames?job_id=123

Response:
{
    "job_id": 123,
    "total_frames": 500,
    "annotated_count": 10,
    "max_references": 50,
    "frames": [
        {"frame": 0, "annotation_count": 2, "labels": ["head", "body"]},
        ...
    ]
}
```

### POST /generate-vpe
Generate VPE from selected reference frames.
```
POST /api/lambda/yoloe/generate-vpe
{
    "job_id": 123,
    "reference_frames": [0, 4, 11, 39, 486]
}

Response:
{
    "success": true,
    "job_id": 123,
    "num_references": 5,
    "total_annotations": 10,
    "class_names": ["body", "head"],
    "ttl_days": 30
}
```

### POST /predict
Run detection on target frames.
```
POST /api/lambda/yoloe/predict
{
    "job_id": 123,
    "frames": [100],
    "threshold": 0.25,
    "output_type": "rectangle"
}

Response:
{
    "results": [
        {
            "frame": 100,
            "detections": [
                {"label": "head", "points": [x1,y1,x2,y2], "type": "rectangle", "confidence": "0.85"},
                ...
            ]
        }
    ]
}
```

### GET /status
Get VPE cache status for a job.
```
GET /api/lambda/yoloe/status?job_id=123

Response:
{
    "exists": true,
    "job_id": 123,
    "reference_frames": [0, 4, 11, 39, 486],
    "class_names": ["body", "head"],
    "num_references": 5,
    "ttl_remaining_days": 29
}
```

### POST /clear
Clear VPE cache for a job.
```
POST /api/lambda/yoloe/clear
{"job_id": 123}
```

## UI Usage

1. **Open job** in CVAT annotation view
2. **Go to AI Tools** tab (magic wand icon)
3. **Select "YOLOE VP"** tab
4. **Select reference frames** with checkboxes (frames you've annotated)
5. **Click "Generate VPE"** to create embeddings
6. **Navigate** to an unannotated frame
7. **Click "Detect Frame #N"** to run detection
8. **Preview results** and click "Apply" to add as annotations

## Best Practices

- **Reference diversity**: Include varied examples (angles, lighting, sizes)
- **Reference count**: 5-20 references usually optimal
- **Threshold tuning**: Start low (0.15-0.25) for cross-image detection
- **Output types**:
  - Rectangle: Fast, good for simple objects
  - Polygon: Better for irregular shapes
  - OBB: Best for rotated objects

## Troubleshooting

### VPE not generating
- Check Redis connectivity (port 6666)
- Verify annotations exist on reference frames
- Check Nuclio function logs: `docker logs nuclio-nuclio-pth-ultralytics-yoloe-visual-prompt`

### Poor detection results
- Add more diverse reference images
- Lower confidence threshold
- Ensure reference annotations are accurate

### Function not healthy
- Check GPU availability
- Verify model downloaded: `/opt/nuclio/yoloe-11l-seg.pt`
- Check Nuclio dashboard: http://localhost:8070

## Files

```
serverless/pytorch/ultralytics/yoloe-visual-prompt/
├── nuclio/
│   ├── function-gpu.yaml    # Nuclio configuration
│   ├── main.py              # HTTP handler & routing
│   ├── model_handler.py     # YOLOE inference logic
│   └── redis_cache.py       # VPE caching with TTL
├── deploy_yoloe.sh          # Deployment script
└── README.md                # This file

cvat/apps/lambda_manager/
├── views_yoloe.py           # Backend API endpoints
└── urls.py                  # URL routing (includes yoloe_router)

cvat-ui/src/components/annotation-page/standard-workspace/controls-side-bar/
├── yoloe-inline-panel.tsx   # React UI component
└── tools-control.tsx        # Integrates YOLOE VP tab
```

## License

MIT License - See LICENSE file
