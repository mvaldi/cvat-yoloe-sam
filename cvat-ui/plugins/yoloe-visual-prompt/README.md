# YOLOE Visual Prompt Plugin for CVAT

This plugin enables YOLOE Visual Prompting functionality in CVAT, allowing users to detect objects based on annotated reference examples.

## Features

- **Visual Prompting**: Use annotated frames as references to detect similar objects
- **Multiple Output Types**: Support for bounding boxes, polygons (segmentation), and OBB (oriented bounding boxes)
- **Confidence Threshold**: Adjustable threshold via slider (0.05 - 0.95)
- **Batch Processing**: Run detection on single frame or range of frames
- **Reference Management**: Select up to 50 annotated frames as references
- **VPE Caching**: Visual Prompt Embeddings are cached in Redis with 30-day TTL

## Workflow

1. **Annotate Reference Frames**: Manually annotate several frames with bounding boxes
2. **Select References**: Open the YOLOE panel and select annotated frames as references (max 50)
3. **Generate VPE**: Click "Generate VPE" to create visual prompt embeddings
4. **Configure Detection**: Set confidence threshold and output type
5. **Run Detection**: Detect on current frame or a range of frames
6. **Apply Results**: Review and apply the detected annotations

## API Endpoints

The plugin communicates with these backend endpoints:

- `GET /api/lambda/yoloe/annotated-frames?job_id=N` - List frames with annotations
- `POST /api/lambda/yoloe/generate-vpe` - Generate VPE from references
- `POST /api/lambda/yoloe/predict` - Run detection on target frames
- `GET /api/lambda/yoloe/status?job_id=N` - Get VPE cache status
- `POST /api/lambda/yoloe/clear` - Clear VPE cache
- `POST /api/lambda/yoloe/apply` - Apply predictions as annotations

## Installation

This plugin is automatically included when building cvat-ui. No additional installation required.

## Requirements

- YOLOE Visual Prompt serverless function must be deployed
- Redis must be available for VPE caching

## Usage Tips

- More reference annotations = better detection accuracy
- Lower threshold for cross-image detection (0.15-0.25)
- Use segmentation output (polygon/OBB) for precise object boundaries
- VPE is cached per job, so different jobs need separate references
