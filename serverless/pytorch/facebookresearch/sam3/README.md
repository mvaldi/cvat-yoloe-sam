# SAM3 - Segment Anything with Concepts

This folder contains the CVAT serverless function for **SAM3** (Segment Anything Model 3) from Meta AI, with full support for text prompts, detection, and video tracking.

## 🎯 Features

SAM3 brings four powerful capabilities to CVAT:

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Text-to-Segment** | Segment objects using natural language | "person in red shirt" → mask |
| **Refine with Clicks** | Interactive refinement with points | Click to include/exclude regions |
| **Text-to-Detect** | Detect ALL instances matching text | "car" → all cars in image |
| **Text-to-Track** | Video tracking with text/mask init | Track objects across frames |

## 📊 Model Specifications

- **Parameters**: 848M
- **Concepts**: 270,000+ unique concepts (open-vocabulary)
- **Benchmark**: 75-80% of human performance on SA-CO
- **Python**: 3.12+
- **PyTorch**: 2.7+
- **CUDA**: 12.6+
- **GPU Memory**: 8GB+ recommended

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SAM3 in CVAT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐  │
│   │ Text-to-    │   │ Text-to-    │   │ Text-to-Track       │  │
│   │ Segment     │   │ Detect      │   │ (Video Tracking)    │  │
│   └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘  │
│          │                 │                     │              │
│          └─────────────────┼─────────────────────┘              │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              model_handler.py (SAM3 Core)                │  │
│   │  - segment_with_text()   - detect_all()                  │  │
│   │  - segment_with_points() - init_tracking()               │  │
│   │  - segment_combined()    - track_frame()                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Redis Cache (TTL: 1 day)                    │  │
│   │  - Embeddings cache per frame                            │  │
│   │  - Tracking session state                                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Setup Instructions

### 1. Request Access to Model Weights

SAM3 requires HuggingFace authentication:

1. Go to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Request access to the model
3. Wait for approval (usually within a few hours)

### 2. Configure Hugging Face Authentication

**Option A: Using HF_TOKEN environment variable**
```bash
export HF_TOKEN="your_huggingface_token"
```

**Option B: Using huggingface-cli**
```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Deploy the Function

```bash
# GPU deployment (recommended)
nuctl deploy --project-name cvat \
  --path serverless/pytorch/facebookresearch/sam3/nuclio \
  --file serverless/pytorch/facebookresearch/sam3/nuclio/function-gpu.yaml \
  --platform local
```

## 📡 API Endpoints

SAM3 exposes multiple endpoints via the Nuclio function:

### Segmentation Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/segment-text` | POST | Text-to-Segment |
| `/segment-points` | POST | Refine with clicks |
| `/segment-combined` | POST | Text + points |

### Detection Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Text-to-Detect (all instances) |

### Tracking Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/track/init` | POST | Initialize tracking session |
| `/track/frame` | POST | Track to next frame |
| `/track/status` | GET | Get session status |
| `/track/clear` | POST | Clear session |

### Cache Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cache/status` | GET | Get cache status |
| `/cache/clear` | POST | Clear cache |

## 💻 Usage in CVAT

### UI: SAM3 Tab

1. Open AI Tools (magic wand icon)
2. Click on "SAM3" tab
3. Choose mode: **Segment**, **Detect**, or **Track**

### Text-to-Segment

```
1. Enter text prompt: "person in red shirt"
2. Click "Segment"
3. Mask appears on canvas
4. (Optional) Click to refine with points
```

### Text-to-Detect

```
1. Enter text prompt: "car"
2. Adjust confidence threshold (default 0.25)
3. Click "Detect All"
4. Review results, toggle selections
5. Click "Apply Annotations"
```

### Text-to-Track

```
1. Navigate to first frame where object appears
2. Enter text prompt: "dog"
3. Click "Start Tracking"
4. Navigate to next frame
5. Click "Track Frame"
6. Repeat for all frames
7. Click "Stop" when done
```

## 🔧 API Examples

### Text-to-Segment

```json
POST /segment-text
{
    "image": "<base64>",
    "text": "person in red shirt",
    "job_id": 123,
    "frame_idx": 0
}

Response:
{
    "masks": [...],
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.95, ...],
    "labels": ["person in red shirt", ...]
}
```

### Text-to-Detect

```json
POST /detect
{
    "image": "<base64>",
    "text": "car",
    "threshold": 0.25,
    "job_id": 123,
    "frame_idx": 0
}

Response:
{
    "detections": [
        {
            "mask": [...],
            "bbox": [x1, y1, x2, y2],
            "polygon": [x1, y1, x2, y2, ...],
            "score": 0.87,
            "label": "car"
        },
        ...
    ],
    "count": 5,
    "prompt": "car"
}
```

### Initialize Tracking

```json
POST /track/init
{
    "image": "<base64>",
    "job_id": 123,
    "init_type": "text",
    "text": "dog"
}

Response:
{
    "session_id": "track_123_5678",
    "frame_idx": 0,
    "mask": [...],
    "polygon": [x1, y1, x2, y2, ...],
    "object_ids": [1]
}
```

### Track Next Frame

```json
POST /track/frame
{
    "image": "<base64>",
    "session_id": "track_123_5678"
}

Response:
{
    "session_id": "track_123_5678",
    "frame_idx": 1,
    "mask": [...],
    "polygon": [x1, y1, x2, y2, ...]
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | cvat_redis_ondisk | Redis host |
| `REDIS_PORT` | 6666 | Redis port |
| `REDIS_TTL` | 86400 | Cache TTL (1 day) |
| `HF_TOKEN` | - | HuggingFace token |
| `HF_HOME` | /opt/nuclio/sam3/hf_cache | HF cache dir |

### Redis Cache

SAM3 uses Redis for:
- **Embeddings cache**: Avoid recomputing for same frame
- **Tracking state**: Persist tracking sessions between requests

Cache TTL is 1 day (86400 seconds).

## 📁 Files

```
sam3/
├── nuclio/
│   ├── function-gpu.yaml   # GPU deployment config
│   ├── main.py             # Multi-endpoint handler
│   └── model_handler.py    # SAM3 core with all methods
└── README.md               # This file
```

## 🆚 Comparison with SAM1

| Feature | SAM1 | SAM3 |
|---------|------|------|
| Text prompts | ❌ | ✅ 270K concepts |
| Open vocabulary | ❌ | ✅ |
| Video tracking | ❌ | ✅ |
| Point prompts | ✅ | ✅ |
| Box prompts | ✅ | ✅ |
| Python | 3.10+ | 3.12+ |
| Parameters | 636M | 848M |
| CUDA | Any | 12.6+ |

## 🐛 Troubleshooting

### "Unauthorized" or "Access denied"
- Verify HuggingFace access was approved
- Check HF_TOKEN is set correctly
- Try `huggingface-cli whoami` to verify auth

### "CUDA out of memory"
- SAM3 requires ~8GB VRAM
- Reduce batch size or image resolution
- Ensure no other processes using GPU

### "Redis connection failed"
- Verify cvat_redis_ondisk is running
- Check network connectivity
- Function will fall back to in-memory cache

### Slow first inference
- Model compilation on first run is normal
- Subsequent inferences will be faster
- Consider warming up with a test request

## 📚 References

- [SAM3 Paper](https://arxiv.org/abs/2511.16719)
- [SAM3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM3 HuggingFace](https://huggingface.co/facebook/sam3)
- [CVAT Documentation](https://docs.cvat.ai/)

### CUDA out of memory
- SAM 3 requires significant GPU memory (8GB+ recommended)
- Try reducing the number of workers in the function.yaml

### Slow inference
- First inference is slow due to model compilation
- Subsequent inferences should be faster

## References

- [SAM 3 Paper](https://arxiv.org/abs/2511.16719)
- [SAM 3 GitHub](https://github.com/facebookresearch/sam3)
- [SAM 3 Demo](https://segment-anything.com/)
- [Meta AI Blog](https://ai.meta.com/blog/segment-anything-model-3/)
