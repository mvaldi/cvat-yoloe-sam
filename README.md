<p align="center">
  <img src="/site/content/en/images/cvat-readme-gif.gif" alt="CVAT Platform" width="100%" max-width="800px">
</p>

# CVAT + YOLOE + SAM3

Custom fork of [CVAT](https://github.com/cvat-ai/cvat) with **YOLOE Visual Prompt** and **SAM3** (Segment Anything Model 3) integration for AI-assisted annotation.

## ✨ Features

| Model | Description | Capabilities |
|-------|-------------|--------------|
| **YOLOE Visual Prompt** | Detection by visual examples | Rectangle, OBB (rotated), Polygon (segmentation) |
| **SAM3** | Text-prompted segmentation | Text-to-Segment, Text-to-Detect, Text-to-Track |

## 📋 Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA 12.4+ (minimum 8GB VRAM)
- [nuctl v1.13.0](https://github.com/nuclio/nuclio/releases/tag/1.13.0) (Nuclio CLI)

```bash
# Install nuctl
wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
chmod +x nuctl-1.13.0-linux-amd64
sudo mv nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
```

### For SAM3 (optional)

SAM3 requires access to the model on HuggingFace:

```bash
# Install HuggingFace CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Login and download model (requires approval at https://huggingface.co/facebook/sam3)
huggingface-cli login
huggingface-cli download facebook/sam3
```

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/mvaldi/cvat-yoloe-sam.git
cd cvat-yoloe-sam

# Start CVAT with all models
./zup.sh

# Or YOLOE only (without SAM3)
./zup.sh --no-sam3

# Or SAM3 only (without YOLOE)
./zup.sh --no-yoloe

# Base CVAT only (no AI models)
./zup.sh --no-sam3 --no-yoloe
```

Access CVAT at: **http://localhost:8080**

### Custom host (remote server)

```bash
./zup.sh --host $(hostname -I | awk '{print $1}')
```

## 🛑 Stop

```bash
# Stop containers
./zdown.sh

# Stop and clean Nuclio functions
./zdown.sh --clean
```

## 📖 Using the Models

### YOLOE Visual Prompt

1. Create a Task and upload images/video
2. Manually annotate some reference frames (minimum 1)
3. Go to **AI Tools** → **YOLOE**
4. Select reference frames and click **Generate VPE**
5. Navigate to an unannotated frame
6. Select **Output Type**: Rectangle | OBB | Polygon
7. Adjust **Confidence** and click **Detect**
8. Review and apply detections

### SAM3 (Segment Anything 3)

1. Go to **AI Tools** → **SAM3**
2. Enter a text prompt (e.g., "person", "car", "dog")
3. Select mode:
   - **Segment**: Segment specific object
   - **Detect**: Detect all instances
   - **Track**: Track object in video
4. Adjust confidence and apply results

## ⚠️ Considerations

### GPU Memory

| Configuration | Required VRAM |
|---------------|---------------|
| YOLOE only | ~4 GB |
| SAM3 only | ~6 GB |
| YOLOE + SAM3 | ~10 GB |

> **Note**: With GPUs <12GB VRAM, use only one model at a time.

### First startup

The first `./zup.sh` will download models and build Docker images. This may take **10-30 minutes** depending on your connection.

### Troubleshooting

```bash
# View server logs
docker logs cvat_server --tail 50

# View YOLOE logs
docker logs nuclio-nuclio-pth-ultralytics-yoloe-visual-prompt --tail 50

# View SAM3 logs
docker logs nuclio-nuclio-pth-facebookresearch-sam3-gpu --tail 50

# Check Nuclio functions
nuctl get function --platform local
```

## 📚 Additional Documentation

For complete CVAT documentation (formats, API, SDK, CLI):
- [Official CVAT Documentation](https://docs.cvat.ai/)
- [Original Repository](https://github.com/cvat-ai/cvat)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

This project includes models with additional licenses:
- **YOLOE**: [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **SAM3**: [Meta AI License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
