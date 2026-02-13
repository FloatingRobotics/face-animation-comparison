# Face Animation from Audio — Comparison of 4 Networks

A side-by-side comparison of four audio-driven face animation networks, each generating a talking-head video from a **single portrait image** and an **audio clip**.

## Networks Compared

| Network | Approach | Head Motion | Hair Motion | Blinking | Peak VRAM | Inference Time (15s audio) |
|---------|----------|:-----------:|:-----------:|:--------:|:---------:|:--------------------------:|
| [SadTalker](https://github.com/OpenTalker/SadTalker) | 3DMM coefficients + face renderer | Yes | No | Yes | ~5-6 GB | ~30s |
| [Hallo](https://github.com/fudan-generative-ai/hallo) | Diffusion UNet (40 DDIM steps) | Yes | Yes | Yes | ~8 GB | ~12 min |
| [EchoMimic](https://github.com/BadToBest/EchoMimic) | Diffusion UNet (6 DDIM steps, accelerated) | Yes | Yes | Yes | ~10-12 GB | ~3 min |
| LivePortrait + JoyVASA | Warping + diffusion motion generator | Yes | Yes (warping) | Yes (synthetic) | ~7 GB | ~22s |

### How Each Network Works

**SadTalker** — Generates 3D Morphable Model (3DMM) expression and head pose coefficients from audio using a learned mapping network, then renders the face using a face-vid2vid renderer with GFPGAN enhancement. Lightweight but produces no hair motion since only the face mesh is animated.

**Hallo** — Full diffusion-based approach using a modified Stable Diffusion UNet. Takes wav2vec2 audio features and animates the entire image through 40 denoising steps per 16-frame chunk. Produces high quality results with natural hair motion but is the slowest.

**EchoMimic** — Similar diffusion approach to Hallo but with an accelerated mode using only 6 DDIM steps. Uses whisper audio features. Faster than Hallo at the cost of some quality.

**LivePortrait + JoyVASA** — A two-stage approach: JoyVASA generates per-frame motion parameters (expression, rotation, translation) from audio using a diffusion model with DDIM fast sampling (50 steps). LivePortrait then warps the source face using these motion parameters via an appearance feature extractor, warping module, and SPADE generator. Includes synthetic eye blinks and EMA temporal smoothing.

## Output Videos

All output videos are in `results/`:

| File | Network | Size | Resolution | FPS | Duration |
|------|---------|------|:----------:|:---:|:--------:|
| `sadtalker_output.mp4` | SadTalker | 3.8 MB | 2048x2048 | 25 | 15s |
| `hallo_output.mp4` | Hallo | 1.6 MB | 512x512 | 25 | 15s |
| `echomimic_output.mp4` | EchoMimic | 1.8 MB | 512x512 | 24 | 15s |
| `liveportrait_output.mp4` | LivePortrait+JoyVASA | 1.3 MB | 512x512 | 25 | 15s |

## Input

- **Face image**: `assets/face.png` — 1024x1024 RGBA portrait
- **Audio**: `news_audio_short.wav` — 15-second English news reading generated via [edge-tts](https://github.com/rany2/edge-tts) with the "en-US-GuyNeural" voice

## Project Structure

```
face/
├── README.md
├── .gitignore
├── assets/
│   └── face.png                    # Source portrait image
├── news_audio_short.wav            # 15s audio input
├── news_audio.wav                  # Full 34s audio (untrimmed)
├── run_liveportrait.py             # Standalone LivePortrait+JoyVASA inference
├── results/
│   ├── sadtalker_output.mp4
│   ├── hallo_output.mp4
│   ├── echomimic_output.mp4
│   └── liveportrait_output.mp4
├── SadTalker/                      # SadTalker source code
│   ├── inference.py                # Main inference script
│   ├── requirements.txt
│   └── ...
├── hallo/                          # Hallo source code
│   ├── scripts/inference.py        # Main inference script
│   ├── configs/inference/default.yaml
│   └── ...
└── echomimic/                      # EchoMimic source code
    ├── infer_audio2vid_acc.py       # Accelerated inference script
    ├── configs/prompts/animation_acc.yaml
    └── ...
```

> **Note:** Model weights are not included in this repository (they total ~46 GB). See the setup instructions below to download them.

## Setup & Run Instructions

### Prerequisites

- **GPU**: NVIDIA GPU with at least 12 GB VRAM (tested on RTX 4090 24GB)
- **OS**: Linux (tested on Ubuntu with kernel 6.8)
- **Python**: 3.10+
- **CUDA**: 12.1+ with PyTorch 2.3+
- **ffmpeg**: Required for video encoding

### 1. SadTalker

```bash
# Create virtual environment
cd SadTalker
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install face_alignment imageio imageio-ffmpeg librosa kornia yacs \
    pydub scipy tqdm safetensors
pip install basicsr facexlib gfpgan

# Download model weights (~1.7 GB)
mkdir -p checkpoints gfpgan/weights
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar -O checkpoints/mapping_00109-model.pth.tar
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar -O checkpoints/mapping_00229-model.pth.tar
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors -O checkpoints/SadTalker_V0.0.2_256.safetensors
wget -nc https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors -O checkpoints/SadTalker_V0.0.2_512.safetensors
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth -O gfpgan/weights/alignment_WFLW_4HG.pth
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -O gfpgan/weights/detection_Resnet50_Final.pth
wget -nc https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O gfpgan/weights/GFPGANv1.4.pth
wget -nc https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth -O gfpgan/weights/parsing_parsenet.pth

# Run inference
python inference.py \
    --driven_audio ../news_audio_short.wav \
    --source_image ../assets/face.png \
    --result_dir ../results/sadtalker \
    --still \
    --preprocess full \
    --enhancer gfpgan \
    --batch_size 1 \
    --size 512

deactivate
```

### 2. Hallo

```bash
# Create virtual environment
cd hallo
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install accelerate audio-separator diffusers==0.27.2 einops \
    insightface mediapipe omegaconf onnxruntime-gpu safetensors \
    transformers==4.39.2 xformers
pip install -e . --no-deps

# Download pretrained models (~9.4 GB)
huggingface-cli download fudan-generative-ai/hallo \
    --local-dir pretrained_models

# Run inference
python scripts/inference.py \
    --source_image ../assets/face.png \
    --driving_audio ../news_audio_short.wav \
    --output ../results/hallo_output.mp4

deactivate
```

### 3. EchoMimic

```bash
# Create virtual environment
cd echomimic
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install dependencies
pip install mediapipe transformers diffusers==0.24.0 torchmetrics \
    facenet_pytorch av moviepy accelerate omegaconf

# Download pretrained weights (~24 GB)
huggingface-cli download BadToBest/EchoMimic \
    --local-dir pretrained_weights

# Edit configs/prompts/animation_acc.yaml to set your image and audio paths:
#   test_cases:
#     "/path/to/face.png":
#       - "/path/to/audio.wav"

# Run accelerated inference
python infer_audio2vid_acc.py \
    --config configs/prompts/animation_acc.yaml

deactivate
```

### 4. LivePortrait + JoyVASA

This uses the pipeline from a separate agent project. To run it standalone:

```bash
# Requires the agent project at /home/salman/agent/ with models already downloaded
# (LivePortrait base models, JoyVASA motion generator, chinese-hubert-base)

cd /home/salman/agent
/home/salman/agent/venv/bin/python /home/salman/face/run_liveportrait.py
```

The script `run_liveportrait.py` can be modified by editing the constants at the top:
- `FACE_IMAGE` — path to the source portrait
- `AUDIO_PATH` — path to the driving audio
- `OUTPUT_PATH` — where to save the output video

## Compatibility Fixes Applied

Several patches were needed to run these networks with modern dependencies (numpy 1.x → 2.x, torchvision updates):

1. **SadTalker `np.VisibleDeprecationWarning`** — Added `hasattr` guard in `src/face3d/util/preprocess.py`
2. **SadTalker `torchvision.transforms.functional_tensor`** — Added try/except fallback import in `basicsr/data/degradations.py` (inside venv, not tracked)
3. **SadTalker `np.float` deprecated** — Changed to `float` in `src/face3d/util/my_awing_arch.py`
4. **SadTalker `np.array` inhomogeneous shape** — Added `float()` casts in `src/face3d/util/preprocess.py`
5. **EchoMimic diffusers version** — Required `diffusers==0.24.0` (not the system 0.27.2)

## Hardware Used

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM)
- **CPU**: AMD (Linux 6.8.0-90-generic)
- **Python**: 3.10.12
- **PyTorch**: 2.3.1+cu121
- **CUDA**: 12.1

## Audio Generation

The test audio was generated using edge-tts:

```bash
pip install edge-tts
edge-tts --voice "en-US-GuyNeural" \
    --text "Breaking news from around the world today..." \
    --write-media news_audio.mp3
ffmpeg -i news_audio.mp3 -ar 24000 -ac 1 news_audio.wav
# Trimmed to 15 seconds:
ffmpeg -i news_audio.wav -t 15 -c copy news_audio_short.wav
```
