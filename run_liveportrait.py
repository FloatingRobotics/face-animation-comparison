#!/usr/bin/env python3
"""Standalone LivePortrait + JoyVASA inference script.

Reuses the pipeline from /home/salman/agent/ to generate a face animation
video from a static image + audio, for comparison with SadTalker/Hallo/EchoMimic.
"""

import sys
import os
import time
import logging

import numpy as np
import cv2
import torch
import librosa
import wave

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("liveportrait_standalone")

# Paths
AGENT_ROOT = "/home/salman/agent"
FACE_IMAGE = "/home/salman/agent/assets/user_face/face.png"
AUDIO_PATH = "/home/salman/face/news_audio_short.wav"
OUTPUT_PATH = "/home/salman/face/results/liveportrait_output.mp4"

# Add agent to sys.path so we can import its modules
sys.path.insert(0, AGENT_ROOT)
os.chdir(AGENT_ROOT)

# Set env so config loads correctly
os.environ.setdefault("APP_CONFIG", os.path.join(AGENT_ROOT, "configs", "app.yaml"))


def read_wav_int16(path: str) -> tuple:
    """Read WAV file and return (int16 array, sample_rate)."""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            audio = audio[::n_channels]  # take first channel
        return audio, sr


def frames_to_video(jpeg_frames: list, fps: int, audio_path: str, output_path: str):
    """Combine JPEG frames + audio into an MP4 video."""
    if not jpeg_frames:
        log.error("No frames to write!")
        return

    # Decode first frame to get dimensions
    first = cv2.imdecode(np.frombuffer(jpeg_frames[0], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]

    # Write frames to a temp video without audio
    tmp_video = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    for fb in jpeg_frames:
        frame = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
        writer.write(frame)
    writer.release()

    # Mux with audio using ffmpeg
    os.system(
        f'ffmpeg -y -i {tmp_video} -i {audio_path} '
        f'-c:v libx264 -preset fast -crf 18 -c:a aac -b:a 128k '
        f'-shortest -movflags +faststart {output_path} 2>/dev/null'
    )
    os.remove(tmp_video)
    log.info("Saved video to %s", output_path)


def main():
    t_start = time.time()

    # Import the pipeline
    from server.avatar.liveportrait_pipeline import LivePortraitPipeline

    log.info("Initializing LivePortrait + JoyVASA pipeline...")
    pipeline = LivePortraitPipeline()

    # Override settings for video output quality
    pipeline.frame_w = 512
    pipeline.frame_h = 512
    pipeline.jpeg_quality = 95
    pipeline._render_fps = 25  # full 25 FPS for video output
    pipeline._ddim_steps = 50  # fast DDIM sampling

    log.info("Loading face image: %s", FACE_IMAGE)
    ok = pipeline.setup_from_photo(FACE_IMAGE)
    if not ok:
        log.error("Failed to setup from photo!")
        return

    log.info("Reading audio: %s", AUDIO_PATH)
    audio_int16, sr = read_wav_int16(AUDIO_PATH)
    log.info("Audio: %d samples, %d Hz, %.1fs", len(audio_int16), sr, len(audio_int16) / sr)

    log.info("Generating frames...")
    t0 = time.time()
    frames, actual_fps = pipeline.generate_frames_for_audio(audio_int16, sample_rate=sr)
    gen_time = time.time() - t0
    log.info("Generated %d frames at %d FPS in %.1fs", len(frames), actual_fps, gen_time)

    if not frames:
        log.error("No frames generated!")
        return

    # Report VRAM usage
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.max_memory_allocated() / 1024**3
        vram_reserved = torch.cuda.max_memory_reserved() / 1024**3
        log.info("Peak VRAM allocated: %.2f GB", vram_alloc)
        log.info("Peak VRAM reserved: %.2f GB", vram_reserved)

    log.info("Writing video to %s", OUTPUT_PATH)
    frames_to_video(frames, actual_fps, AUDIO_PATH, OUTPUT_PATH)

    total_time = time.time() - t_start
    log.info("Total time: %.1fs", total_time)


if __name__ == "__main__":
    main()
