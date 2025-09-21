# -*- coding: utf-8 -*-
import subprocess, math, tempfile
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import librosa
import cv2
import ffmpeg
import torch
from PIL import Image
import matplotlib.pyplot as plt

# VAD (opcional)
try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except Exception:
    webrtcvad = None
    _HAS_WEBRTCVAD = False

def prefer_gpu_log():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA disponible: usando GPU '{name}'")
    else:
        print("⚠ CUDA no disponible: usando CPU (más lento)")

def run_cmd(cmd: List[str]):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{err.decode('utf-8', errors='ignore')}")
    return out

def extract_audio_ffmpeg(video_path: str, out_wav: str, sr: int = 16000):
    (
        ffmpeg
        .input(video_path)
        .output(out_wav, ac=1, ar=sr, format='wav')
        .overwrite_output()
        .run(quiet=True)
    )
    return out_wav

def to_prob_dict(preds):
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], list):
        preds = preds[0]
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict) and "label" in preds[0]:
        return {d["label"]: float(d["score"]) for d in preds}
    if isinstance(preds, dict) and "labels" in preds and "scores" in preds:
        return {lbl: float(s) for lbl, s in zip(preds["labels"], preds["scores"])}
    return {}

def segment_audio_vad(wav_path: str, frame_ms=30, aggressiveness=2, min_segment_ms=1000, max_segment_ms=8000):
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)

    if _HAS_WEBRTCVAD:
        vad = webrtcvad.Vad(aggressiveness)
        frame_len = int(sr * (frame_ms/1000.0))
        frames = []
        for i in range(0, len(audio), frame_len):
            chunk = audio[i:i+frame_len]
            if len(chunk) < frame_len:
                break
            pcm16 = (chunk * 32767.0).astype(np.int16).tobytes()
            voiced = vad.is_speech(pcm16, sr)
            frames.append((i, i+frame_len, voiced))
        segments = []
        seg_start = None
        for (s, e, v) in frames:
            if v and seg_start is None:
                seg_start = s
            elif not v and seg_start is not None:
                segments.append((seg_start, s))
                seg_start = None
        if seg_start is not None:
            segments.append((seg_start, frames[-1][1]))
    else:
        frame_len = int(sr * (frame_ms/1000.0))
        energies = []
        for i in range(0, len(audio), frame_len):
            chunk = audio[i:i+frame_len]
            if len(chunk) < frame_len:
                break
            energies.append((i, i+frame_len, float((chunk**2).mean())))
        med = np.median([e for (_,_,e) in energies]) if energies else 0.0
        thr = med * 1.5
        segments, seg_start = [], None
        for (s, e, en) in energies:
            v = en >= thr
            if v and seg_start is None:
                seg_start = s
            elif not v and seg_start is not None:
                segments.append((seg_start, s))
                seg_start = None
        if seg_start is not None and energies:
            segments.append((seg_start, energies[-1][1]))

    refined = []
    for (s, e) in segments:
        dur = (e - s) / sr
        if dur < (min_segment_ms/1000.0):
            continue
        max_s = max_segment_ms/1000.0
        t = s
        while (e - t)/sr > max_s:
            refined.append((t, t + int(max_s*sr)))
            t += int(max_s*sr)
        refined.append((t, e))
    return audio, sr, refined

def sample_video_frames(video_path: str, sample_every_n_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el video: {video_path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % sample_every_n_frames == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames

def detect_main_face_mediapipe(image_bgr, min_conf=0.6):
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_conf) as fd:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = fd.process(image_rgb)
        if not res.detections:
            return None
        det = max(res.detections, key=lambda d: d.score[0])
        bbox = det.location_data.relative_bounding_box
        h, w = image_bgr.shape[:2]
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))
        if x2<=x1 or y2<=y1:
            return None
        return (x1, y1, x2, y2)

def save_plot_as_png(fig, out_path):
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def extract_final_span(text: str) -> str:
    if not text:
        return ""
    a = text.find("<final>")
    b = text.find("</final>")
    if a != -1 and b != -1 and b > a:
        return text[a+8:b].strip()
    return ""

def sanitize_llm_output(text: str) -> str:
    if not text:
        return ""
    bad_markers = ["Alright", "I need to", "Let's", "First, I'll", "In conclusion", "<think>", "</think>"]
    out = "\n".join([ln for ln in text.splitlines() if not any(m in ln for m in bad_markers)])
    return out.strip()

def absorb_empty_dicts(items):
    return [d for d in items if isinstance(d, dict) and len(d)]
