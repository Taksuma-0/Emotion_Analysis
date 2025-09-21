# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from .pipelines import Pipelines
from .utils import to_prob_dict, segment_audio_vad, sample_video_frames, detect_main_face_mediapipe, absorb_empty_dicts

def asr_transcribe(pipes: Pipelines, wav_path: str, language="es"):
    segments, info = pipes.asr.transcribe(wav_path, language=language, vad_filter=True)
    rows = []
    for seg in segments:
        rows.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
    return pd.DataFrame(rows)

def text_emotions(pipes: Pipelines, df_asr: pd.DataFrame):
    outs = []
    for _, r in df_asr.iterrows():
        if not r["text"]:
            continue
        preds = pipes.text(r["text"])
        prob = to_prob_dict(preds)
        prob["start"] = r["start"]
        prob["end"] = r["end"]
        outs.append(prob)
    return pd.DataFrame(absorb_empty_dicts(outs))

def audio_emotions(pipes: Pipelines, wav_path: str, cfg):
    audio, sr, segs = segment_audio_vad(
        wav_path,
        frame_ms=cfg["vad_frame_ms"],
        aggressiveness=cfg["vad_aggressiveness"],
        min_segment_ms=1000,
        max_segment_ms=int(cfg["audio_segment_seconds"]*1000)
    )
    outs = []
    for (s, e) in segs:
        clip = audio[s:e].astype(np.float32)
        preds = pipes.audio({"array": clip, "sampling_rate": sr})
        prob  = to_prob_dict(preds)
        prob["start"] = s / sr
        prob["end"]   = e / sr
        outs.append(prob)
    return pd.DataFrame(absorb_empty_dicts(outs))

def video_emotions(pipes: Pipelines, video_path: str, cfg):
    frames = sample_video_frames(video_path, cfg["sample_every_n_frames"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    outs = []
    for idx, frame in tqdm(frames, desc="Video frames", total=len(frames)):
        face = detect_main_face_mediapipe(frame, cfg["min_face_detection_conf"])
        if face is None:
            continue
        x1, y1, x2, y2 = face
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        try:
            pil_img = Image.fromarray(crop_rgb)
            preds = pipes.vision(pil_img)
            prob = to_prob_dict(preds)
            t = idx / fps
            prob["time"] = t
            outs.append(prob)
        except Exception:
            continue
    return pd.DataFrame(absorb_empty_dicts(outs))
