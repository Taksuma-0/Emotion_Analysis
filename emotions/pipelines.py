# -*- coding: utf-8 -*-
from dataclasses import dataclass
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoImageProcessor
)
from faster_whisper import WhisperModel
from .utils import prefer_gpu_log

@dataclass
class Pipelines:
    text: any
    audio: any
    vision: any
    asr: any

def load_pipelines(cfg) -> Pipelines:
    prefer_gpu_log()
    use_cuda = torch.cuda.is_available()
    device_index = 0 if use_cuda else -1
    device_str = "cuda:0" if use_cuda else "cpu"

    tok_txt = AutoTokenizer.from_pretrained(cfg["text_model"])
    mdl_txt = AutoModelForSequenceClassification.from_pretrained(cfg["text_model"])
    if use_cuda:
        mdl_txt = mdl_txt.to(device_str)
    text_pipe = pipeline(
        task="text-classification",
        model=mdl_txt,
        tokenizer=tok_txt,
        top_k=None,
        device=device_index
    )

    audio_model = AutoModelForAudioClassification.from_pretrained(cfg["audio_emotion_model"])
    audio_feat = AutoFeatureExtractor.from_pretrained(cfg["audio_emotion_model"])
    if use_cuda:
        audio_model = audio_model.to(device_str)
    audio_pipe = pipeline(
        task="audio-classification",
        model=audio_model,
        feature_extractor=audio_feat,
        top_k=None,
        device=device_index
    )

    vis_proc = AutoImageProcessor.from_pretrained(cfg["video_emotion_model"])
    vis_model = AutoModelForImageClassification.from_pretrained(cfg["video_emotion_model"])
    if use_cuda:
        vis_model = vis_model.to(device_str)
    vision_pipe = pipeline(
        task="image-classification",
        model=vis_model,
        feature_extractor=vis_proc,
        top_k=None,
        device=device_index
    )

    asr_model = WhisperModel(
        cfg["whisper_model_size"],
        device="cuda" if use_cuda else "cpu",
        compute_type="float16" if use_cuda else "int8"
    )
    return Pipelines(text=text_pipe, audio=audio_pipe, vision=vision_pipe, asr=asr_model)
