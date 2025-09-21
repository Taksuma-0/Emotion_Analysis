# -*- coding: utf-8 -*-
DEFAULT_CFG = {
    "text_model": "pysentimiento/robertuito-emotion-analysis",
    "audio_emotion_model": "superb/hubert-large-superb-er",
    "video_emotion_model": "trpakov/vit-face-expression",
    "whisper_model_size": "large-v3",        # tiny/base/small/medium/large-v3
    "sample_every_n_frames": 6,
    "min_face_detection_conf": 0.5,
    "vad_frame_ms": 10,
    "vad_aggressiveness": 2,
    "audio_segment_seconds": 2.0,
    "fusion_weights": {"text": 0.33, "audio": 0.33, "video": 0.34},
}
