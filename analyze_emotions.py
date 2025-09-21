#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal Emotion Analysis (ES): Video + Audio + Texto + (opcional) Análisis psicológico con Ollama
(versión modular con integración del análisis estructurado, contexto multimodal y batches siempre al final)
"""
import os, argparse, yaml, sys, subprocess, json
from pathlib import Path
import pandas as pd
import cv2

# Importa módulos del paquete
from emotions.config import DEFAULT_CFG
from emotions.pipelines import load_pipelines
from emotions.utils import extract_audio_ffmpeg
from emotions.inference import asr_transcribe, text_emotions, audio_emotions, video_emotions
from emotions.fusion import align_and_fuse, compute_insights
from emotions.ollama_utils import (
    build_transcript_summary_with_ollama,
    run_psych_analysis_with_ollama,
    build_ollama_batches
)
from emotions.reporting import make_pdf_report


def _write_transcript_txt(df_asr: pd.DataFrame, out_path: Path):
    """Crea un .txt legible a partir de la transcripción para el analizador estructurado."""
    lines = []
    for _, r in df_asr.iterrows():
        lines.append(f"[{int(r['start']):04d}-{int(r['end']):04d}] {r['text']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Analysis (ES) — video+audio+texto → PDF (+ Ollama opcional)")
    parser.add_argument("--video", help="Ruta al archivo de video (mp4/mov/avi)")
    parser.add_argument("--audio", help="Ruta a un archivo de audio (wav/mp3/m4a). Si se especifica junto a --video, se prioriza este audio.")
    parser.add_argument("--out", default="out_report.pdf", help="PDF de salida")
    parser.add_argument("--config", default="config.yaml", help="YAML de configuración")
    parser.add_argument("--language", default="es", help="Código de idioma para ASR Whisper (ej. es)")
    parser.add_argument("--fps", type=float, default=None, help="Forzar FPS efectivo (submuestreo temporal)")

    # Ollama flags (originales)
    parser.add_argument("--ollama", action="store_true", help="Habilita análisis con Ollama (deepseek-r1:7b)")
    parser.add_argument("--ollama_host", default="http://localhost:11434", help="Host de Ollama")
    parser.add_argument("--ollama_model", default="deepseek-r1:7b", help="Modelo en Ollama")
    parser.add_argument("--batch_seconds", type=int, default=60, help="Tamaño de ventana para lotes a Ollama")
    parser.add_argument("--batch_max_chars", type=int, default=800, help="Máx. caracteres de transcripción por lote")
    parser.add_argument("--ollama_predict", type=int, default=950, help="Tokens por respuesta de Ollama")
    parser.add_argument("--ollama_temp", type=float, default=0.25, help="Temperatura Ollama")
    parser.add_argument("--ollama_top_p", type=float, default=0.9, help="Top-p Ollama")
    parser.add_argument("--ollama_ctx", type=int, default=8192, help="Contexto (num_ctx) Ollama")
    parser.add_argument("--ollama_timeout", type=int, default=180, help="Timeout por request a Ollama (s)")
    parser.add_argument("--ollama_transcript_summary", action="store_true", help="Genera resumen global del discurso (map-reduce) y úsalo como contexto.")

    # Análisis psicológico estructurado (JSON/TXT)
    parser.add_argument("--ollama_structured", action="store_true",
                        help="Ejecuta análisis psicológico estructurado vía analyze_txt_ollama.py sobre la transcripción")
    parser.add_argument("--structured_out_json", default="outputs/analisis.json",
                        help="Ruta de salida JSON del análisis estructurado")
    parser.add_argument("--structured_out_txt", default="outputs/analisis.txt",
                        help="Ruta de salida TXT legible del análisis estructurado")

    args = parser.parse_args()

    if not args.video and not args.audio:
        raise SystemExit("Debes pasar --video o --audio.")

    # Config
    if os.path.exists(args.config):
        cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) or {}
        for k, v in DEFAULT_CFG.items():
            cfg.setdefault(k, v)
    else:
        cfg = DEFAULT_CFG.copy()

    print("[1/6] Cargando pipelines...")
    pipes = load_pipelines(cfg)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Audio de entrada
    if args.audio:
        source_desc = f"audio={args.audio}"
        wav_path = args.audio
        print("[2/6] Usando audio proporcionado…")
    else:
        if not args.video:
            raise SystemExit("Si no das --audio, debes dar --video.")
        source_desc = f"video={args.video}"
        print("[2/6] Extrayendo audio con FFmpeg...")
        wav_path = str(outputs_dir / "audio_16k.wav")
        extract_audio_ffmpeg(args.video, wav_path, sr=16000)

    # ASR
    print("[3/6] Transcribiendo audio (Whisper)...")
    df_asr = asr_transcribe(pipes, wav_path, language=args.language)
    df_asr.to_csv(outputs_dir / "transcript.csv", index=False)

    # Texto → emociones
    print("[4/6] Emociones desde texto...")
    text_df = text_emotions(pipes, df_asr)
    text_df.to_csv(outputs_dir / "emotions_text.csv", index=False)

    # Audio → emociones
    print("[5/6] Emociones desde audio (VAD + segments)...")
    audio_df = audio_emotions(pipes, wav_path, cfg)
    audio_df.to_csv(outputs_dir / "emotions_audio.csv", index=False)

    # Video → emociones (si hay video)
    video_df = pd.DataFrame()
    if args.video:
        print("[6/6] Emociones desde video (frames + face)...")
        if args.fps is not None and args.fps > 0:
            cap = cv2.VideoCapture(args.video)
            real_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            interval = max(1, int(real_fps / args.fps))
            cfg["sample_every_n_frames"] = interval
            print(f"Usando muestreo de 1 frame cada {interval} frames (fps real ~ {real_fps:.2f})")
        video_df = video_emotions(pipes, args.video, cfg)
        video_df.to_csv(outputs_dir / "emotions_video.csv", index=False)

    # Fusión tardía (si hay video)
    fused_df = pd.DataFrame()
    if args.video and len(video_df):
        print("Fusionando modalidades...")
        fused_df = align_and_fuse(
            text_df=text_df if len(text_df) else None,
            audio_df=audio_df if len(audio_df) else None,
            video_df=video_df if len(video_df) else None,
            weights=cfg["fusion_weights"],
            step=1.0
        )
        if len(fused_df):
            fused_df.to_csv(outputs_dir / "emotions_fused.csv", index=False)

    # Insights (nuestros)
    print("Generando insights…")
    insights_text, kpis = compute_insights(
        audio_df=audio_df if len(audio_df) else None,
        video_df=video_df if len(video_df) else None,
        text_df=text_df if len(text_df) else None,
        fused_df=fused_df if len(fused_df) else None
    )

    # (Opcional) Resumen de la transcripción completo (map-reduce)
    transcript_summary = ""
    if args.ollama and args.ollama_transcript_summary:
        print("Ollama: construyendo resumen global del discurso (map-reduce)…")
        transcript_summary = build_transcript_summary_with_ollama(df_asr, args)
        with open(outputs_dir / "ollama_transcript_summary.txt", "w", encoding="utf-8") as f:
            f.write(transcript_summary or "")

    # Análisis psicológico por lotes + síntesis global
    # AHORA: si --ollama está activo, SIEMPRE calculamos batches (se insertan al final del PDF)
    ollama_global, ollama_batches = "", []
    if args.ollama:
        print("Ollama: análisis psicológico (por lotes)…")
        ollama_global, ollama_batches = run_psych_analysis_with_ollama(
            df_asr=df_asr if len(df_asr) else None,
            text_df=text_df if len(text_df) else None,      # <-- CAMBIO: se añade text_df
            audio_df=audio_df if len(audio_df) else None,
            video_df=video_df if len(video_df) else None,
            fused_df=fused_df if len(fused_df) else None,
            args=args,
            transcript_summary=transcript_summary
        )
        with open(outputs_dir / "ollama_analysis_global.txt", "w", encoding="utf-8") as f:
            f.write(ollama_global or "")
        with open(outputs_dir / "ollama_analysis_batches.txt", "w", encoding="utf-8") as f:
            for rng, txt in ollama_batches:
                f.write(f"[{rng}]\n{txt}\n\n")

    # Análisis psicológico estructurado (JSON + TXT) con CONTEXTO MULTIMODAL
    structured_report = None
    if args.ollama_structured:
        print("Ollama (estructurado): generando transcripción .txt y contexto multimodal…")
        transcript_txt = outputs_dir / "transcript_for_psych.txt"
        _write_transcript_txt(df_asr, transcript_txt)

        # --- Construir contexto multimodal para enriquecer el análisis estructurado ---
        def _avg(df):
            if df is None or not len(df): return {}
            cols = [c for c in df.columns if c not in ("time","start","end","label","score","text")]
            if not cols: return {}
            return {k: float(v) for k, v in df[cols].mean().sort_values(ascending=False).items()}

        ctx_batches = build_ollama_batches(
            df_asr=df_asr if len(df_asr) else None,
            text_df=text_df if len(text_df) else None,
            audio_df=audio_df if len(audio_df) else None,
            video_df=video_df if len(video_df) else None,
            fused_df=fused_df if len(fused_df) else None,
            batch_seconds=args.batch_seconds,
            max_chars=args.batch_max_chars
        )

        context_payload = {
            "kpis": kpis,                                 # de compute_insights()
            "insights_text": insights_text,               # párrafos internos
            "averages": {
                "texto": _avg(text_df if len(text_df) else None),
                "audio": _avg(audio_df if len(audio_df) else None),
                "video": _avg(video_df if len(video_df) else None),
                "fusion": _avg(fused_df if len(fused_df) else None),
            },
            "batches": ctx_batches                        # ventanas con top3 por modalidad + snippet
        }

        context_json_path = outputs_dir / "structured_context.json"
        context_json_path.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # --- Llamar al analizador estructurado con contexto ---
        cmd = [
            sys.executable, "analyze_txt_ollama.py",
            "--txt", str(transcript_txt),
            "--model", args.ollama_model,
            "--num_ctx", str(args.ollama_ctx),
            "--temperature", str(args.ollama_temp),
            "--out_txt", str(Path(args.structured_out_txt)),
            "--out_json", str(Path(args.structured_out_json)),
            "--context_json", str(context_json_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"[OK] Estructurado JSON -> {Path(args.structured_out_json).resolve()}")
        print(f"[OK] Estructurado TXT  -> {Path(args.structured_out_txt).resolve()}")

        with open(args.structured_out_json, "r", encoding="utf-8") as f:
            structured_report = json.load(f)

    # PDF
    print("Generando PDF...")
    meta = dict(
        source_desc=source_desc,
        text_model=cfg["text_model"],
        audio_model=cfg["audio_emotion_model"],
        video_model=cfg.get("video_emotion_model","-"),
        fusion_weights=cfg.get("fusion_weights",{})
    )
    make_pdf_report(
        args.out, meta,
        text_df, audio_df,
        video_df if len(video_df) else None,
        fused_df if len(fused_df) else None,
        insights_text, kpis,
        ollama_global=ollama_global,              # ← ahora siempre posible si --ollama
        ollama_batches=ollama_batches,
        structured_report=structured_report,      # ← embebido en el PDF
        transcript_summary=transcript_summary,
        model_name=(args.ollama_model if (args.ollama or args.ollama_structured) else "")
    )
    print(f"Listo: {args.out}")


if __name__ == "__main__":
    main()
