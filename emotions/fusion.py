# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import math

def _classes_from_dfs(dfs: List[Optional[pd.DataFrame]]) -> List[str]:
    classes = set()
    for df in dfs:
        if df is None or not len(df):
            continue
        classes.update([c for c in df.columns if c not in ("start","end","time","label","score","text")])
    return sorted(list(classes))

def _probs_at_time(df, t, mode, classes):
    if df is None or not len(df):
        return {c:0.0 for c in classes}
    if mode in ("text","audio"):
        r = df[(df["start"]<=t) & (df["end"]>=t)]
        if len(r)==0: return {c:0.0 for c in classes}
        r = r.iloc[0]
        return {c: float(r.get(c, 0.0)) for c in classes}
    if mode in ("video","fused"):
        r = df.iloc[(df["time"]-t).abs().argsort()[:1]]
        r = r.iloc[0] if len(r) else None
        if r is None: return {c:0.0 for c in classes}
        return {c: float(r.get(c, 0.0)) for c in classes}
    return {c:0.0 for c in classes}

def build_grid(df: Optional[pd.DataFrame], mode: str, step=1.0, t_end: Optional[float]=None) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    if mode in ("text","audio"):
        T = df["end"].max()
    else:
        T = df["time"].max()
    if t_end is not None:
        T = min(T, t_end)
    classes = [c for c in df.columns if c not in ("start","end","time","label","score","text")]
    times = np.arange(0, math.ceil(T)+1e-6, step)
    rows = []
    for t in times:
        p = _probs_at_time(df, t, mode, classes)
        top_label = max(p.items(), key=lambda kv: kv[1])[0] if p else None
        top_score = p.get(top_label, 0.0) if top_label else 0.0
        rows.append({"time": t, "label": top_label, "score": top_score, **p})
    return pd.DataFrame(rows)

def align_and_fuse(text_df, audio_df, video_df, weights: Dict[str, float], classes_union=None, step=1.0, t_end=None):
    end_candidates = []
    if text_df is not None and len(text_df): end_candidates.append(text_df["end"].max())
    if audio_df is not None and len(audio_df): end_candidates.append(audio_df["end"].max())
    if video_df is not None and len(video_df): end_candidates.append(video_df["time"].max())
    if not end_candidates:
        return pd.DataFrame()
    T = t_end or max(end_candidates)
    times = np.arange(0, math.ceil(T)+1e-6, step)

    classes = set(classes_union) if classes_union else _classes_from_dfs([text_df,audio_df,video_df])

    rows = []
    for t in times:
        pt = _probs_at_time(text_df, t, "text", classes)
        pa = _probs_at_time(audio_df, t, "audio", classes)
        pv = _probs_at_time(video_df, t, "video", classes)
        fused = {c: weights.get("text",0)*pt[c] + weights.get("audio",0)*pa[c] + weights.get("video",0)*pv[c] for c in classes}
        fused["time"] = t
        rows.append(fused)
    return pd.DataFrame(rows)

def compute_insights(audio_df: Optional[pd.DataFrame], video_df: Optional[pd.DataFrame], text_df: Optional[pd.DataFrame], fused_df: Optional[pd.DataFrame]) -> Tuple[str, List[List[str]]]:
    lines = []
    kpis = []

    def avg_top(df, name):
        if df is None or not len(df): return None, 0.0
        classes = [c for c in df.columns if c not in ("start","end","time","label","score","text")]
        if not classes: return None, 0.0
        avg = df[classes].mean()
        top = avg.idxmax()
        val = float(avg.max())
        kpis.append([f"Emoción dominante ({name})", f"{top} ({val:.2f})"])
        return top, val

    a_top, a_val = avg_top(audio_df, "audio")
    v_top, v_val = avg_top(video_df, "video")
    t_top, t_val = avg_top(text_df, "texto")
    f_top, f_val = avg_top(fused_df, "fusión")

    if audio_df is not None and len(audio_df) and video_df is not None and len(video_df):
        T = min(audio_df["end"].max(), video_df["time"].max())
        g_audio = build_grid(audio_df, "audio", step=1.0, t_end=T)
        g_video = build_grid(video_df, "video", step=1.0, t_end=T)
        m = min(len(g_audio), len(g_video))
        agree = strong_agree = 0
        strong_diverge = []
        for i in range(m):
            la, sa = g_audio.loc[i, "label"], g_audio.loc[i, "score"]
            lv, sv = g_video.loc[i, "label"], g_video.loc[i, "score"]
            if la == lv and la is not None:
                agree += 1
                if sa >= 0.60 and sv >= 0.60:
                    strong_agree += 1
            else:
                if sa >= 0.60 and sv >= 0.60:
                    strong_diverge.append((g_audio.loc[i, "time"], la, sa, lv, sv))
        agree_rate = (agree / m) * 100 if m else 0.0
        strong_rate = (strong_agree / m) * 100 if m else 0.0
        kpis.append(["Acuerdo audio↔video (1s)", f"{agree_rate:.1f}%"])
        kpis.append(["Acuerdo fuerte (≥0.60 ambos)", f"{strong_rate:.1f}%"])
        for t, la, sa, lv, sv in strong_diverge[:3]:
            lines.append(f"- t={t:.0f}s: audio={la} ({sa:.2f}) vs video={lv} ({sv:.2f})")

    if fused_df is not None and len(fused_df):
        classes = [c for c in fused_df.columns if c not in ("time","start","end","text")]
        top_series = fused_df[classes].max(axis=1)
        times = fused_df["time"].values
        idxs = np.argsort(-top_series.values)[:3]
        peaks = [(times[i], top_series.iloc[i]) for i in idxs]
        lines.append("<b>Momentos de mayor intensidad emocional (fusión)</b>:")
        for t, v in peaks:
            lines.append(f"- t={t:.0f}s con confianza {v:.2f}")

    if (video_df is None or not len(video_df)) and (audio_df is not None and len(audio_df)):
        voiced_sec = float((audio_df["end"] - audio_df["start"]).clip(lower=0).sum())
        kpis.append(["Duración con voz (estimada)", f"{voiced_sec:.1f} s"])
        lines.append("<b>Resumen audio</b>: Predomina la emoción acústica; valida con texto y expresión facial si están disponibles.")

    if text_df is not None and len(text_df):
        lines.append("<b>Notas sobre texto</b>: emociones derivadas de la transcripción; útiles para matices semánticos.")

    if not lines:
        lines.append("No se encontraron señales suficientes para un resumen detallado.")
    return ("\n".join(lines), kpis)
