# -*- coding: utf-8 -*-
import tempfile, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

from .visual import set_corp, PALETTE
from .utils import save_plot_as_png, sanitize_llm_output

# Paleta ejecutiva
COLOR_TITLE = colors.HexColor("#0f172a")
COLOR_HEAD  = colors.HexColor("#111827")
COLOR_ACCENT= colors.HexColor("#1f77b4")
COLOR_TEXT  = colors.HexColor("#111827")
COLOR_MUTED = colors.HexColor("#64748b")
COLOR_MUTED2= colors.HexColor("#94a3b8")
COLOR_BGROW = colors.HexColor("#f6f8fa")


# ------------------------
# Gráficos
# ------------------------
def plot_topk_bar(prob_dict: Dict[str, float], title: str, k=6):
    set_corp()
    items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    labels = [x[0] for x in items]
    vals = [x[1] for x in items]
    fig = plt.figure()
    bars = plt.bar(labels, vals, color=PALETTE[:len(labels)])
    plt.ylabel("Probabilidad")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, max(0.75, (max(vals) + 0.1) if vals else 1.0))
    plt.tight_layout()
    return fig


def plot_timeline(df: pd.DataFrame, title: str):
    set_corp()
    if df is None or not len(df):
        fig = plt.figure()
        plt.title(f"{title} (sin datos)")
        return fig
    classes = [c for c in df.columns if c not in ("time", "start", "end", "label", "score", "text")]
    top = df.copy()
    if "time" not in top.columns and "start" in top.columns:
        top = top.rename(columns={"start": "time"})
    max_conf = top[classes].max(axis=1)
    fig = plt.figure()
    plt.plot(top["time"], max_conf, label="Confianza máx.")
    plt.fill_between(top["time"], max_conf, alpha=0.2)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Confianza")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def _classes_from_df(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None or not len(df):
        return []
    return [c for c in df.columns if c not in ("start", "end", "time", "label", "score", "text")]


def _probs_at_time(df: Optional[pd.DataFrame], t: float, mode: str, classes: List[str]) -> Dict[str, float]:
    if df is None or not len(df):
        return {c: 0.0 for c in classes}
    if mode in ("text", "audio"):
        r = df[(df["start"] <= t) & (df["end"] >= t)]
        if len(r) == 0:
            return {c: 0.0 for c in classes}
        r = r.iloc[0]
        return {c: float(r.get(c, 0.0)) for c in classes}
    else:
        r = df.iloc[(df["time"] - t).abs().argsort()[:1]]
        if len(r) == 0:
            return {c: 0.0 for c in classes}
        r = r.iloc[0]
        return {c: float(r.get(c, 0.0)) for c in classes}


def _build_grid(df: Optional[pd.DataFrame], mode: str, step=1.0, t_end: Optional[float] = None) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    if mode in ("text", "audio"):
        T = float(df["end"].max())
    else:
        T = float(df["time"].max())
    if t_end is not None:
        T = min(T, t_end)
    classes = _classes_from_df(df)
    times = np.arange(0, math.ceil(T) + 1e-6, step)
    rows = []
    for t in times:
        p = _probs_at_time(df, t, mode, classes)
        rows.append({"time": t, **p})
    return pd.DataFrame(rows)


def plot_top3_comparative(
    text_df: Optional[pd.DataFrame],
    audio_df: Optional[pd.DataFrame],
    video_df: Optional[pd.DataFrame],
    fused_df: Optional[pd.DataFrame],
    title: str = "Comparativa temporal (Top-3 emociones) — Texto vs Audio vs Video",
    step: float = 1.0
):
    set_corp()

    def _avg(df):
        if df is None or not len(df):
            return {}
        cols = [c for c in df.columns if c not in ("start", "end", "time", "label", "score", "text")]
        if not cols:
            return {}
        return df[cols].mean().to_dict()

    avg_fused = _avg(fused_df) if fused_df is not None and len(fused_df) else {}
    if avg_fused:
        top3 = [k for k, _ in sorted(avg_fused.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    else:
        merged = {}
        for m in [text_df, audio_df, video_df]:
            a = _avg(m)
            for k, v in a.items():
                merged[k] = merged.get(k, 0.0) + float(v)
        top3 = [k for k, _ in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:3]]

    def _build_grid_local(df: Optional[pd.DataFrame], mode: str, step=1.0, t_end: Optional[float] = None) -> pd.DataFrame:
        if df is None or not len(df):
            return pd.DataFrame()
        if mode in ("text", "audio"):
            T = float(df["end"].max())
        else:
            T = float(df["time"].max())
        if t_end is not None:
            T = min(T, t_end)
        classes = [c for c in df.columns if c not in ("start", "end", "time", "label", "score", "text")]
        times = np.arange(0, math.ceil(T) + 1e-6, step)
        rows = []
        for t in times:
            if mode in ("text", "audio"):
                r = df[(df["start"] <= t) & (df["end"] >= t)]
                if len(r):
                    r = r.iloc[0]
                    rows.append({"time": t, **{c: float(r.get(c, 0.0)) for c in classes}})
                else:
                    rows.append({"time": t, **{c: 0.0 for c in classes}})
            else:
                r = df.iloc[(df["time"] - t).abs().argsort()[:1]]
                if len(r):
                    r = r.iloc[0]
                    rows.append({"time": t, **{c: float(r.get(c, 0.0)) for c in classes}})
                else:
                    rows.append({"time": t, **{c: 0.0 for c in classes}})
        return pd.DataFrame(rows)

    tg = _build_grid_local(text_df, "text", step=step) if text_df is not None and len(text_df) else pd.DataFrame()
    ag = _build_grid_local(audio_df, "audio", step=step) if audio_df is not None and len(audio_df) else pd.DataFrame()
    vg = _build_grid_local(video_df, "video", step=step) if video_df is not None and len(video_df) else pd.DataFrame()

    T_candidates = []
    for g in [tg, ag, vg]:
        if len(g):
            T_candidates.append(float(g["time"].max()))
    if not T_candidates:
        fig = plt.figure()
        plt.title(f"{title} (sin datos)")
        return fig
    T = max(T_candidates)
    times = np.arange(0, math.ceil(T) + 1e-6, step)

    color_map = {emo: PALETTE[i % len(PALETTE)] for i, emo in enumerate(top3)}
    style_map = {"Texto": "-", "Audio": "--", "Video": ":"}
    marker_map = {"Texto": "o", "Audio": "s", "Video": "^"}

    fig = plt.figure(figsize=(10.5, 6.2))

    def _series(grid, label_prefix):
        if grid is None or not len(grid):
            return
        g = grid.set_index("time").reindex(times, method="nearest", fill_value=0.0)
        for emo in top3:
            if emo not in g.columns:
                continue
            y = g[emo].values
            plt.plot(
                times, y,
                label=f"{emo} ({label_prefix})",
                color=color_map.get(emo, None),
                linestyle=style_map.get(label_prefix, "-"),
                marker=marker_map.get(label_prefix, None),
                markersize=3.5,
                linewidth=2.2,
                alpha=0.95,
            )

    _series(tg, "Texto")
    _series(ag, "Audio")
    _series(vg, "Video")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Probabilidad")
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, times[-1] if len(times) else 1.0)
    plt.grid(True, alpha=0.25, linestyle=":")
    plt.title(title, fontweight="bold")

    handles, labels = plt.gca().get_legend_handles_labels()
    ncols = min(4, max(1, len(labels)))
    plt.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncols,
        frameon=False,
        fontsize=9,
        handlelength=3.0,
        columnspacing=1.2,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    return fig


# ------------------------
# Encabezado/Pie
# ------------------------
def _on_page(c: canvas.Canvas, doc):
    c.setFont("Helvetica", 9)
    c.setFillColor(COLOR_MUTED)
    c.drawString(2 * cm, A4[1] - 1.2 * cm, "Reporte Multimodal de Emociones")

    c.setFont("Helvetica", 9)
    c.setFillColor(COLOR_MUTED2)
    c.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Página {doc.page}")


# ------------------------
# PDF
# ------------------------
def make_pdf_report(
    out_pdf: str, meta: Dict, text_df, audio_df, video_df, fused_df,
    insights_text: str, kpis: List[List[str]],
    ollama_global: str = "", ollama_batches: Optional[List[Tuple[str, str]]] = None,
    structured_report: Optional[dict] = None,
    transcript_summary: str = "", model_name: str = ""
):
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=24,
        textColor=COLOR_TITLE, alignment=1, spaceAfter=12
    )
    style_h = ParagraphStyle(
        "H2Tight", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=COLOR_HEAD, spaceBefore=10, spaceAfter=6, keepWithNext=True
    )
    style_p = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=10.5, leading=14.5,
        textColor=COLOR_TEXT, spaceAfter=6
    )

    doc = SimpleDocTemplate(
        out_pdf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=1.8 * cm
    )

    story = []

    portada = [
        Paragraph("<b>Reporte Multimodal de Emociones (ES)</b>", style_title),
        Spacer(1, 0.2 * cm),
        Paragraph(f"Origen: {meta.get('source_desc','')}", style_p),
        Paragraph(
            f"Modelos: texto={meta['text_model']}, audio={meta['audio_model']}, video={meta.get('video_model','-')}",
            style_p
        ),
        Paragraph(f"Pesos de fusión: {meta.get('fusion_weights',{})}", style_p),
    ]
    if model_name:
        portada.append(Paragraph(f"Análisis (Ollama): {model_name}", style_p))
    portada.append(Spacer(1, 0.4 * cm))
    story.append(KeepTogether(portada))

    if kpis:
        table_data = [["Métrica", "Valor"]] + kpis
        tbl = Table(table_data, hAlign="LEFT", colWidths=[8 * cm, 7 * cm], repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, COLOR_BGROW]),
        ]))
        story.append(KeepTogether([
            Paragraph("<b>Indicadores Clave</b>", style_h),
            tbl,
            Spacer(1, 0.5 * cm),
        ]))

    if transcript_summary:
        bloque = [Paragraph("<b>Resumen del Discurso (Ollama)</b>", style_h)]
        for line in transcript_summary.split("\n"):
            bloque.append(Paragraph(line, style_p))
        bloque.append(Spacer(1, 0.5 * cm))
        story.append(KeepTogether(bloque))

    bloque_ins = [Paragraph("<b>Insights Ejecutivos (modelos)</b>", style_h)]
    for line in insights_text.split("\n"):
        bloque_ins.append(Paragraph(line, style_p))
    bloque_ins.append(Spacer(1, 0.5 * cm))
    story.append(KeepTogether(bloque_ins))

    images = []
    for name, df in [("Texto", text_df), ("Audio", audio_df), ("Video", video_df), ("Fusión", fused_df)]:
        if df is None or not len(df):
            continue
        classes = [c for c in df.columns if c not in ("time", "start", "end", "label", "score", "text")]
        if not classes:
            continue
        avg = df[classes].mean().to_dict()
        fig = plot_topk_bar(avg, f"Top emociones promedio — {name}")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        save_plot_as_png(fig, tmp.name)
        images.append((name, tmp.name))

    if images:
        story.append(Paragraph("<b>Distribución promedio por modalidad</b>", style_h))
        for _, img in images:
            story.append(KeepTogether([
                RLImage(img, width=15 * cm, height=10 * cm),
                Spacer(1, 0.3 * cm),
            ]))

    comp_fig = plot_top3_comparative(
        text_df if (text_df is not None and len(text_df)) else None,
        audio_df if (audio_df is not None and len(audio_df)) else None,
        video_df if (video_df is not None and len(video_df)) else None,
        fused_df if (fused_df is not None and len(fused_df)) else None,
    )
    tmpc = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    save_plot_as_png(comp_fig, tmpc.name)
    story.append(KeepTogether([
        Paragraph("<b>Comparativa temporal (Top-3) — Texto vs Audio vs Video</b>", style_h),
        RLImage(tmpc.name, width=16 * cm, height=8.5 * cm),
        Spacer(1, 0.4 * cm),
    ]))

    td = None if text_df is None or not len(text_df) else text_df.copy()
    if td is not None and "time" not in td.columns and "start" in td.columns:
        td = td.rename(columns={"start": "time"})
    ad = None if audio_df is None or not len(audio_df) else audio_df.copy()
    if ad is not None and "time" not in ad.columns and "start" in ad.columns:
        ad = ad.rename(columns={"start": "time"})

    for name, df in [
        ("Línea de tiempo — Texto", td),
        ("Línea de tiempo — Audio", ad),
        ("Línea de tiempo — Video", video_df),
        ("Línea de tiempo — Fusión", fused_df)
    ]:
        if df is None or not len(df):
            continue
        fig = plot_timeline(df, name)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        save_plot_as_png(fig, tmp.name)
        story.append(KeepTogether([
            RLImage(tmp.name, width=16 * cm, height=7.5 * cm),
            Spacer(1, 0.2 * cm),
        ]))

    # --------- ÚNICO bloque para Ventanas con Ollama ----------
    if ollama_global or (ollama_batches and len(ollama_batches)):
        story.append(PageBreak())
        story.append(Paragraph("<b>Análisis Psicológico por Ventanas (Ollama)</b>", style_h))

        if ollama_global:
            story.append(Paragraph("<b>Síntesis global</b>", style_h))
            for line in ollama_global.split("\n"):
                story.append(Paragraph(line, style_p))
            story.append(Spacer(1, 0.4 * cm))

        if ollama_batches:
            # sin subtítulo “Resumen por ventanas”: listamos directamente cada rango
            for rng, txt in ollama_batches:
                story.append(Paragraph(f"<b>[{rng}]</b>", style_p))
                for line in txt.split("\n"):
                    story.append(Paragraph(line, style_p))
                story.append(Spacer(1, 0.2 * cm))

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
