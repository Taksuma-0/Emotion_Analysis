# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional
import requests
import pandas as pd
from tqdm import tqdm

from .utils import extract_final_span, sanitize_llm_output

# -----------------------------
# Helpers numéricos por ventana
# -----------------------------
def _avg_topk_in_window(
    df: Optional[pd.DataFrame], start: float, end: float, mode: str, topk: int = 3
) -> List[Tuple[str, float]]:
    if df is None or not len(df):
        return []
    candidate_cols = [c for c in df.columns if c not in ("start", "end", "time", "label", "score", "text")]
    if not candidate_cols:
        return []
    num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return []
    if mode in ("text", "audio"):
        mask = (df["start"] < end) & (df["end"] > start)
    else:
        mask = (df["time"] >= start) & (df["time"] < end)
    sub = df.loc[mask, num_cols]
    if not len(sub):
        return []
    avg = sub.mean().sort_values(ascending=False)
    return [(k, float(v)) for k, v in avg.head(topk).items()]


def _top1(items: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
    if not items:
        return None, 0.0
    return items[0][0], float(items[0][1])


def _transcript_snippet(df_asr: Optional[pd.DataFrame], start: float, end: float, max_chars: int = 800) -> str:
    if df_asr is None or not len(df_asr):
        return ""
    mask = (df_asr["start"] < end) & (df_asr["end"] > start)
    texts = df_asr.loc[mask, "text"].tolist()
    snippet = " ".join(texts)
    snippet = " ".join(snippet.split())
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
    return snippet


# -----------------------------
# Construcción de lotes/ventanas
# -----------------------------
def build_ollama_batches(
    df_asr,
    text_df,
    audio_df,
    video_df,
    fused_df,
    batch_seconds: int = 60,
    max_chars: int = 800
) -> List[Dict]:
    """Arma ventanas temporales y contexto multimodal (Top-3 por modalidad + snippet + coherencia A↔V)."""
    end_candidates = []
    if audio_df is not None and len(audio_df): end_candidates.append(float(audio_df["end"].max()))
    if video_df is not None and len(video_df): end_candidates.append(float(video_df["time"].max()))
    if df_asr   is not None and len(df_asr):   end_candidates.append(float(df_asr["end"].max()))
    if fused_df is not None and len(fused_df): end_candidates.append(float(fused_df["time"].max()))
    if text_df  is not None and len(text_df):  end_candidates.append(float(text_df["end"].max()))
    if not end_candidates:
        return []

    T = max(end_candidates)
    batches = []
    t = 0.0
    while t < T:
        w_start, w_end = t, min(t + batch_seconds, T)

        # Top-3 por modalidad (incluye TEXTO ahora)
        text_top = _avg_topk_in_window(text_df,  w_start, w_end, "text",  topk=3)
        audio_top = _avg_topk_in_window(audio_df, w_start, w_end, "audio", topk=3)
        video_top = _avg_topk_in_window(video_df, w_start, w_end, "video", topk=3) if (video_df is not None and len(video_df)) else []
        fused_top = _avg_topk_in_window(fused_df, w_start, w_end, "fused", topk=3) if (fused_df is not None and len(fused_df)) else []

        # Coherencia A↔V (simple): acuerdo fuerte si top1 coincide y ambas confianzas >= 0.60
        a_lbl, a_sc = _top1(audio_top)
        v_lbl, v_sc = _top1(video_top)
        if a_lbl and v_lbl:
            if (a_lbl == v_lbl) and (a_sc >= 0.60) and (v_sc >= 0.60):
                coherencia_av = f"acuerdo fuerte ({a_lbl} | audio={a_sc:.2f}, video={v_sc:.2f})"
            elif (a_lbl == v_lbl):
                coherencia_av = f"acuerdo débil ({a_lbl} | audio={a_sc:.2f}, video={v_sc:.2f})"
            else:
                coherencia_av = f"incongruencia (audio={a_lbl}:{a_sc:.2f} vs video={v_lbl}:{v_sc:.2f})"
        else:
            coherencia_av = "sin datos suficientes"

        snippet = _transcript_snippet(df_asr, w_start, w_end, max_chars=max_chars)

        def fmt(items):
            return "; ".join([f"{k}={v:.2f}" for k, v in items]) if items else "sin datos"

        # Bloque estructurado para el LLM (más rico y segmentado)
        batch_text = (
            f"[VENTANA] {w_start:.0f}s–{w_end:.0f}s\n"
            f"[MODALIDADES]\n"
            f"- Texto(top3): {fmt(text_top)}\n"
            f"- Audio(top3): {fmt(audio_top)}\n"
            f"- Video(top3): {fmt(video_top)}\n"
            f"- Fusión(top3): {fmt(fused_top)}\n"
            f"[COHERENCIA_A↔V] {coherencia_av}\n"
            f"[TEXTO] \"{snippet if snippet else 'N/A'}\"\n"
        )

        batches.append({
            "start": w_start,
            "end": w_end,
            "summary": batch_text,
            "tops": {
                "text": text_top, "audio": audio_top, "video": video_top, "fused": fused_top,
                "coherencia_av": coherencia_av
            }
        })
        t = w_end

    return batches


# -----------------------------
# Cliente Ollama (HTTP)
# -----------------------------
def ollama_generate(
    host, model, prompt, num_predict=950, temperature=0.25, top_p=0.9,
    num_ctx=8192, num_thread=None, think=False, timeout=180
):
    import multiprocessing as mp
    if num_thread is None:
        try:
            num_thread = max(4, (mp.cpu_count() or 8) // 2)
        except Exception:
            num_thread = 8

    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(num_predict),
            "num_ctx": int(num_ctx),
            "num_thread": int(num_thread),
            "think": bool(think)  # evitamos razonamientos visibles si el modelo los soporta
        },
        "keep_alive": "30m"
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# -----------------------------
# Resumen global (map-reduce)
# -----------------------------
def build_transcript_summary_with_ollama(df_asr: pd.DataFrame, args) -> str:
    if df_asr is None or not len(df_asr):
        return ""
    chunks, cur, cur_chars = [], [], 0
    target_chars = max(2000, args.batch_max_chars * 2)

    for _, r in df_asr.iterrows():
        frag = f"[{int(r['start']):04d}-{int(r['end']):04d}] {r['text']}"
        if cur_chars + len(frag) > target_chars and cur:
            chunks.append("\n".join(cur))
            cur, cur_chars = [], 0
        cur.append(frag)
        cur_chars += len(frag) + 1
    if cur:
        chunks.append("\n".join(cur))

    partials = []
    for i, ch in enumerate(tqdm(chunks, desc="Ollama resumen (parciales)")):
        prompt = (
            "RESPONDE EXCLUSIVAMENTE EN ESPAÑOL (es-CL). No incluyas <think>.\n"
            "Corrige errores típicos de ASR y resume los puntos clave del fragmento.\n"
            "Devuelve solo el resultado final entre <final> ... </final> en viñetas claras.\n\n"
            f"Fragmento #{i+1}:\n{ch}\n\n<final>"
        )
        resp = ollama_generate(
            host=args.ollama_host, model=args.ollama_model, prompt=prompt,
            num_predict=min(args.ollama_predict, 700),
            temperature=args.ollama_temp, top_p=args.ollama_top_p,
            num_ctx=args.ollama_ctx, timeout=args.ollama_timeout
        )
        out = extract_final_span(resp) or sanitize_llm_output(resp)
        partials.append(f"- Fragmento #{i+1}:\n{out}")

    joined = "\n".join(partials)
    reduce_prompt = (
        "RESPONDE EXCLUSIVAMENTE EN ESPAÑOL (es-CL). No incluyas <think>.\n"
        "Integra los resúmenes parciales en un RESUMEN GLOBAL coherente del discurso.\n"
        "Normaliza errores de ASR en citas breves cuando sea necesario.\n"
        "Devuelve SOLO el resultado entre <final> ... </final>:\n\n"
        f"{joined}\n\n<final>"
    )
    final = ollama_generate(
        host=args.ollama_host, model=args.ollama_model, prompt=reduce_prompt,
        num_predict=min(args.ollama_predict * 2, 1200),
        temperature=args.ollama_temp, top_p=args.ollama_top_p,
        num_ctx=args.ollama_ctx, timeout=max(args.ollama_timeout, 240)
    )
    return extract_final_span(final) or sanitize_llm_output(final)


# -----------------------------
# Análisis psicológico por lotes
# -----------------------------
def run_psych_analysis_with_ollama(
    df_asr, text_df, audio_df, video_df, fused_df, args, transcript_summary: str = ""
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Devuelve:
      - final_clean: síntesis ejecutiva global (más completa y fundamentada)
      - per_batch_summaries: lista de (rango, texto) por ventana, con secciones Audio/Video/Texto/Fusión/Coherencia/Acciones.
    """
    batches = build_ollama_batches(
        df_asr=df_asr,
        text_df=text_df if (text_df is not None and len(text_df)) else None,
        audio_df=audio_df if (audio_df is not None and len(audio_df)) else None,
        video_df=video_df if (video_df is not None and len(video_df)) else None,
        fused_df=fused_df if (fused_df is not None and len(fused_df)) else None,
        batch_seconds=args.batch_seconds,
        max_chars=args.batch_max_chars
    )
    if not batches:
        return "", []

    per_batch_summaries = []
    context_block = f"\n\n[CONTEXTO_GLOBAL]\n{transcript_summary}\n" if transcript_summary else ""

    # Prompt reforzado y segmentado por modalidad
    for b in tqdm(batches, desc="Ollama (ventanas)", total=len(batches)):
        prompt = f"""RESPONDE EXCLUSIVAMENTE EN ESPAÑOL (es-CL). No incluyas <think> ni razonamientos paso a paso.
Eres un analista de psicología del comportamiento. Para la ventana indicada, realiza un ANÁLISIS MULTIMODAL DETALLADO,
siempre fundamentando cada afirmación con la modalidad que la respalda (Audio/Video/Texto/Fusión) y citando explícitamente
emociones Top-3 con probabilidades, y fragmentos textuales entre comillas.

[INSTRUCCIONES DE SALIDA]
- Estructura por secciones con estos encabezados (en este orden): 
  "Audio (prosodia)", "Video (expresión facial)", "Texto (semántica)", "Integración (fusión)", "Coherencia A↔V", "Acciones".
- En cada sección, usa viñetas del tipo: Afirmación — Fundamento: <modalidad y evidencia numérica/textual>.
- "Fundamento" DEBE mencionar explícitamente la modalidad y, cuando aplique, las emociones Top-3 con sus probabilidades
  proporcionadas, o citas literales del snippet de texto.
- Si falta información en alguna modalidad, indícalo ("sin datos suficientes") y continua con las demás.
- Sé conciso, profesional y específico. No repitas lo obvio, no inventes datos.

[BLOQUE TEMPORAL]
{b['summary']}{context_block}

<final>"""
        resp = ollama_generate(
            host=args.ollama_host, model=args.ollama_model,
            prompt=prompt,
            num_predict=args.ollama_predict,
            temperature=args.ollama_temp, top_p=args.ollama_top_p,
            num_ctx=args.ollama_ctx, timeout=args.ollama_timeout
        )
        cleaned = extract_final_span(resp) or sanitize_llm_output(resp)
        rng = f"{b['start']:.0f}s–{b['end']:.0f}s"
        per_batch_summaries.append((rng, cleaned))

    # Síntesis ejecutiva más rica (integra ventanas, coherencia y referencias)
    joined = "\n\n".join([f"[{rng}]\n{txt}" for rng, txt in per_batch_summaries])
    context_for_final = f"[CONTEXTO_GLOBAL]\n{transcript_summary}\n" if transcript_summary else ""
    final_prompt = f"""RESPONDE EXCLUSIVAMENTE EN ESPAÑOL (es-CL). No incluyas <think>.
Elabora una SÍNTESIS EJECUTIVA GLOBAL (8–14 viñetas) integrando:
- Dinámica emocional general (picos, recuperaciones, cambios de valencia) a lo largo de las ventanas.
- Patrones de COHERENCIA/INCONGRUENCIA Audio↔Video y su interpretación conductual.
- Matices semánticos relevantes del texto y cómo se reflejan (o no) en prosodia y expresión facial.
- Recomendaciones estratégicas (tono, pausas, estructura, apoyos visuales).

Formato de cada viñeta:
Afirmación — Fundamento: referencia a ventanas [Xs–Ys], modalidad(es) implicada(s) y evidencia numérica/textual citada.

{context_for_final}
[VENTANAS_ANALIZADAS]
{joined}

Devuelve SOLO el resultado final entre <final> ... </final>.
<final>"""
    final_resp = ollama_generate(
        host=args.ollama_host, model=args.ollama_model,
        prompt=final_prompt,
        num_predict=min(args.ollama_predict * 2, 1400),
        temperature=args.ollama_temp, top_p=args.ollama_top_p,
        num_ctx=args.ollama_ctx, timeout=max(args.ollama_timeout, 240)
    )
    final_clean = extract_final_span(final_resp) or sanitize_llm_output(final_resp)
    return final_clean, per_batch_summaries
