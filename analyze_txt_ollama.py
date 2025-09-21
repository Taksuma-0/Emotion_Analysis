#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, pathlib, re, sys
import ollama

DEFAULT_SYSTEM = (
    "Eres un analista en español. Realiza un ANÁLISIS PSICOLÓGICO DETALLADO del HABLANTE "
    "a partir de la transcripción. No hagas diagnósticos clínicos ni etiquetas médicas; "
    "describe patrones observables y probables rasgos. Devuelve SOLO un JSON válido con las claves:\n"
    "resumen: string\n"
    "topicos: string[]\n"
    "sentimiento: string  # ej. positivo/negativo/mixto/neutral\n"
    "perfil_psicologico: {\n"
    "  rasgos: string[],                # p.ej. asertivo, perfeccionista, cooperativo\n"
    "  motivaciones: string[],          # metas, incentivos percibidos\n"
    "  emociones_dominantes: string[],  # emociones más presentes\n"
    "  estilo_comunicacion: string[],   # directo, narrativo, analítico, evasivo, etc.\n"
    "  sesgos_cognitivos: string[],     # heurísticas/posibles sesgos\n"
    "  necesidades: string[],           # lo que parece buscar del interlocutor/entorno\n"
    "  valores: string[],               # lo que aprecia/defiende\n"
    "  estresores: string[],            # fuentes de presión o conflicto\n"
    "  red_flags: string[]              # señales de alerta NO clínicas (inconsistencias, agresividad, etc.)\n"
    "}\n"
    "evidencia: string[]                # frases/fragmentos literales que sustentan el análisis\n"
    "acciones: string[]                 # recomendaciones o próximos pasos\n"
    "entidades: { personas: string[], organizaciones: string[], lugares: string[] }\n"
    "fechas: string[]\n"
    "riesgos: string[]\n"
)

def parse_args():
    ap = argparse.ArgumentParser(description="Analiza una transcripción .txt con un LLM vía Ollama (enfoque psicológico)")
    ap.add_argument("--txt", default="salida.txt", help="Ruta al archivo de transcripción (.txt)")
    ap.add_argument("--model", default="deepseek-r1",
                    help="Modelo Ollama: mistral | qwen2:7b-instruct | phi3:mini ...")
    ap.add_argument("--num_ctx", type=int, default=4096, help="Contexto del modelo")
    ap.add_argument("--temperature", type=float, default=0.2, help="Creatividad (0 = determinista)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM, help="Mensaje de sistema")
    ap.add_argument("--out_txt", default="analisis.txt", help="Archivo de salida (informe legible)")
    ap.add_argument("--out_json", default="analisis.json", help="Archivo JSON con la respuesta estructurada")
    ap.add_argument("--context_json", default=None, help="Ruta a un JSON con contexto multimodal para enriquecer el análisis")
    return ap.parse_args()

def extract_json(text: str):
    """Intenta extraer JSON de la respuesta del modelo (maneja code-fences y texto extra)."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL)
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                import ast
                return ast.literal_eval(candidate)
            except Exception:
                pass
    raise ValueError("No se pudo extraer JSON válido de la respuesta del modelo.")

def coerce_schema(obj: dict):
    """Asegura que existan las claves esperadas con tipos razonables."""
    def as_list(x):
        if x is None: return []
        if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
        return [str(x).strip()] if str(x).strip() else []

    def as_str(x, default=""):
        return str(x).strip() if x is not None else default

    # Perfil psicológico
    pp = obj.get("perfil_psicologico", {}) or {}
    perfil_psicologico = {
        "rasgos": as_list(pp.get("rasgos")),
        "motivaciones": as_list(pp.get("motivaciones")),
        "emociones_dominantes": as_list(pp.get("emociones_dominantes")),
        "estilo_comunicacion": as_list(pp.get("estilo_comunicacion")),
        "sesgos_cognitivos": as_list(pp.get("sesgos_cognitivos")),
        "necesidades": as_list(pp.get("necesidades")),
        "valores": as_list(pp.get("valores")),
        "estresores": as_list(pp.get("estresores")),
        "red_flags": as_list(pp.get("red_flags")),
    }

    entidades_in = obj.get("entidades", {}) or {}
    entidades = {
        "personas": as_list(entidades_in.get("personas")),
        "organizaciones": as_list(entidades_in.get("organizaciones")),
        "lugares": as_list(entidades_in.get("lugares")),
    }

    return {
        "resumen": as_str(obj.get("resumen")),
        "topicos": as_list(obj.get("topicos")),
        "sentimiento": as_str(obj.get("sentimiento")),
        "perfil_psicologico": perfil_psicologico,
        "evidencia": as_list(obj.get("evidencia")),
        "acciones": as_list(obj.get("acciones")),
        "entidades": entidades,
        "fechas": as_list(obj.get("fechas")),
        "riesgos": as_list(obj.get("riesgos")),
    }

def render_text(report: dict) -> str:
    """Devuelve un informe legible en secciones."""
    lines = []
    lines.append("# Informe de análisis (énfasis psicológico)")
    lines.append("")
    if report["resumen"]:
        lines.append("## Resumen")
        lines.append(report["resumen"])
        lines.append("")
    if report["topicos"]:
        lines.append("## Tópicos")
        for t in report["topicos"]:
            lines.append(f"- {t}")
        lines.append("")
    if report["sentimiento"]:
        lines.append("## Sentimiento global")
        lines.append(report["sentimiento"])
        lines.append("")
    # --- Perfil psicológico detallado ---
    pp = report.get("perfil_psicologico", {}) or {}
    if any(pp.values()):
        lines.append("## Perfil psicológico detallado (no clínico)")
        if pp.get("rasgos"):
            lines.append("**Rasgos**")
            for x in pp["rasgos"]: lines.append(f"- {x}")
        if pp.get("motivaciones"):
            lines.append("**Motivaciones**")
            for x in pp["motivaciones"]: lines.append(f"- {x}")
        if pp.get("emociones_dominantes"):
            lines.append("**Emociones dominantes**")
            for x in pp["emociones_dominantes"]: lines.append(f"- {x}")
        if pp.get("estilo_comunicacion"):
            lines.append("**Estilo de comunicación**")
            for x in pp["estilo_comunicacion"]: lines.append(f"- {x}")
        if pp.get("sesgos_cognitivos"):
            lines.append("**Sesgos cognitivos (posibles)**")
            for x in pp["sesgos_cognitivos"]: lines.append(f"- {x}")
        if pp.get("necesidades"):
            lines.append("**Necesidades**")
            for x in pp["necesidades"]: lines.append(f"- {x}")
        if pp.get("valores"):
            lines.append("**Valores**")
            for x in pp["valores"]: lines.append(f"- {x}")
        if pp.get("estresores"):
            lines.append("**Estresores**")
            for x in pp["estresores"]: lines.append(f"- {x}")
        if pp.get("red_flags"):
            lines.append("**Red flags (no clínicas)**")
            for x in pp["red_flags"]: lines.append(f"- {x}")
        lines.append("")
    # --- Evidencia ---
    if report["evidencia"]:
        lines.append("## Evidencia (frases clave)")
        for e in report["evidencia"]:
            lines.append(f"- “{e}”")
        lines.append("")
    # --- Acciones/Entidades/Fechas/Riesgos ---
    if report["acciones"]:
        lines.append("## Acciones sugeridas")
        for a in report["acciones"]:
            lines.append(f"- {a}")
        lines.append("")
    ent = report["entidades"]
    if any([ent["personas"], ent["organizaciones"], ent["lugares"]]):
        lines.append("## Entidades")
        if ent["personas"]:
            lines.append("**Personas**");  [lines.append(f"- {p}") for p in ent["personas"]]
        if ent["organizaciones"]:
            lines.append("**Organizaciones**");  [lines.append(f"- {o}") for o in ent["organizaciones"]]
        if ent["lugares"]:
            lines.append("**Lugares**");  [lines.append(f"- {l}") for l in ent["lugares"]]
        lines.append("")
    if report["fechas"]:
        lines.append("## Fechas / Hitos")
        for f in report["fechas"]:
            lines.append(f"- {f}")
        lines.append("")
    if report["riesgos"]:
        lines.append("## Riesgos / Ambigüedades")
        for r in report["riesgos"]:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("_Aviso: este informe es orientativo y no constituye evaluación clínica ni diagnóstico._")
    return "\n".join(lines).strip() + "\n"

def main():
    args = parse_args()

    # 1) Leer el .txt
    txt_path = pathlib.Path(args.txt)
    if not txt_path.exists():
        print(f"[ERROR] No existe el archivo: {txt_path}")
        sys.exit(1)
    text = txt_path.read_text(encoding="utf-8")

    # 1.b) (Opcional) Cargar contexto multimodal JSON
    context_block = ""
    if args.context_json:
        cj = pathlib.Path(args.context_json)
        if cj.exists():
            ctx_raw = cj.read_text(encoding="utf-8")
            context_block = f"\n[CONTEXTO_MULTIMODAL]\n{ctx_raw}\n"

    # 2) Armar prompt (inyecta contexto multimodal si existe)
    prompt = (
        f"[SYSTEM]\n{args.system}\n"
        f"{context_block}\n"
        "[USER]\nAnaliza la siguiente transcripción y responde SOLO JSON válido:\n\n"
        f"{text}"
    )

    # 3) Llamar a Ollama (JSON mode)
    resp = ollama.generate(
        model=args.model,
        prompt=prompt,
        stream=False,
        format="json",  # fuerza salida JSON (si el modelo lo soporta)
        options={
            "num_ctx": args.num_ctx,
            "temperature": args.temperature
        }
    )

    raw = resp.get("response", "")
    try:
        data = extract_json(raw)
    except Exception:
        data = extract_json(raw)  # reintento simple

    clean = coerce_schema(data)

    # 4) Guardar JSON y TXT
    out_json = pathlib.Path(args.out_json)
    out_txt = pathlib.Path(args.out_txt)

    out_json.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text(render_text(clean), encoding="utf-8")

    print(f"[OK] JSON -> {out_json.resolve()}")
    print(f"[OK] TXT  -> {out_txt.resolve()}")

if __name__ == "__main__":
    main()
