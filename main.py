import os
import re
import math
import time
import requests
from collections import Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from scipy.stats import entropy
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from numpy import dot, mean, array
from numpy.linalg import norm

load_dotenv()

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"⚠️ Aviso NLTK: {e}")

app = FastAPI(title="ICA Cognitive Metrics & Smith Topology API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"⚠️ Error inicializando Supabase: {e}")


class SessionData(BaseModel):
    session_id: str
    phase_1_text: str = ""
    phase_5_text: str = ""


class SmithSynthesisRequest(BaseModel):
    participant_email: str
    usuario_id: str | None = None
    language: str = "es"


DOMAIN_MAP = {
    "d1": "Educación",
    "d2": "Selección Profesional",
    "d3": "Salud Mental",
    "d4": "IA y Espiritualidad",
    "d5": "Política y Sociedad",
}


def safe_tokenize(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    try:
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum()]
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())


def extract_key_concepts(text: str, top_n: int = 3) -> list[str]:
    stopwords = {
        "para", "como", "pero", "mas", "más", "este", "esta", "estos", "estas",
        "todo", "toda", "todos", "todas", "otro", "otra", "entre", "sobre", "desde",
        "hasta", "donde", "cuando", "porque", "aunque", "parece", "puede", "pueden",
        "tiene", "tienen", "hacer", "hace", "vemos", "observa", "plano", "imagen",
        "escena", "personas", "persona", "algo", "mismo", "misma", "with", "that",
        "this", "from", "have", "what", "when", "where", "which", "their", "there",
        "about", "would", "could", "should", "into", "while", "Dans", "avec", "pour"
    }
    tokens = [w for w in safe_tokenize(text) if len(w) > 4 and w not in stopwords]
    if not tokens:
        return []
    return [w for w, _ in Counter(tokens).most_common(top_n)]


def calculate_shannon_entropy(text: str) -> float:
    tokens = safe_tokenize(text)
    if not tokens:
        return 0.0
    counts = list(Counter(tokens).values())
    val = float(entropy(counts, base=2))
    return round(val, 4) if not math.isnan(val) else 0.0


def calculate_lexical_diversity(text: str) -> float:
    tokens = safe_tokenize(text)
    if not tokens:
        return 0.0
    return round(float(len(set(tokens)) / len(tokens)), 4)


def count_epistemic_markers(text: str) -> int:
    markers = {
        "quizás", "quizas", "tal", "vez", "posiblemente", "depende", "incertidumbre",
        "duda", "parece", "podría", "podria", "asumí", "asumi", "cambié", "cambie",
        "antes", "ahora", "contraste", "anomalía", "anomalia", "sin", "embargo",
        "perhaps", "maybe", "possibly", "depends", "uncertainty", "might", "could",
        "although", "however", "assumed", "shifted", "tension", "tensión"
    }
    tokens = safe_tokenize(text)
    return sum(1 for t in tokens if t in markers)


def count_new_relations(text1: str, text5: str) -> int:
    t1 = set(w for w in safe_tokenize(text1) if len(w) > 4)
    t5 = set(w for w in safe_tokenize(text5) if len(w) > 4)
    return len(t5 - t1)


def local_lexical_cosine_shift(text1: str, text5: str) -> float:
    tokens1 = safe_tokenize(text1)
    tokens5 = safe_tokenize(text5)
    if not tokens1 or not tokens5:
        return 0.35
    vocab = list(set(tokens1 + tokens5))
    c1, c5 = Counter(tokens1), Counter(tokens5)
    v1 = array([c1.get(w, 0) for w in vocab], dtype=float)
    v5 = array([c5.get(w, 0) for w in vocab], dtype=float)
    denom = norm(v1) * norm(v5)
    if denom == 0:
        return 0.35
    cos_sim = float(dot(v1, v5) / denom)
    return round(max(0.0, min(1.0, 1.0 - cos_sim)), 4)


def ensure_1d_vector(vec) -> list[float]:
    arr = array(vec, dtype=float)
    if arr.ndim == 2:
        arr = mean(arr, axis=0)
    return arr.flatten().tolist()


def calculate_semantic_shift(text1: str, text5: str) -> float:
    if not text1.strip() or not text5.strip():
        return 0.35
    if HF_TOKEN:
        api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-MiniLM-L6-v2"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        try:
            for _ in range(2):
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": [text1, text5], "options": {"wait_for_model": True}},
                    timeout=8
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) >= 2:
                        emb1 = ensure_1d_vector(data[0])
                        emb5 = ensure_1d_vector(data[1])
                        denom = norm(emb1) * norm(emb5)
                        if denom > 0:
                            cos_sim = float(dot(emb1, emb5) / denom)
                            return round(max(0.0, min(1.0, 1.0 - cos_sim)), 4)
                elif "estimated_time" in response.text:
                    time.sleep(min(float(response.json().get("estimated_time", 2.0)), 3.0))
                else:
                    break
        except Exception as e:
            print(f"⚠️ Aviso HF API (usando cálculo vectorial local): {e}")
    return local_lexical_cosine_shift(text1, text5)


def calculate_smith_chart(entropy_val: float, semantic_shift: float, lex_div: float = 0.65) -> dict:
    """
    Transformación de Möbius para proyectar la Entropía de Shannon (H) y el
    Desplazamiento Semántico (Ds) de Hugging Face al Cuadro de Smith Cognitivo.
    """
    H_0 = 3.5  # Punto de resonancia epistémica (bits)
    D_0 = 0.35 # Umbral de actualización equilibrada

    r_c = H_0 / (max(entropy_val, 0.0) + 0.1)
    x_c = 2.0 * (D_0 - semantic_shift) + ((0.65 - lex_div) * 0.25)

    denom = ((r_c + 1.0) ** 2) + (x_c ** 2)
    gamma_u = ((r_c ** 2) + (x_c ** 2) - 1.0) / denom
    gamma_v = (2.0 * x_c) / denom
    gamma_mag = min(0.99, math.sqrt((gamma_u ** 2) + (gamma_v ** 2)))
    vswr_c = (1.0 + gamma_mag) / (1.0 - gamma_mag)

    denom_adm = (r_c ** 2) + (x_c ** 2)
    g_c = (r_c / denom_adm) if denom_adm > 0 else 1.0
    b_c = (-x_c / denom_adm) if denom_adm > 0 else 0.0

    if gamma_mag <= 0.32:
        zone_type = "balanced"
    elif x_c >= 0:
        zone_type = "inductive"
    else:
        zone_type = "capacitive"

    counterpart_zone = "balanced" if zone_type == "balanced" else ("capacitive" if zone_type == "inductive" else "inductive")

    return {
        "entropy_shannon": round(entropy_val, 4),
        "semantic_shift": round(semantic_shift, 4),
        "lexical_diversity": round(lex_div, 4),
        "impedance_r": round(r_c, 4),
        "reactance_x": round(x_c, 4),
        "gamma_u": round(gamma_u, 4),
        "gamma_v": round(gamma_v, 4),
        "reflection_magnitude": round(gamma_mag, 4),
        "assimilation_index": round(1.0 - gamma_mag, 4),
        "vswr_cognitive": round(vswr_c, 4),
        "zone_type": zone_type,
        "counterpart_zone": counterpart_zone,
        "ideological_counterpart": {
            "admittance_g": round(g_c, 4),
            "susceptance_b": round(b_c, 4),
            "counterpart_u": round(-gamma_u, 4),
            "counterpart_v": round(-gamma_v, 4),
        }
    }


@app.get("/")
async def health_check():
    return {"status": "online", "service": "ICA Cognitive Metrics API v3.0 (Two-Pass Smith Synthesis)"}


@app.post("/api/analyze")
async def analyze_metrics(data: SessionData):
    """
    PRIMER RECORRIDO EN PYTHON:
    Calcula Entropía de Shannon, Diversidad Léxica y Distancia Semántica (Hugging Face),
    guarda en `ica_derived_metrics` y retorna la primera transformación de Smith.
    """
    try:
        p1 = data.phase_1_text or ""
        p5 = data.phase_5_text or ""

        entropy_val = calculate_shannon_entropy(p5)
        lex_div = calculate_lexical_diversity(p5)
        semantic_shift = calculate_semantic_shift(p1, p5)
        epistemic_cnt = count_epistemic_markers(p5)
        new_rel_cnt = count_new_relations(p1, p5)

        smith_data = calculate_smith_chart(entropy_val, semantic_shift, lex_div)

        metric_data = {
            "session_id": data.session_id,
            "transfer_overlap": smith_data["assimilation_index"],
            "integration_density": lex_div,
            "prediction_error": entropy_val,
            "semantic_shift": semantic_shift,
            "epistemic_markers": epistemic_cnt,
            "new_relations": new_rel_cnt,
            "quality_flag_short": len(p5.split()) < 10
        }

        if supabase:
            try:
                supabase.table("ica_derived_metrics").insert(metric_data).execute()
            except Exception as db_err:
                print(f"⚠️ Error guardando en Supabase: {db_err}")

        return {
            "success": True,
            "metrics": metric_data,
            "smith_chart": smith_data
        }
    except Exception as e:
        print(f"⚠️ Error recuperado en /api/analyze: {e}")
        fallback_smith = calculate_smith_chart(3.2, 0.35, 0.65)
        return {
            "success": True,
            "metrics": {
                "session_id": data.session_id,
                "transfer_overlap": fallback_smith["assimilation_index"],
                "integration_density": 0.65,
                "prediction_error": 3.2,
                "semantic_shift": 0.35,
                "epistemic_markers": 1,
                "new_relations": 3,
                "quality_flag_short": False
            },
            "smith_chart": fallback_smith,
            "recovered": True
        }


@app.post("/api/smith-synthesis")
async def smith_second_pass(req: SmithSynthesisRequest):
    """
    SEGUNDO RECORRIDO EN PYTHON (EL ÚLTIMO DOBLE CIEGO):
    Toma las métricas ya calculadas en `ica_derived_metrics` y los textos de `ica_telemetry_events`
    para un usuario específico, calcula la topología de Smith por dominio (D1..D5),
    el historial longitudinal de ciclos de 90 días (S1..Sn) + Vector Global,
    y genera los parámetros personalizados para cada participante.
    """
    try:
        email = req.participant_email or "evaluable@ica-research.org"
        derived_rows = []
        telemetry_rows = []

        if supabase:
            res_d = (
                supabase.table("ica_derived_metrics")
                .select("*")
                .ilike("session_id", f"{email}%")
                .order("calculated_at", desc=False)
                .execute()
            )
            derived_rows = res_d.data or []

            res_t = (
                supabase.table("ica_telemetry_events")
                .select("domain, phase_id, response_text, response_latency_ms, created_at")
                .eq("participant_id", email)
                .order("created_at", desc=False)
                .execute()
            )
            telemetry_rows = res_t.data or []

        # Extraer conceptos reales usados por este participante
        all_user_text = " ".join(r.get("response_text", "") for r in telemetry_rows if r.get("phase_id") in ("fase_4", "fase_5"))
        if not all_user_text:
            all_user_text = " ".join(r.get("response_text", "") for r in telemetry_rows)
        user_concepts = extract_key_concepts(all_user_text, top_n=4)

        # Mapear los 5 dominios del participante desde `ica_derived_metrics`
        domain_vectors = {}
        for row in derived_rows:
            sid = row.get("session_id", "")
            for d_key, d_name in DOMAIN_MAP.items():
                if f"_{d_key}_" in sid:
                    h_val = float(row.get("prediction_error") or 3.2)
                    s_val = float(row.get("semantic_shift") or 0.35)
                    l_val = float(row.get("integration_density") or 0.65)
                    domain_vectors[d_key] = {
                        "domain_key": d_key,
                        "domain_name": d_name,
                        "epistemic_markers": int(row.get("epistemic_markers") or 1),
                        "new_relations": int(row.get("new_relations") or 3),
                        **calculate_smith_chart(h_val, s_val, l_val),
                    }

        # Si algún dominio no estaba en derived_metrics pero sí en telemetry_events, calcularlo al vuelo
        for idx, (d_key, d_name) in enumerate(DOMAIN_MAP.items(), start=1):
            if d_key not in domain_vectors:
                dom_msgs = [r.get("response_text", "") for r in telemetry_rows if str(idx) in str(r.get("domain", "")) or d_name.lower()[:5] in str(r.get("domain", "")).lower()]
                if dom_msgs:
                    p1_txt = dom_msgs[0]
                    p5_txt = dom_msgs[-1]
                    h_val = calculate_shannon_entropy(p5_txt)
                    l_val = calculate_lexical_diversity(p5_txt)
                    s_val = local_lexical_cosine_shift(p1_txt, p5_txt)
                    domain_vectors[d_key] = {
                        "domain_key": d_key,
                        "domain_name": d_name,
                        "epistemic_markers": count_epistemic_markers(p5_txt),
                        "new_relations": count_new_relations(p1_txt, p5_txt),
                        **calculate_smith_chart(h_val, s_val, l_val),
                    }

        dom_list = list(domain_vectors.values())
        if not dom_list and derived_rows:
            last = derived_rows[-1]
            base_s = calculate_smith_chart(
                float(last.get("prediction_error") or 3.2),
                float(last.get("semantic_shift") or 0.35),
                float(last.get("integration_density") or 0.65),
            )
            dom_list = [{"domain_key": "d1", "domain_name": "Educación", **base_s}]

        # Calcular puntos 1, 2 y 3 desde las métricas reales de cada dominio
        reevaluaciones = sum(1 for d in dom_list if d["semantic_shift"] >= 0.25 and d["entropy_shannon"] >= 2.0)
        mantenimientos = len(dom_list) - reevaluaciones
        if len(dom_list) > 0 and len(dom_list) != 5:
            reevaluaciones = min(5, round((reevaluaciones / len(dom_list)) * 5))
            mantenimientos = 5 - reevaluaciones
        elif len(dom_list) == 0:
            reevaluaciones, mantenimientos = 3, 2

        sobrecorrecciones = sum(1 for d in dom_list if d["semantic_shift"] > 0.78 and d["entropy_shannon"] < 1.85)

        # Identificar el dominio de mayor adaptación y el de mayor inercia estructural
        sorted_by_assim = sorted(dom_list, key=lambda x: x["assimilation_index"], reverse=True)
        highest_assim_dom = sorted_by_assim[0] if sorted_by_assim else None
        lowest_assim_dom = sorted_by_assim[-1] if sorted_by_assim else None

        # Vector global de la sesión actual
        avg_h = float(mean([d["entropy_shannon"] for d in dom_list])) if dom_list else 3.35
        avg_s = float(mean([d["semantic_shift"] for d in dom_list])) if dom_list else 0.38
        avg_l = float(mean([d["lexical_diversity"] for d in dom_list])) if dom_list else 0.68
        session_smith = calculate_smith_chart(avg_h, avg_s, avg_l)

        # Agrupar por fecha de simulación para el historial longitudinal de 90 días (S1, S2, ..., Sn)
        by_date = {}
        for row in derived_rows:
            date_key = str(row.get("calculated_at", ""))[:10] or "ciclo_1"
            by_date.setdefault(date_key, []).append(row)

        longitudinal_cycles = []
        for idx, (dt, rows_in_cycle) in enumerate(sorted(by_date.items()), start=1):
            c_h = float(mean([float(r.get("prediction_error") or 3.2) for r in rows_in_cycle]))
            c_s = float(mean([float(r.get("semantic_shift") or 0.35) for r in rows_in_cycle]))
            c_l = float(mean([float(r.get("integration_density") or 0.65) for r in rows_in_cycle]))
            longitudinal_cycles.append({
                "cycle_num": idx,
                "date": dt,
                "day_label": (idx - 1) * 90,
                **calculate_smith_chart(c_h, c_s, c_l)
            })

        return {
            "success": True,
            "participant_email": email,
            "user_concepts": user_concepts,
            "points_1_2_3": {
                "reevaluaciones": reevaluaciones,
                "mantenimientos": mantenimientos,
                "sobrecorrecciones": sobrecorrecciones,
                "highest_adaptation_domain": highest_assim_dom,
                "highest_inertia_domain": lowest_assim_dom,
            },
            "domain_smith_nodes": dom_list,
            "session_smith": session_smith,
            "longitudinal_cycles": longitudinal_cycles,
        }
    except Exception as e:
        print(f"⚠️ Error recuperado en /api/smith-synthesis: {e}")
        return {"success": False, "error": str(e)}