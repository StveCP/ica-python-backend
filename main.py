import os
import requests
import time
from collections import Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from scipy.stats import entropy
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from numpy import dot
from numpy.linalg import norm

# Cargar variables de entorno
load_dotenv()

# Descargar paquetes de tokenización para NLP (solo lo ligero)
nltk.download('punkt')
nltk.download('punkt_tab')

# Inicializar la API
app = FastAPI(title="ICA Cognitive Metrics API")

# Credenciales
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("⚠️ Faltan credenciales de Supabase en .env")

class SessionData(BaseModel):
    session_id: str
    phase_1_text: str
    phase_5_text: str

def calculate_shannon_entropy(text: str) -> float:
    tokens = word_tokenize(text.lower())
    if not tokens: return 0.0
    counts = list(Counter(tokens).values())
    return float(entropy(counts, base=2))

def calculate_lexical_diversity(text: str) -> float:
    tokens = word_tokenize(text.lower())
    if not tokens: return 0.0
    return float(len(set(tokens)) / len(tokens))

# Función que llama a la API de Hugging Face en lugar de procesar localmente
def get_hf_embeddings(texts: list[str]) -> list:
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Intenta hacer la petición (Hugging Face a veces tarda unos segundos en "despertar" el modelo)
    for _ in range(3):
        response = requests.post(api_url, headers=headers, json={"inputs": texts})
        if response.status_code == 200:
            return response.json()
        elif "estimated_time" in response.text:
            time.sleep(response.json().get("estimated_time", 2.0))
        else:
            raise Exception(f"Error de HF API: {response.text}")
            
    raise Exception("Timeout esperando a Hugging Face")

@app.post("/api/analyze")
async def analyze_metrics(data: SessionData):
    try:
        # 1. Calcular Entropía y Diversidad Léxica (Local en Render, muy ligero)
        entropy_val = calculate_shannon_entropy(data.phase_5_text)
        lex_div = calculate_lexical_diversity(data.phase_5_text)

        # 2. Calcular Distancia Semántica usando los super-servidores de Hugging Face
        embeddings = get_hf_embeddings([data.phase_1_text, data.phase_5_text])
        emb1, emb5 = embeddings[0], embeddings[1]
        
        cos_sim = dot(emb1, emb5) / (norm(emb1) * norm(emb5))
        semantic_shift = float(1.0 - cos_sim)

        # 3. Construir y guardar resultados inmutables en Supabase
        metric_data = {
            "session_id": data.session_id,
            "prediction_error": entropy_val, 
            "integration_density": lex_div,
            "semantic_shift": semantic_shift,
            "quality_flag_short": len(data.phase_5_text.split()) < 10
        }
        
        supabase.table("ica_derived_metrics").insert(metric_data).execute()

        return {"success": True, "metrics": metric_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))