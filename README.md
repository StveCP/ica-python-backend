# 🧠 ICA Python Backend: Deterministic NLP & Cognitive Smith Chart Engine

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face_API-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

This repository houses the stateless mathematical evaluation microservice for the **Adaptive Understanding Index (ICA)** ([www.ica-observatory.com](https://www.ica-observatory.com)). It is architecturally isolated from the conversational LLM mediator to eliminate the *"LLM-as-a-judge"* fallacy and prevent evaluative sycophancy.

---

## 🔬 Core Responsibilities (Two-Pass Architecture)

### 1. First Pass (`POST /api/analyze`)
Triggered upon the completion of Phase 5 in each domain, this endpoint receives the participant's baseline observation (`phase_1_text`) and post-contradiction synthesis (`phase_5_text`) to compute:
- **Shannon Entropy ($H$):** Information-theoretic uncertainty calculated via `scipy.stats.entropy` (base 2).
- **Lexical Diversity ($L$):** Vocabulary richness ratio via `nltk` tokenization.
- **Semantic Shift ($D_s$):** Cosine distance ($1 - \cos(\vec{v}_1, \vec{v}_5)$) computed via **Hugging Face Inference API** (`sentence-transformers/paraphrase-MiniLM-L6-v2`) with local vector fallback for zero-downtime reliability.
- **Epistemic Markers & New Relations:** Counts metacognitive uncertainty markers and newly integrated conceptual tokens.

### 2. Second Pass (`POST /api/smith-synthesis`) — "The Final Double-Blind"
Consolidates the participant's 5 domain traces (`d1..d5`) and longitudinal history (`85+ day` quarterly windows) from `ica_telemetry_events` and `ica_derived_metrics`:
- **Automatic Domain Backfill:** Ensures any missing domain metrics are deterministically computed from raw telemetry and persisted to PostgreSQL.
- **Conformal Smith Chart Mapping:** Projects $(H, D_s, L, \tau)$ onto the complex unit disk:
  - $R_c = \frac{3.5}{\max(H, 0) + 0.1} + 0.08\tau$
  - $X_c = 2.0(0.35 - D_s) + 0.25(0.65 - L)$
  - $\Gamma = \frac{Z_c - 1}{Z_c + 1} = u + jv$
  - $\text{VSWR}_c = \frac{1 + |\Gamma|}{1 - |\Gamma|}$
- **180° Ideological Counterpart ($Y_c = 1/Z_c$):** Computes the opposite admittance vector $(-\Gamma_u, -\Gamma_v)$ and classifies both the participant's hidden cognitive bias (`bias_code`) and their analytical shadow (`shadow_code`).

---

## 🚀 Endpoints

- `GET /` — Health check and service version status.
- `POST /api/analyze` — First-pass domain NLP evaluation and `ica_derived_metrics` persistence.
- `POST /api/smith-synthesis` — Full 5-domain Smith topology, 180° counterpart projection, and 90-day longitudinal cycle aggregation.