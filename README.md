# 👁️ Adaptive Understanding Index (ICA): A Multimodal HCI Observatory

![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Anthropic](https://img.shields.io/badge/Claude_3.5-1A1A1A?style=for-the-badge&logo=anthropic&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

## 📌 Abstract & Scientific Contribution

Historically, cognitive flexibility and ambiguity tolerance have been measured through static psychometric tools (e.g., self-reported Likert scales) that rely on user honesty and fail to capture the *temporally structured process* of meaning-making. Furthermore, contemporary attempts to use Large Language Models (LLMs) in qualitative research often suff from the "LLM-as-a-judge" fallacy, where models introduce semantic bias, sycophancy (flattery), and opaque evaluations.

The **Adaptive Understanding Index (ICA)** proposes a methodological rupture. It is a deployed, production-ready **Human-Computer Interaction (HCI) Observatory** designed to empirically measure *epistemic updating* under conditions of controlled uncertainty. 

Instead of asking users if they are flexible, the ICA subjects them to a generative visual perturbation (Dual-Frame visual dissonance), facilitates a strictly neutral 5-phase Socratic dialogue mediated by Claude 3.5 Sonnet, and extracts immutable telemetry. Crucially, **the architecture separates adaptive elicitation from downstream measurement**, pushing the mathematical evaluation (Shannon Entropy, Semantic Distance, and Lexical Diversity) to an isolated Python NLP pipeline.

## ⚙️ Distributed Systems Architecture

To ensure zero-cost scalability while maintaining institutional-grade data rigor, the ICA operates on a hybrid, serverless microservices architecture:

### 1. Frontend & User Experience (Edge)
* **Framework:** Next.js (React) + Tailwind CSS, deployed on Vercel.
* **Native Internationalization (i18n):** Dynamic routing supporting **38 native languages** without hardcoded text, ensuring cross-cultural research viability without compilation overhead.
* **State Preservation:** Leverages `sessionStorage` to protect API budgets and prevent data loss against accidental reloads during the cognitive simulation.

### 2. Generative Visual Stimulus (FLUX.1 via FAL.ai)
* **Dynamic Perturbation:** Generates photorealistic "Dual-Frame" scenes contrasting an *Institutional Norm* in the foreground with a *Social Anomaly* in the background across 5 fixed domains (Education, Work, Mental Health, AI/Religion, Politics). This ensures standardized cognitive dissonance rather than aesthetic noise.

### 3. Socratic Mediation Engine (Anthropic API)
* **Strict Elicitation Guardrails:** Claude 3.5 Sonnet operates exclusively as a neutral mediator. Through extensive System Prompt engineering and **Prompt Caching** (reducing token costs by ~90%), the LLM is restricted to a 5-phase protocol (Observation, Dual Focus, Contradiction, Cross-checking, Closure) with absolute prohibition on evaluating the user or introducing new vocabulary.

### 4. Immutable Telemetry & Auth (Supabase / PostgreSQL)
* **BaaS & Gated Access:** Passwordless authentication via Magic Links (Resend API) anchored to a strict 90-day longitudinal lock (`Delta 90`). 
* **Data Ethics (RLS):** Employs Row Level Security (RLS) across two heavily isolated tables: `ica_telemetry_events` (raw, immutable chat logs with latency/token tracking) and `ica_derived_metrics` (safe haven for Python processing).

### 5. Deterministic NLP Backend (Python + FastAPI)
* **Hybrid Processing Pipeline:** Hosted on Render, this isolated Python server retrieves raw text and interfaces via API with **Hugging Face**. This circumvents RAM bottlenecks for heavy models (e.g., `sentence-transformers` for cosine semantic distance), enabling high-end Data Science calculations (MATTR, Shannon Entropy) inside a resource-constrained, zero-cost environment.

## 🔬 Methodology: The Automated Double-Blind Elicitation

A critical bottleneck in contemporary Human-Computer Interaction (HCI) research is the reliance on "LLM-as-a-judge" architectures. Large Language Models inherently exhibit sycophancy (the tendency to echo a user's stance) and semantic bias, rendering them invalid as objective evaluators of human cognition. 

The ICA dismantles this flaw through an **Automated Double-Blind Architecture**. By strictly decoupling the *elicitation of data* from the *evaluation of data*, the system guarantees empirical rigor.

### The Neutral Socratic Mediator (LLM Guardrails)
The conversational engine (Claude 3.5 Sonnet) is stripped of all evaluative capabilities via rigorous System Prompt engineering and Prompt Caching. It acts exclusively as a blind elicitor, pushing the user through a standardized 5-phase temporal protocol:

1. **Observation:** Capturing the baseline perception of the Dual-Frame visual stimulus.
2. **Dual Focus:** Forcing the articulation of the foreground (Institutional Norm) versus the background (Social Anomaly).
3. **Controlled Contradiction:** Inducing cognitive dissonance by questioning the initial baseline.
4. **Cross-Checking:** Stress-testing the user's rationalization of the anomaly.
5. **Closure:** Harvesting the final consolidated mental model.

The LLM operates under absolute constraints: it cannot praise, diagnose, summarize, or introduce external vocabulary. It solely maps the user's internal semantic boundaries, generating raw, uncontaminated linguistic traces.

## 📊 Data Telemetry & Deterministic NLP Pipeline

Because the LLM is prohibited from scoring the interaction, the analytical heavy lifting is offloaded to a deterministic, stateless Python backend. This process is governed by a strict, two-tier database architecture in PostgreSQL (Supabase) secured by Row Level Security (RLS).

### 1. Immutable Event Sourcing (`ica_telemetry_events`)
Every keystroke, conversational turn, and system response is logged as an immutable event. The system captures:
* `response_text`: The uncontaminated linguistic footprint.
* `response_latency_ms`: Time-to-response, serving as a proxy for cognitive friction and processing load.
* `token_count` & `phase_id`: For normalizing data density across the 5 Socratic phases.

### 2. Isolated Mathematical Evaluation (`ica_derived_metrics`)
Once Phase 5 concludes, a webhook triggers the isolated Python backend (hosted on Render, interfacing with Hugging Face Inference APIs). This engine calculates candidate indicators of **Epistemic Updating** without altering the raw evidence:

* **Information-Theoretic Uncertainty (Shannon Entropy):** Calculated via `scipy.stats.entropy`. Instead of naïve word-counting, the system measures the distributional shift in lexical choices between Phase 1 (pre-dissonance) and Phase 4 (post-contradiction).
* **Vocabulary Richness & Rigidity (MATTR/MTLD):** Utilizing the `TAALED` (Tool for the Automatic Analysis of Lexical Diversity) framework. MATTR (Moving-Average Type-Token Ratio) isolates lexical diversity from text-length dependency, quantifying whether a user's descriptive model expanded or rigidly looped.
* **Semantic Shift (Cosine Distance):** Leveraging `sentence-transformers` (e.g., `all-mpnet-base-v2`) and `scikit-learn` to generate dense vector embeddings of the user's initial and final stances. The mathematical distance D_semantic = 1 - cos(e_F1, e_F4) provides an exploratory, quantitative proxy for how much a mental model shifted after encountering the visual anomaly.

## ⏳ Longitudinal Epistemic Tracking (Delta 90)

A fundamental limitation in contemporary cognitive evaluation is the reliance on cross-sectional (single-point) data capture, which only measures reactive states. True cognitive plasticity—the genuine updating of a mental model—can only be verified by observing the persistence or decay of that model over time. 

To achieve this, the ICA implements a strict **90-Day Longitudinal Protocol (Δ90) **:
* **Cryptographic Lockdown:** Upon completion of Phase 5, the user's session is cryptographically sealed. A dual-layer locking mechanism—enforced via Supabase server-side timestamps and localized `sessionStorage`—blocks further evaluations for exactly 90 days.
* **Frictionless Retention (Magic Links):** To guarantee high user retention for the critical second evaluation without compromising security, the system utilizes passwordless authentication via Resend API. 
* **Measuring Epistemic Shift:** The return session generates a comparative matrix (Mes 0 vs. Mes 3), allowing the Python backend to calculate the true longitudinal delta in semantic distance and lexical entropy.

## 🛡️ Data Ethics, Privacy & Anti-Gaming Mechanisms

To maintain the integrity of the empirical data and protect user privacy, the architecture strictly adheres to international data protection standards and ethical HCI guidelines:

* **Row Level Security (RLS):** The PostgreSQL database enforces strict RLS policies. The application can only `INSERT` telemetry anonymously; it is architecturally prohibited from executing `SELECT` queries across other participants' data, eliminating any risk of cross-contamination or unauthorized scraping.
* **Observer Effect Mitigation (The Dual-Architecture PDF):** If users know exactly how they are being scored, they inevitably attempt to "game" the system in subsequent sessions. The ICA solves this via a dual-state UI (`report-modal.tsx`). While the web interface is interactive, the final exported PDF report acts as a premium, tangible "trophy" of introspection for the user—ensuring engagement and return rates—but deliberately obfuscates the underlying mathematical "black box" (exact entropy/MATTR scores). This prevents users from altering their natural linguistic behavior in the Δ90 session.
* **Strict Non-Clinical Policy:** The system explicitly prohibits algorithmic diagnosis. The variables extracted are candidate indicators of *discursive updating*, not clinical psychometrics or HR screening metrics.

---

> *"Understanding is not the accumulation of knowledge, but the adaptive capacity to detect the failure of an internal model, tolerate the universal fall, and reorganize constraints to navigate an incomplete reality."* — ICA Foundation.

