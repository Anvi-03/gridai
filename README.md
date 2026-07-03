# ⚡ GridPulse AI — Smart Grid Telemetry & GenAI Copilot

GridPulse AI is a next-generation, high-performance IoT Smart Grid predictive telemetry platform. It is designed to ingest high-frequency electrical measurements from smart meters, perform dual-layer machine learning anomaly detection and load forecasting, calculate grid-edge financial impact, and provide an interactive LLM-powered Copilot for grid operators.

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                    React Dashboard (UI)                  │
                  └─────────────────────────────▲────────────────────────────┘
                                                │
                                                ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │                 FastAPI Gateway (Uvicorn)                │
                  └──────┬──────────────────────▲──────────────────────▲─────┘
                         │                      │                      │
                  (Batch │ Telemetry     (Query │ DB contexts          │ (Copilot
                  Ingest)│ Flows)       Averages)                      │ Queries)
                         ▼                      │                      ▼
  ┌──────────────┐ ┌─────┴────────┐ ┌───────────┴──────────┐ ┌─────────┴─────────┐
  │ Smart Meters │ │ PostgreSQL   │ │ ML / Forecast        │ │ GenAI Copilot     │
  │ (Edge Z-Score│ │ Database     │ │ Background Services  │ │ (Google Gemini    │
  │ Pre-Screen)  │ │ (Time-series)│ │ (Ridge / IsoForest)  │ │   RAG-Lite Engine)│
  └──────────────┘ └──────────────┘ └──────────────────────┘ └───────────────────┘
```

---

## 🏗️ Architectural Layout & Data Flow

GridPulse AI implements an **Edge-to-Cloud** pipeline with asynchronous background workers to guarantee sub-millisecond API response latencies even under massive simulator workloads.

```mermaid
graph TD
    subgraph Edge Meters
        M1[Smart Meter 1] --> |Raw V, I, PF| E1[Local Filter: Rolling Z-Score]
        M2[Smart Meter 2] --> |Raw V, I, PF| E2[Local Filter: Rolling Z-Score]
    end

    subgraph Cloud Backend (FastAPI)
        E1 -.-> |HTTP POST Telemetry Batch| API[FastAPI Telemetry API]
        E2 -.-> |HTTP POST Telemetry Batch| API
        
        API --> |1. Quick Ingest & Commit| DB[(PostgreSQL Database)]
        
        API --> |2. Async Task Spawn| BG[Analytics Background Service]
        BG --> |Retrieve History| DB
        
        subgraph ML Pipeline
            BG --> |Anomaly Detection| AD[Dual Layer: Guard-Rails + Isolation Forest]
            BG --> |Load Forecasting| LF[Ridge Regression Forecaster]
            BG --> |Economic Assessment| FE[Financial Engine]
        end
        
        AD & LF & FE --> |Bulk Batch Update| DB
        
        COPE[GenAI Copilot Engine] --> |RAG Context Retrieval| DB
    end

    subgraph Client Application (React + Vite)
        UI[Digital Twin Dashboard] <--> |Fetch Stats & Telemetry| API
        UI <--> |Natural Language Queries| COPE
    end
    
    COPE <--> |google-genai SDK| GEMINI[(Google Gemini API)]
```

### Key Components

1. **FastAPI Ingestion Gateway (`main.py`, [api/v1](file:///d:/gridAI/gridai/api/v1))**
   - High-throughput asynchronous batch endpoint (`POST /api/v1/telemetry`) accepting meter payloads.
   - Saves records instantly to PostgreSQL and spawns a background analysis worker (`asyncio.create_task`) before returning an HTTP 201 response. This eliminates ML processing latency on the ingestion loop.

2. **Edge Local Pre-Screening Filter ([edge/local_filter.py](file:///d:/gridAI/gridai/edge/local_filter.py))**
   - A lightweight, dependency-free utility running standard library structures designed to mimic low-power microcontrollers (e.g., ARM Cortex-M).
   - Computes a rolling Z-score online via **Welford's algorithm** in $O(1)$ time complexity and flags abnormal values (`edge_flagged: true`).
   - Flagged telemetry bypasses cloud-based Isolation Forest scans entirely, reducing server-side GPU/CPU overhead.

3. **Dual-Layer ML Anomaly Engine ([ml/anomaly_detector.py](file:///d:/gridAI/gridai/ml/anomaly_detector.py))**
   - **Layer 1 (Deterministic Guard-rails):** Physical thresholds (e.g., severe voltage drop or theft signature of voltage sag + current spike).
   - **Layer 2 (Isolation Forest):** Scikit-Learn unsupervised Isolation Forest trained on normal grid operating conditions. Flags complex outliers.

4. **Detrended Ridge Regression Forecaster ([ml/load_forecaster.py](file:///d:/gridAI/gridai/ml/load_forecaster.py))**
   - Learns residuals against a 24-hour rolling mean (residual = actual - rolling_mean) to avoid non-stationary time-series drift.
   - Fits regularized Ridge regression against multi-lag feature vectors (lags, variance, local trends) to predict 24-hour grid demand profiles.
   - Robust fallbacks: reverts to Exponential Moving Average (EMA) for meters with sparse history.

5. **Financial & Outage Risk Engine ([services/financial_engine.py](file:///d:/gridAI/gridai/services/financial_engine.py))**
   - Converts detected power anomalies into estimated financial revenue loss (INR) using power deficits and local commercial utility tariffs.
   - Evaluates a transformer stress risk score (0 to 100) based on load margins relative to substation capacities.

6. **GenAI Grid Copilot RAG ([services/copilot_engine.py](file:///d:/gridAI/gridai/services/copilot_engine.py))**
   - Grounded RAG-Lite system. Automatically compiles real-time grid metrics, active anomalies, cash bleeding rates, and risk zones into a structured snapshot.
   - Feeds this context to the Google Gemini API alongside user operators' natural language queries to diagnose issues and suggest grid stabilization recommendations.

7. **Vite React Frontend Dashboard ([frontend](file:///d:/gridAI/gridai/frontend))**
   - Premium interface including a **Digital Twin Grid Topology Map** (visualizing meters, power flows, and alert levels), active diagnostic charts, simulation scenario controls, and an AI Copilot drawer.

---

## 📁 Repository Structure

```
gridai/
├── api/                    # FastAPI routing endpoints
│   └── v1/
│       ├── copilot.py      # Copilot chat endpoints
│       ├── forecasting.py  # Grid forecasting endpoints
│       └── simulation.py   # Simulation trigger endpoints
├── edge/                   # Smart Meter edge logic
│   └── local_filter.py     # Rolling Z-score anomaly detector
├── ml/                     # Machine Learning engine
│   ├── anomaly_detector.py # Isolation Forest + Guardrails
│   └── load_forecaster.py  # Ridge regression & LSTM load forecasting
├── services/               # Core business services
│   ├── analytics.py        # Async telemetry processing loop
│   ├── context_retriever.py# DB context compiler for Gemini RAG
│   ├── copilot_engine.py   # Gemini API client interface
│   ├── financial_engine.py # Revenue loss & transformer risk calculator
│   └── forecasting_service.py # Fleet forecast scheduling
├── frontend/               # Vite + React + TypeScript App
│   ├── src/
│   │   ├── components/     # Digital Twin Map, Charts, Copilot UI
│   │   ├── lib/api.ts      # Fetch client with automated offline fallbacks
│   │   └── App.tsx         # Dashboard main page layout
├── database.py             # SQLAlchemy Async Engine & connection pool
├── main.py                 # FastAPI Application Entrypoint & Lifespan
├── simulator.py            # Multithreaded Smart Meter simulator
├── check_pipeline.py       # Comprehensive diagnostic suite & load stress rig
└── requirements.txt        # Python dependency manifest
```

---

## ⚡ Quick Start Guide (Run in 3 Minutes)

Follow these steps to set up and run the entire ecosystem locally.

### 📋 Prerequisites
- **Python 3.10+** (Ensure it is in your PATH)
- **Node.js v18+** & **npm**
- **PostgreSQL** running locally
- **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

---

### Step 1: Spin up the Python Backend

1. **Create and activate a virtual environment** in the project root:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Copy `.env.example` to `.env` and fill in your details:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and insert your PostgreSQL connection string and Gemini API Key:
   ```env
   DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/gridpulse
   GEMINI_API_KEY=AIzaSy...your_real_key...
   ```

4. **Launch the FastAPI Server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   > [!NOTE]
   > The database tables (`telemetry_readings` and `forecast_snapshots`) are automatically verified and created on startup by the application lifespan hooks. No manual DDL executions are required in development.

---

### Step 2: Run the Smart Meter Simulator
The simulator launches `N` concurrent asynchronous virtual meters transmitting live grid measurements (voltage, current, and power factor) containing random anomalies and edge-pre-screened flags.

1. Open a new terminal window, activate the virtual environment, and run:
   ```bash
   python simulator.py
   ```
   You will see an ongoing throughput, success rate, and edge-flag stats output updated every 5 seconds.

---

### Step 3: Run the Frontend App

1. Open a new terminal window, navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. **Install node dependencies**:
   ```bash
   npm install
   ```

3. **Run the Vite development server**:
   ```bash
   npm run dev
   ```
   Open your browser to the URL printed in the terminal (typically `http://localhost:5173`).

---

## 🧪 Testing & Diagnostics

GridPulse AI ships with a suite of automated verification and load-stress diagnostics to assert compliance across components.

### Standalone Health Diagnostic Rig
Run the custom diagnostic command in your terminal to execute connection tests, validation constraints checks, data ingestion verification, and a live uvicorn-to-simulator concurrent stress test:
```bash
python check_pipeline.py
```

### PyTest Suite
Execute specialized tests targeting backend pipelines:
```bash
# Verify standard Isolation Forest and ML ingestion pipelines
pytest test_ml_pipeline.py

# Verify schema compatibility and edge-enriched pre-screen routing
pytest test_edge_pipeline.py

# Verify predictive forecasting sweeps and filters
pytest test_feature5_smoke.py
pytest test_feature5_e2e.py
```

---

## 💡 Tech Stack Overview

- **Backend Architecture:** FastAPI (Python), Uvicorn ASGI Server, SQLAlchemy ORM (Async I/O via `asyncpg`), Pydantic Settings.
- **Analytics & Machine Learning:** Scikit-Learn (Isolation Forest), NumPy, PyTorch (Optional LSTM), Welford's Incremental Statistics.
- **GenAI Copilot:** Google Gemini 2.5 Flash, `google-genai` Python SDK.
- **Frontend Dashboard:** React, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts (Charts).