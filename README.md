<table>
  <tr>
    <td width="180" align="center">
      <img src="src/frontend/src/assets/logo.png" alt="SafeDx logo" width="96" />
    </td>
    <td>
      <h1>SafeDx</h1>
      <p>Uncertainty-Aware Diagnostic Support Software for Medical Imaging</p>
    </td>
  </tr>
</table>

SafeDx combines software engineering with state-of-the-art conformal prediction, specifically **Conformal Risk Control (CRC)**. AI models are often overconfident, and in a field as sensitive as medicine there is no room for unchecked mistakes. SafeDx provides a full calibration pipeline through a web interface, allowing developers to publish statistically guaranteed models that clinicians can use with confidence.

---

## Who is this for?

| Role | What they do |
|---|---|
| **Developers / Researchers** | Upload trained models, run calibration, validate results, publish to the community |
| **Clinicians** | Browse published calibrated models, create patient cases, run uncertainty-aware diagnoses |

---

## How it works

```
Developer uploads model + calibration dataset
        ↓
Backend stores artifacts in Azure Blob Storage
        ↓
Calibration job computes softmax scores → λ̂ threshold (CRC)
        ↓
Validation checks FNR, set size, and calibration quality
        ↓
Model published to the platform
        ↓
Clinician selects model → uploads patient image → gets prediction set + uncertainty level
```

---

## Key Features

- **Calibration pipeline** — computes λ̂ thresholds using Conformal Risk Control
- **Validation graphs** — FNR, average set size, and calibration quality visualised
- **Model publishing** — visibility controls and community sharing
- **Clinician inference** — prediction sets with uncertainty flags, not just single labels
- **Patient management** — patient profiles and full case history
- **Domain-agnostic** — works across chest X-ray, retinal imaging, and beyond
- **Azure Blob Storage** — all model and dataset artifacts stored securely
- **Supabase (PostgreSQL)** — robust relational database backing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, SQLAlchemy |
| Database | Supabase (PostgreSQL) |
| Storage | Azure Blob Storage |
| ML | PyTorch, TorchScript |
| Calibration | Conformal Prediction (CRC) |

---

## Project Structure

```
src/
├── frontend/               # React/Vite frontend
└── backend/
    ├── routes/             # API route handlers
    ├── services/           # Core business logic (calibration, inference, storage)
    ├── enums/              # Shared backend enums
    ├── tests/              # Backend test suite
    └── ODIR-5K/            # Retinal imaging training pipeline
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- Supabase project (PostgreSQL)
- Azure Storage account

### Backend

```bash
cd src/backend
pip install -r requirements.txt
uvicorn api:app --reload
```

### Frontend

```bash
cd src/frontend
npm install
npm run dev
```

### Environment Variables

Create a `.env` file in `src/backend/` with:

```
DATABASE_URL=postgresql://...
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_CONTAINER_NAME=...
```

---

## API Docs

Once the backend is running, full interactive API documentation is available at:

```
http://127.0.0.1:8000/docs
```

---

## Future Work

- Cloud deployment to accelerate calibration and inference jobs
- Per-organisation access control, allowing institutions to adopt SafeDx as a SaaS product where clinicians only access their organisation's published models
- Extended risk control options, including False Positive Rate (FPR) control

---

## Contact

For questions or collaboration, reach out at [Abdullahmmmaghrabi@gmail.com](mailto:Abdullahmmmaghrabi@gmail.com)

---
