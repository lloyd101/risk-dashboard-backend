"""
FastAPI application serving risk scores for the risk scoring dashboard.

This backend exposes a handful of REST endpoints to support the React
frontend:

* ``GET /api/applicants`` – returns all applicant records (synthetic for now)
  along with baseline risk scores computed at startup.
* ``GET /api/risk-scores`` – accepts a query parameter ``age_weight`` and
  returns the same records with risk scores recomputed using the
  provided weight on the age feature.
* ``GET /api/applicant/{applicant_id}`` – returns a single applicant
  record by index.
* ``POST /api/score-batch`` – accepts a CSV file with applicant
  features and returns their computed risk scores (simple demonstration).

The application currently reads from a local CSV (``risk_scores.csv``)
to populate data and train the model.  In a production setting you
would load data from a Postgres/PostGIS database (e.g. Supabase) and
possibly use a persisted model.  The ``DATABASE_URL`` environment
variable is accepted but unused in this minimal example.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import io
from typing import List, Optional

from .modeling import RiskModel

# Application instance
app = FastAPI(title="Risk Scoring API", version="0.1.0")

# Allow CORS for all origins (for development).  In production
# restrict this to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data from CSV on startup.  Use environment variable
# RISK_DATA_PATH or default to risk_scores.csv in repository root.
# Default data path is one directory up from backend (i.e. in risk_dashboard_webapp/risk_scores.csv)
DATA_PATH = os.getenv('RISK_DATA_PATH', os.path.join(os.path.dirname(__file__), '..', 'risk_scores.csv'))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

# Read CSV into DataFrame
df = pd.read_csv(DATA_PATH)

# Instantiate risk model (will train if target exists)
risk_model = RiskModel(df)


class ApplicantResponse(BaseModel):
    """Response model for an applicant with risk score."""
    id: int
    age: float
    tenure: float
    prior_claims: int
    credit_band: str
    asset_type: str
    asset_value: float
    lat: float
    lon: float
    geo_risk: float
    risk_score: float


def compute_responses(risks: List[float]) -> List[ApplicantResponse]:
    """Construct response objects from the DataFrame and risk array."""
    responses: List[ApplicantResponse] = []
    for idx, (row, score) in enumerate(zip(risk_model.df.itertuples(index=False), risks)):
        responses.append(
            ApplicantResponse(
                id=idx,
                age=row.age,
                tenure=row.tenure,
                prior_claims=row.prior_claims,
                credit_band=row.credit_band,
                asset_type=row.asset_type,
                asset_value=row.asset_value,
                lat=row.lat,
                lon=row.lon,
                geo_risk=row.geo_risk,
                risk_score=float(score),
            )
        )
    return responses


@app.get("/api/applicants", response_model=List[ApplicantResponse])
async def get_applicants():
    """Return all applicants with baseline risk scores."""
    if 'baseline_risk' in risk_model.df.columns:
        risks = risk_model.df['baseline_risk'].values
    else:
        # Compute baseline risk if not previously computed
        risks = risk_model.compute_risk(age_weight=1.0)
        risk_model.df['baseline_risk'] = risks
    return compute_responses(risks.tolist())


@app.get("/api/risk-scores", response_model=List[ApplicantResponse])
async def get_risk_scores(age_weight: float = 1.0):
    """Return applicants with risk scores recomputed using the given age weight."""
    risks = risk_model.compute_risk(age_weight)
    return compute_responses(risks.tolist())


@app.get("/api/applicant/{applicant_id}", response_model=ApplicantResponse)
async def get_single_applicant(applicant_id: int, age_weight: float = 1.0):
    """Return a single applicant by index with risk recomputed if desired."""
    if applicant_id < 0 or applicant_id >= len(risk_model.df):
        raise HTTPException(status_code=404, detail="Applicant not found")
    risks = risk_model.compute_risk(age_weight)
    row = risk_model.df.iloc[applicant_id]
    score = float(risks[applicant_id])
    return ApplicantResponse(
        id=applicant_id,
        age=row.age,
        tenure=row.tenure,
        prior_claims=int(row.prior_claims),
        credit_band=row.credit_band,
        asset_type=row.asset_type,
        asset_value=row.asset_value,
        lat=row.lat,
        lon=row.lon,
        geo_risk=row.geo_risk,
        risk_score=score,
    )


@app.post("/api/score-batch", response_model=List[ApplicantResponse])
async def score_batch(file: UploadFile = File(...), age_weight: float = 1.0):
    """Score a batch of applicants from an uploaded CSV file.

    The CSV must contain the same columns as the training data: age,
    tenure, prior_claims, credit_band, asset_type, asset_value, lat,
    lon, geo_risk.  Any additional columns are ignored.  A
    demonstration only; no data is persisted.
    """
    # Read uploaded file into DataFrame
    content = await file.read()
    try:
        uploaded_df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")
    # Ensure required columns exist
    required_cols = ['age', 'tenure', 'prior_claims', 'credit_band', 'asset_type', 'asset_value', 'lat', 'lon', 'geo_risk']
    missing = [c for c in required_cols if c not in uploaded_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")
    # Use existing RiskModel instance to transform features
    # Use model's preprocessor to transform incoming data to same feature space
    X_proc = risk_model.model.named_steps['preprocessor'].transform(
        uploaded_df[required_cols]
    )
    # Use classifier coefficients to compute linear predictor
    linear = np.dot(X_proc, risk_model.coefficients) + risk_model.intercept
    age_contribution = risk_model.coefficients[risk_model.age_idx] * X_proc[:, risk_model.age_idx]
    adjusted_linear = linear - age_contribution + age_contribution * age_weight
    probs = 1 / (1 + np.exp(-adjusted_linear))
    # Build responses
    responses: List[ApplicantResponse] = []
    for idx, (row, score) in enumerate(zip(uploaded_df.itertuples(index=False), probs)):
        responses.append(
            ApplicantResponse(
                id=idx,
                age=row.age,
                tenure=row.tenure,
                prior_claims=row.prior_claims,
                credit_band=row.credit_band,
                asset_type=row.asset_type,
                asset_value=row.asset_value,
                lat=row.lat,
                lon=row.lon,
                geo_risk=row.geo_risk,
                risk_score=float(score),
            )
        )
    return responses
