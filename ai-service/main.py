import math
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import IsolationForest, RandomForestRegressor


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class RiskScoreRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    temperature: float = Field(default=30, alias="temperature")
    humidity: float = Field(default=50, alias="humidity")
    rainfall: float = Field(default=0, alias="rainfall")
    wind_speed: float = Field(default=5, alias="windSpeed")
    latitude: float = Field(default=19.0, alias="latitude")
    longitude: float = Field(default=72.0, alias="longitude")
    historical_claims: int = Field(default=0, alias="historicalClaims")
    aqi: int = Field(default=50, alias="aqi")


class RiskScoreResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    risk_score: float = Field(alias="riskScore")
    suggested_premium: float = Field(alias="suggestedPremium")
    risk_level: str = Field(alias="riskLevel")


class FraudCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(default=0, alias="userId")
    claim_amount: float = Field(default=0, alias="claimAmount")
    latitude: float = Field(default=0, alias="latitude")
    longitude: float = Field(default=0, alias="longitude")
    user_latitude: float = Field(default=0, alias="userLatitude")
    user_longitude: float = Field(default=0, alias="userLongitude")
    temperature: float = Field(default=30, alias="temperature")
    humidity: float = Field(default=50, alias="humidity")
    rainfall: float = Field(default=0, alias="rainfall")
    wind_speed: float = Field(default=5, alias="windSpeed")
    claim_frequency: int = Field(default=0, alias="claimFrequency")
    total_claims: int = Field(default=0, alias="totalClaims")


class FraudCheckResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    is_fraudulent: bool = Field(alias="isFraudulent")
    fraud_score: float = Field(alias="fraudScore")
    reason: str = Field(alias="reason")


# ---------------------------------------------------------------------------
# Globals – populated during startup
# ---------------------------------------------------------------------------

risk_model: Optional[RandomForestRegressor] = None
fraud_model: Optional[IsolationForest] = None


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two points."""
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Synthetic data generation & training
# ---------------------------------------------------------------------------

def _train_risk_model() -> RandomForestRegressor:
    """Train a RandomForestRegressor on synthetic Indian weather risk data."""
    rng = np.random.RandomState(42)
    n = 1200

    temperature = rng.uniform(20, 48, n)
    humidity = rng.uniform(30, 100, n)
    rainfall = rng.uniform(0, 100, n)
    wind_speed = rng.uniform(0, 35, n)
    latitude = rng.uniform(8, 37, n)
    longitude = rng.uniform(68, 97, n)
    historical_claims = rng.randint(0, 20, n)
    aqi = rng.randint(0, 500, n)

    # Component risks normalised to 0-100
    temp_risk = np.clip((temperature - 20) / (48 - 20) * 100, 0, 100)
    humidity_risk = np.clip((humidity - 30) / (100 - 30) * 100, 0, 100)
    rainfall_risk = np.clip(rainfall / 100 * 100, 0, 100)
    wind_risk = np.clip(wind_speed / 35 * 100, 0, 100)

    risk_score = (
        0.3 * temp_risk
        + 0.2 * humidity_risk
        + 0.3 * rainfall_risk
        + 0.2 * wind_risk
        + rng.normal(0, 5, n)
    )
    risk_score = np.clip(risk_score, 0, 100)

    x = np.column_stack(
        [temperature, humidity, rainfall, wind_speed, latitude, longitude, historical_claims, aqi]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, risk_score)
    return model


def _train_fraud_model() -> IsolationForest:
    """Train an IsolationForest on synthetic normal + anomalous claims."""
    rng = np.random.RandomState(42)

    # --- 900 normal samples ---
    n_normal = 900
    claim_amount_n = rng.uniform(1000, 50000, n_normal)
    lat_n = rng.uniform(8, 37, n_normal)
    lon_n = rng.uniform(68, 97, n_normal)
    # User location close to claim location
    user_lat_n = lat_n + rng.normal(0, 0.5, n_normal)
    user_lon_n = lon_n + rng.normal(0, 0.5, n_normal)
    temp_n = rng.uniform(20, 45, n_normal)
    hum_n = rng.uniform(30, 95, n_normal)
    rain_n = rng.uniform(0, 80, n_normal)
    wind_n = rng.uniform(0, 25, n_normal)
    freq_n = rng.randint(0, 3, n_normal)
    total_n = rng.randint(0, 5, n_normal)
    dist_n = np.array(
        [haversine(lat_n[i], lon_n[i], user_lat_n[i], user_lon_n[i]) for i in range(n_normal)]
    )

    # --- 100 anomalous samples ---
    n_anom = 100
    claim_amount_a = rng.uniform(80000, 500000, n_anom)
    lat_a = rng.uniform(8, 37, n_anom)
    lon_a = rng.uniform(68, 97, n_anom)
    # User location far from claim location
    user_lat_a = lat_a + rng.uniform(5, 15, n_anom) * rng.choice([-1, 1], n_anom)
    user_lon_a = lon_a + rng.uniform(5, 15, n_anom) * rng.choice([-1, 1], n_anom)
    temp_a = rng.uniform(5, 55, n_anom)
    hum_a = rng.uniform(5, 100, n_anom)
    rain_a = rng.uniform(0, 200, n_anom)
    wind_a = rng.uniform(0, 50, n_anom)
    freq_a = rng.randint(5, 20, n_anom)
    total_a = rng.randint(10, 50, n_anom)
    dist_a = np.array(
        [haversine(lat_a[i], lon_a[i], user_lat_a[i], user_lon_a[i]) for i in range(n_anom)]
    )

    # Combine
    x = np.vstack(
        [
            np.column_stack([claim_amount_n, lat_n, lon_n, user_lat_n, user_lon_n,
                             temp_n, hum_n, rain_n, wind_n, freq_n, total_n, dist_n]),
            np.column_stack([claim_amount_a, lat_a, lon_a, user_lat_a, user_lon_a,
                             temp_a, hum_a, rain_a, wind_a, freq_a, total_a, dist_a]),
        ]
    )

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(x)
    return model


# ---------------------------------------------------------------------------
# Application lifespan – train models on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global risk_model, fraud_model
    risk_model = _train_risk_model()
    fraud_model = _train_fraud_model()
    yield


app = FastAPI(title="EarnSafe AI Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": risk_model is not None and fraud_model is not None}


@app.post("/predict/risk", response_model=RiskScoreResponse, response_model_by_alias=True)
async def predict_risk(req: RiskScoreRequest):
    features = np.array(
        [[req.temperature, req.humidity, req.rainfall, req.wind_speed,
          req.latitude, req.longitude, req.historical_claims, req.aqi]]
    )
    score = float(risk_model.predict(features)[0])  # type: ignore[union-attr]
    score = max(0.0, min(100.0, score))
    premium = 50 + (score / 100) * 150

    if score < 33:
        level = "LOW"
    elif score < 66:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return RiskScoreResponse(
        risk_score=round(score, 2),
        suggested_premium=round(premium, 2),
        risk_level=level,
    )


@app.post("/predict/fraud", response_model=FraudCheckResponse, response_model_by_alias=True)
async def predict_fraud(req: FraudCheckRequest):
    dist = haversine(req.latitude, req.longitude, req.user_latitude, req.user_longitude)

    features = np.array(
        [[req.claim_amount, req.latitude, req.longitude,
          req.user_latitude, req.user_longitude,
          req.temperature, req.humidity, req.rainfall, req.wind_speed,
          req.claim_frequency, req.total_claims, dist]]
    )

    prediction = fraud_model.predict(features)[0]  # type: ignore[union-attr]
    raw_score = fraud_model.decision_function(features)[0]  # type: ignore[union-attr]

    # Map decision_function output to 0-1 fraud score: offset centres normal
    # samples around 0.5, negative raw_score (anomaly) pushes fraud_score > 0.5
    FRAUD_SCORE_OFFSET = 0.5
    fraud_score = max(0.0, min(1.0, FRAUD_SCORE_OFFSET - raw_score))
    is_fraud = bool(prediction == -1)

    reasons = []
    if dist > 100:
        reasons.append("large distance between claim and user location")
    if req.claim_amount > 100000:
        reasons.append("unusually high claim amount")
    if req.claim_frequency > 5:
        reasons.append("high claim frequency")
    if req.total_claims > 10:
        reasons.append("excessive total claims")

    reason = "; ".join(reasons) if reasons else ("anomaly detected by model" if is_fraud else "no fraud indicators detected")

    return FraudCheckResponse(
        is_fraudulent=is_fraud,
        fraud_score=round(fraud_score, 4),
        reason=reason,
    )
