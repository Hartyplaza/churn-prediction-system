from __future__ import annotations

import io

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas.request_response import (
    BatchPredictRequest,
    BatchPredictionResponse,
    CustomerPayload,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.predictor import get_predictor_service


settings = get_settings()
app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    description="Production-style churn prediction API with explainability and retention recommendations.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor = get_predictor_service()
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        project_name=settings.project_name,
        version=settings.project_version,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    predictor = get_predictor_service()
    return ModelInfoResponse(**predictor.model_info())


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerPayload) -> PredictionResponse:
    predictor = get_predictor_service()
    try:
        result = predictor.predict_record(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictRequest | None = Body(default=None),
    file: UploadFile | None = File(default=None),
) -> BatchPredictionResponse:
    predictor = get_predictor_service()
    if payload is None and file is None:
        raise HTTPException(status_code=400, detail="Provide either JSON records or a CSV file.")

    try:
        if file is not None:
            content = await file.read()
            frame = pd.read_csv(io.BytesIO(content))
            predictions = predictor.predict_batch(frame)
        else:
            assert payload is not None
            predictions = predictor.predict_batch([record.model_dump() for record in payload.records])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BatchPredictionResponse(
        prediction_count=len(predictions),
        predictions=[PredictionResponse(**item) for item in predictions],
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    predictor = get_predictor_service()
    try:
        recommendation = predictor.recommend_only(
            payload.customer.model_dump(),
            risk_band=payload.risk_band,
            top_risk_drivers=payload.top_risk_drivers,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RecommendationResponse(**recommendation)
