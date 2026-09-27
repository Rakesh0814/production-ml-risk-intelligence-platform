from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR
from .model_service import get_model_service
from .schemas import CreditRiskRequest

app = FastAPI(
    title="Production ML Risk Intelligence Platform",
    version="1.0.0",
    description="Credit default-risk scoring with XGBoost and SHAP.",
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    service = get_model_service()
    return {
        "status": "ok",
        "model_loaded": service.loaded,
        "model": service.metadata.get("model", "XGBoost"),
    }


@app.get("/api/model-info")
def model_info():
    service = get_model_service()
    return {
        "model_loaded": service.loaded,
        **service.metadata,
    }


@app.post("/api/predict")
def predict(request: CreditRiskRequest):
    service = get_model_service()

    try:
        return service.predict(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
