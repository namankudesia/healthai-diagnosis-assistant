"""FastAPI routes for HealthAI Diagnosis Assistant."""
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import PatientContext, DiagnosisResult, FeedbackRequest
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()

def get_engine():
    from app.core.diagnosis_engine import DiagnosisEngine
    return DiagnosisEngine()

@router.post("/analyze", response_model=DiagnosisResult,
             summary="Analyze patient symptoms and generate differential diagnosis")
async def analyze(patient: PatientContext):
    try:
        engine = get_engine()
        result = engine.analyze(patient)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback", summary="Submit clinician feedback for RLHF training")
async def feedback(req: FeedbackRequest):
    from app.pipeline.safety_layer import SafetyLayer
    SafetyLayer().log_rlhf_feedback(req.model_dump())
    return {"status": "feedback logged", "session_id": req.session_id}

@router.get("/health")
async def health():
    return {"status": "ok", "model": settings.fine_tuned_model or settings.base_model,
            "service": settings.app_name}
