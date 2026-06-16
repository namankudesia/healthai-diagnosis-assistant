"""
HealthAI Diagnosis Assistant — Entry Point
Run: uvicorn main:app --reload --port 8001
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="HealthAI Diagnosis Assistant",
    description="Advanced multi-modal LLM for clinical decision support. FOR CLINICIANS ONLY.",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"service": "HealthAI Diagnosis Assistant v1.0", "docs": "/docs",
            "warning": "For clinical decision support only. Not for direct patient use."}
