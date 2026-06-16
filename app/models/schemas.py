"""Pydantic schemas for all API requests/responses."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Symptom(BaseModel):
    name: str
    duration: Optional[str] = None
    severity: Severity = Severity.MEDIUM
    notes: Optional[str] = None

class PatientContext(BaseModel):
    age: int = Field(..., ge=0, le=130)
    sex: str
    symptoms: List[Symptom]
    medical_history: List[str] = []
    current_medications: List[str] = []
    allergies: List[str] = []
    vitals: Optional[Dict[str, str]] = None

class DiagnosisResult(BaseModel):
    session_id: str
    primary_hypothesis: str
    differential_diagnoses: List[Dict[str, str]]   # [{name, probability, reasoning}]
    red_flags: List[str]
    recommended_investigations: List[str]
    recommended_specialist: str
    confidence_score: float
    disclaimer: str
    requires_human_review: bool
    sources: List[str] = []

class FeedbackRequest(BaseModel):
    session_id: str
    diagnosis_id: str
    rating: int = Field(..., ge=1, le=5)
    correct_diagnosis: Optional[str] = None
    clinician_notes: Optional[str] = None
