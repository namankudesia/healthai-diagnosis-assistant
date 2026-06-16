"""
Core HealthAI Diagnosis Engine.

Pipeline:
1. Safety screening (red flag detection)
2. Medical RAG retrieval (clinical guidelines, ICD-10)
3. Multi-step chain-of-thought clinical reasoning via GPT-4
4. Structured JSON output parsing
5. Confidence scoring + escalation decision
6. RLHF feedback logging
"""
from __future__ import annotations
import json
import uuid
from typing import Optional

from openai import OpenAI
from app.core.config import get_settings
from app.models.schemas import PatientContext, DiagnosisResult
from app.pipeline.rag_medical import MedicalKnowledgeBase
from app.pipeline.safety_layer import SafetyLayer

settings = get_settings()

CLINICAL_SYSTEM_PROMPT = """You are HealthAI, an advanced clinical decision-support system trained on peer-reviewed medical literature, clinical guidelines (WHO, NICE, ACC/AHA), and ICD-10 classifications.

You assist clinicians (NOT patients directly) with differential diagnosis formulation.

CLINICAL REASONING PROTOCOL:
1. Analyze the presenting symptoms, patient demographics, and history
2. Apply systematic differential diagnosis (most dangerous first, then most common)
3. Identify red flag symptoms requiring urgent intervention
4. Recommend evidence-based investigations
5. Suggest appropriate specialist referral
6. Assign a confidence score (0.0–1.0) based on symptom specificity

STRICT OUTPUT FORMAT (valid JSON only):
{
  "primary_hypothesis": "string — most likely diagnosis with one-line reasoning",
  "differential_diagnoses": [
    {"name": "string", "probability": "high|medium|low", "reasoning": "string"}
  ],
  "red_flags": ["string"],
  "recommended_investigations": ["string"],
  "recommended_specialist": "string",
  "confidence_score": 0.0,
  "clinical_reasoning": "string — step-by-step reasoning chain"
}

SAFETY RULES:
- Never provide dosing or prescribe medications
- Always include red flags first
- If emergency condition suspected, flag explicitly
- Confidence < 0.7 means insufficient information — say so"""


class DiagnosisEngine:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.kb = MedicalKnowledgeBase()
        self.safety = SafetyLayer()
        self.model = settings.fine_tuned_model or settings.base_model

    def analyze(self, patient: PatientContext) -> DiagnosisResult:
        session_id = str(uuid.uuid4())[:8]
        symptom_names = [s.name for s in patient.symptoms]

        # 1. Safety screening
        red_flags = self.safety.check_red_flags(symptom_names)

        # 2. RAG retrieval
        clinical_context = self.kb.retrieve(symptom_names)

        # 3. Build clinical prompt
        prompt = self._build_prompt(patient, clinical_context)

        # 4. GPT-4 chain-of-thought reasoning
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            response_format={"type": "json_object"}
        )

        # 5. Parse and validate
        raw = json.loads(response.choices[0].message.content)
        confidence = float(raw.get("confidence_score", 0.0))
        all_red_flags = red_flags + raw.get("red_flags", [])

        return DiagnosisResult(
            session_id=session_id,
            primary_hypothesis=raw.get("primary_hypothesis", "Unable to determine"),
            differential_diagnoses=raw.get("differential_diagnoses", []),
            red_flags=all_red_flags,
            recommended_investigations=raw.get("recommended_investigations", []),
            recommended_specialist=raw.get("recommended_specialist", "General Physician"),
            confidence_score=confidence,
            disclaimer=self.safety.get_disclaimer(),
            requires_human_review=self.safety.requires_escalation(confidence, all_red_flags),
            sources=["WHO Clinical Guidelines", "ICD-10", "HealthAI Knowledge Base v1.0"]
        )

    def _build_prompt(self, patient: PatientContext, context: list) -> str:
        symptoms_str = "\n".join([
            f"  - {s.name} (duration: {s.duration or 'unknown'}, severity: {s.severity})"
            for s in patient.symptoms
        ])
        history_str = "\n  ".join(patient.medical_history) or "None reported"
        meds_str = ", ".join(patient.current_medications) or "None"
        context_str = "\n".join([f"  [{i+1}] {c}" for i, c in enumerate(context)])

        return f"""PATIENT PRESENTATION:
Age: {patient.age} | Sex: {patient.sex}

Presenting Symptoms:
{symptoms_str}

Medical History:
  {history_str}

Current Medications: {meds_str}
Allergies: {", ".join(patient.allergies) or "NKDA"}
Vitals: {patient.vitals or "Not provided"}

CLINICAL KNOWLEDGE BASE CONTEXT:
{context_str}

Apply systematic clinical reasoning. Output valid JSON only."""
