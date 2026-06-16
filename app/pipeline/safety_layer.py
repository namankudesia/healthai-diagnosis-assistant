"""
Safety & ethics layer for clinical AI:
- Confidence thresholding → escalation
- Red flag detection → urgent care routing
- Disclaimer injection
- RLHF feedback logging
"""
from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from app.core.config import get_settings

settings = get_settings()

CRITICAL_RED_FLAGS = [
    "chest pain", "shortness of breath", "stroke", "seizure", "unconscious",
    "severe bleeding", "anaphylaxis", "suicidal", "poisoning", "head trauma",
    "aortic dissection", "pulmonary embolism", "STEMI", "sepsis"
]

DISCLAIMER = (
    "⚠️ IMPORTANT: This AI output is for educational and decision-support purposes ONLY. "
    "It does NOT constitute medical advice. Always consult a licensed healthcare professional "
    "for diagnosis and treatment. In emergencies, call 112 (India) / 911 immediately."
)


class SafetyLayer:
    def __init__(self):
        self.log_path = Path(settings.rlhf_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def check_red_flags(self, symptoms: list) -> list:
        flags = []
        symptom_text = " ".join(symptoms).lower()
        for flag in CRITICAL_RED_FLAGS:
            if flag.lower() in symptom_text:
                flags.append(f"🚨 URGENT: {flag.title()} detected — immediate evaluation required")
        return flags

    def requires_escalation(self, confidence: float, red_flags: list) -> bool:
        return confidence < settings.safety_threshold or len(red_flags) > 0

    def get_disclaimer(self) -> str:
        return DISCLAIMER

    def log_rlhf_feedback(self, feedback: dict):
        entry = {**feedback, "timestamp": datetime.now().isoformat()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def sanitize_input(self, text: str) -> str:
        text = re.sub(r"[<>{}\\]", "", text)
        return text[:2000]
