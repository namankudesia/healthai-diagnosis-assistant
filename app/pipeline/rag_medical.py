"""
Medical RAG pipeline with specialty-aware retrieval.
Retrieves from curated medical knowledge base (ICD-10, clinical guidelines, drug interactions).
"""
from __future__ import annotations
import json
from typing import List, Dict, Optional
from pathlib import Path


class MedicalKnowledgeBase:
    """
    In production: connects to a FAISS/Pinecone index built from:
    - PubMed abstracts (via BioASQ dataset)
    - ICD-10 clinical descriptions
    - WHO clinical guidelines
    - DrugBank interaction data
    This stub demonstrates the interface.
    """
    SAMPLE_KNOWLEDGE = {
        "chest_pain": [
            "Differential for acute chest pain includes: ACS (STEMI/NSTEMI), pulmonary embolism, aortic dissection, pneumothorax, pericarditis, GERD.",
            "Red flags in chest pain: radiation to jaw/left arm, diaphoresis, syncope, hypotension, pulse deficit.",
            "ECG and troponin I/T are first-line investigations for suspected ACS."
        ],
        "dyspnea": [
            "Acute dyspnea differentials: pulmonary edema, pneumonia, COPD exacerbation, pulmonary embolism, cardiac tamponade.",
            "BNP >500 pg/mL suggests heart failure. D-dimer used to rule out PE.",
        ],
        "fever": [
            "Fever workup: CBC with differential, CRP, procalcitonin, blood cultures x2, urinalysis.",
            "Fever >38.5°C with rigors suggests bacteremia. Fever with neck stiffness: LP to rule out meningitis."
        ]
    }

    def retrieve(self, symptoms: List[str], k: int = 5) -> List[str]:
        results = []
        for symptom in symptoms:
            sym_lower = symptom.lower()
            for key, docs in self.SAMPLE_KNOWLEDGE.items():
                if key in sym_lower or any(w in sym_lower for w in key.split("_")):
                    results.extend(docs)
        return list(set(results))[:k] if results else ["No specific clinical guidelines found for these symptoms. Use general clinical reasoning."]
