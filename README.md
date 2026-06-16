# HealthAI Diagnosis Assistant

> Advanced clinical decision-support LLM with **RAG + Fine-Tuning + RLHF pipeline** — built for real-world healthcare workflows

---

## What This Is

HealthAI is a production-grade AI system that assists clinicians with differential diagnosis. It is **NOT** a consumer chatbot — it is a clinician-facing decision-support tool with:

- **GPT-4 + Medical RAG** for grounded clinical reasoning
- **Fine-tuning pipeline** (OpenAI fine-tuning API) on curated clinical cases
- **RLHF feedback loop** — clinician ratings improve the model over time
- **Multi-layer safety system** — red flag detection, confidence thresholding, mandatory disclaimers
- **Structured output** — parseable JSON with differential diagnoses, investigations, specialist routing

---

## Architecture

```
Patient Data (symptoms, history, vitals)
        │
        ▼
Safety Screening Layer          ← Red flag detection (PE, STEMI, sepsis, etc.)
        │
        ▼
Medical RAG Retrieval           ← ICD-10, WHO guidelines, PubMed abstracts (FAISS)
        │
        ▼
Chain-of-Thought Clinical       ← GPT-4 / Fine-tuned model
Reasoning (JSON output)
        │
        ▼
Confidence Scoring              ← < 0.85 → flag for human review
        │
        ▼
Structured DiagnosisResult      ← Differentials, investigations, specialist, disclaimer
        │
        ▼
RLHF Feedback Logging           ← Clinician ratings → future fine-tuning
```

---

## How to Run

```bash
git clone https://github.com/namankudesia/healthai-diagnosis-assistant.git
cd healthai-diagnosis-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY

uvicorn main:app --reload --port 8001
```

### Example API Call

```bash
curl -X POST http://localhost:8001/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "age": 52,
    "sex": "male",
    "symptoms": [
      {"name": "chest pain", "duration": "2 hours", "severity": "high"},
      {"name": "diaphoresis", "duration": "1 hour", "severity": "high"},
      {"name": "left arm radiation", "duration": "2 hours", "severity": "medium"}
    ],
    "medical_history": ["hypertension", "type 2 diabetes"],
    "current_medications": ["metformin", "amlodipine"]
  }'
```

---

## Fine-Tuning Pipeline

```bash
# Step 1: Prepare training data from clinical cases
python training/prepare_dataset.py --input data/clinical_cases.csv --output data/finetune/

# Step 2: Launch fine-tuning job
python training/finetune_job.py --train data/finetune/train.jsonl --val data/finetune/val.jsonl

# Step 3: Update .env with returned model ID, restart server
```

---

## RLHF Feedback Loop

```bash
# After a clinician reviews a diagnosis, submit feedback:
curl -X POST http://localhost:8001/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "diagnosis_id": "TC_001", "rating": 5,
       "correct_diagnosis": "NSTEMI", "clinician_notes": "Correct. Troponin confirmed."}'

# Feedback saved to data/rlhf_feedback.jsonl for next training cycle
```

---

## Safety & Ethics
- **Never diagnose patients directly** — clinician-facing only
- Red flags trigger automatic escalation flags
- Confidence < 85% forces human review
- All outputs carry mandatory medical disclaimer
- RLHF feedback loop ensures continuous improvement

## Tech Stack
`FastAPI` · `OpenAI GPT-4` · `Fine-Tuning API` · `FAISS` · `Pydantic` · `Python 3.11`

> ⚠️ This system is for clinical decision support only. Not for direct patient use.

> Built by [Naman Kudesia](https://github.com/namankudesia)
