# AI-Assisted Early Detection System for Osteoarthritis (OA) Risk Markers in NER

A Streamlit MVP for collecting selected osteoarthritis risk markers and generating an **educational risk estimate**. It is not a diagnostic tool and must not replace clinical assessment.

## Features

- Patient-friendly risk assessment form
- Transparent rule-based baseline model that works immediately
- Optional scikit-learn model training from CSV data
- Explainable risk factors and prevention guidance
- NER-oriented fields such as district, community, occupation, and access to care
- No patient data is stored by the demo app

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit, usually `http://localhost:8501`.

## Train an optional model

The included `data/sample_oa_risk.csv` is **synthetic demonstration data only**. Replace it with ethically approved, de-identified clinical data before any research use.

```bash
python train_model.py --data data/sample_oa_risk.csv --output models/oa_risk_model.joblib
```

The app automatically uses `models/oa_risk_model.joblib` when present; otherwise it uses the transparent baseline score.

## Important safeguards

- Obtain ethics approval, informed consent, and community consultation before collecting real data.
- Do not upload personally identifiable or sensitive health information to this prototype.
- Validate calibration, fairness, and clinical utility across NER states, languages, ages, genders, and communities.
- A high score indicates need for follow-up—not confirmed osteoarthritis.

## Suggested next steps

1. Engage orthopaedic and primary-care clinicians in NER to define the outcome label.
2. Build a multilingual consent and questionnaire workflow.
3. Collect a representative, de-identified dataset with an approved protocol.
4. Perform external validation and prospective evaluation before deployment.
5. Add secure authentication, audit logging, encryption, and role-based access for production.
