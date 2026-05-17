# ⚖️ Unbiased-FairAI
**Detecting and mitigating bias in ML models for fair, transparent, and explainable automated decision-making.**

> Google Solution Challenge 2026 | Built by Aman Kushwaha

---

## 🚨 The Problem
ML models silently discriminate. A model predicting loan approval or hiring decisions looks fair on the surface — but systematically disadvantages women or minorities. Nobody notices. Nobody fixes it.

**Unbiased-FairAI detects and fixes exactly that.**

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| 📊 Pre-Model Audit | Detects bias in raw data before training |
| 🤖 Model Training | RandomForest trained on Adult Income dataset |
| ⚖️ Fairness Metrics | Demographic parity before & after mitigation |
| 🔧 Bias Mitigation | ExponentiatedGradient reduces bias by 98.5% |
| 🤖 Gemini Explainer | Plain-English explanation of WHY model is biased |
| 🔍 What-If Simulator | Change gender/race — see if prediction changes |
| 📋 Audit Log | Full downloadable CSV audit trail |

---

## 📊 Results

| Metric | Value |
|---|---|
| Model Accuracy | 85%+ |
| Demographic Parity Before | 0.1949 |
| Demographic Parity After | 0.0029 |
| Bias Reduction | **98.5%** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML & Fairness | Python, Scikit-learn, Fairlearn |
| AI Explanation | Google Gemini API |
| Frontend | Streamlit |
| Data | UCI Adult Income Dataset |

---

## 🚀 Getting Started

```bash
git clone https://github.com/AMAN194701/Unbiased-FairAI
cd Unbiased-FairAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your Gemini API key
streamlit run src/app.py
```

---

## 🔑 Environment VariablesGEMINI_API_KEY=your_gemini_api_key_here

---

## 📄 License
MIT License
