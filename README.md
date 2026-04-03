# ⚖️ Unbiased-FairAI

> **Detecting and mitigating bias in ML models for fair, transparent, and explainable automated decision-making.**

![Status](https://img.shields.io/badge/status-in--progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Gemini](https://img.shields.io/badge/Google-Gemini%20AI-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Cloud](https://img.shields.io/badge/Deploy-Google%20Cloud%20Run-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚨 The Problem

Automated ML models silently discriminate.

A model predicting loan approval, hiring decisions, or criminal risk **looks fair on the surface** — but buried inside are patterns that systematically disadvantage women, minorities, or low-income groups. Nobody notices. Nobody fixes it.

**Unbiased-FairAI is built to detect and fix exactly that.**

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| 🔍 Bias Detection | Measures demographic parity, equalized odds across groups |
| 📊 Fairness Dashboard | Visual before/after comparison of model fairness |
| 🔧 Bias Mitigation | Applies reweighing + threshold optimization to fix bias |
| 🤖 Gemini Explainer | Plain English explanation of WHY the model is biased |
| 📈 SHAP Analysis | Shows which features are driving unfair decisions |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML & Fairness | Python, scikit-learn, fairlearn |
| Explainability | SHAP |
| AI Explanation | Google Gemini API |
| Frontend | Streamlit |
| Deployment | Google Cloud Run |

---

---

## 🧠 Dataset

**Adult Income Dataset (UCI)**
- Task: Predict whether a person earns >$50K/year
- Known bias: gender and racial disparities in predictions
- Size: 32,561 records, 15 features

---

## 👥 Target Users

- HR teams using AI for hiring decisions
- Startups building ML-powered products
- Researchers working on ethical AI

---

## 🚀 Getting Started
```bash
# Clone the repo
git clone https://github.com/AMAN194701/Unbiased-FairAI
cd Unbiased-FairAI

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
```

---

## 🗺️ Roadmap

- [x] Project setup and repository structure
- [ ] Exploratory data analysis
- [ ] Bias detection pipeline
- [ ] Fairness metrics dashboard
- [ ] Bias mitigation using reweighing
- [ ] Gemini API integration
- [ ] Streamlit UI
- [ ] Google Cloud Run deployment
- [ ] Demo video

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
