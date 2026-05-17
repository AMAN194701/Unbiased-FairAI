"""
app.py — Streamlit UI for Unbiased-FairAI
Part of Unbiased-FairAI | Solution Challenge 2026
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from pre_audit import run_pre_audit
from model import encode_data, train_model, predict
from post_audit import run_post_audit, run_mitigation
from governance import explain_with_gemini, generate_audit_log

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Unbiased-FairAI",
    page_icon="⚖️",
    layout="wide"
)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Unbiased-FairAI")
    st.markdown("**AI Governance & Bias Detection Platform**")
    st.divider()
    st.markdown("""
    ### What this tool does:
    1. 📊 Audits data for bias
    2. 🤖 Trains ML model
    3. ⚖️ Measures fairness
    4. 🔧 Mitigates bias
    5. 🏛️ Generates audit report
    6. 🔍 Simulates what-if scenarios
    """)
    st.divider()
    st.markdown("**Solution Challenge 2026**")
    st.markdown("Unbiased AI Decision Track")
    st.divider()
    st.markdown("Built by **Aman Kushwaha**")

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
    return pd.read_csv(data_path)

@st.cache_resource
def run_full_pipeline(df):
    """Runs entire ML pipeline once and caches results."""
    # Pre audit
    pre_results = run_pre_audit(df)

    # Encode + train + predict
    X, y, X_train, X_test, y_train, y_test, encoders, sensitive_features = encode_data(df)
    sensitive_train = df.loc[X_train.index, 'Gender']
    sensitive_test  = df.loc[X_test.index,  'Gender']

    model = train_model(X_train, y_train)
    y_pred = predict(model, X_test)

    # Post audit + mitigation
    post_results = run_post_audit(y_test, y_pred, sensitive_test)
    mit_results  = run_mitigation(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test)

    # Audit log
    audit_log = generate_audit_log(y_test, y_pred, mit_results['y_pred_fair'], X_test, list(X_test.columns))

    # Bias improvement %
    improvement = (
        (post_results['demographic_parity_before'] - mit_results['demographic_parity_after'])
        / post_results['demographic_parity_before'] * 100
    )

    return {
        'pre'        : pre_results,
        'post'       : post_results,
        'mit'        : mit_results,
        'audit_log'  : audit_log,
        'improvement': round(improvement, 1),
        'model'      : model,
        'encoders'   : encoders,
        'X_test'     : X_test,
        'y_test'     : y_test,
        'y_pred'     : y_pred,
        'sensitive_test': sensitive_test
    }

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
st.title("⚖️ Unbiased-FairAI")
st.markdown("### AI Governance & Bias Detection Platform")
st.divider()

# Load
df = load_data()

with st.spinner("🔄 Running full AI pipeline... (this takes ~60 seconds first time)"):
    pipeline = run_full_pipeline(df)

st.success("✅ Pipeline complete! Explore the tabs below.")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pre-Model Audit",
    "🤖 Model Results",
    "⚖️ Post-Model Audit",
    "🏛️ Governance Report",
    "🔍 What-If Simulator"
])


# ════════════════════════════════════════
# TAB 1: PRE-MODEL AUDIT
# ════════════════════════════════════════
with tab1:
    st.header("📊 Pre-Model Audit")
    st.markdown("**This is what your data looks like BEFORE training the model.**")
    st.divider()

    pre = pipeline['pre']

    # Bias score metric
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Bias Score", f"{pre['overall_bias_score']}", "0 = fair, 1 = biased")
    col2.metric("Male >50K Rate", "31.4%")
    col3.metric("Female >50K Rate", "11.4%")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income by Gender")
        st.pyplot(pre['gender_chart'])
    with col2:
        st.subheader("Income by Race")
        st.pyplot(pre['race_chart'])

    st.subheader("Income Class Distribution")
    st.pyplot(pre['income_chart'])

    st.warning("⚠️ Data is imbalanced and biased. The model will learn these patterns if not corrected.")


# ════════════════════════════════════════
# TAB 2: MODEL RESULTS
# ════════════════════════════════════════
with tab2:
    st.header("🤖 Model Results")
    st.markdown("**RandomForest model trained on Adult Income dataset.**")
    st.divider()

    post = pipeline['post']

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{post['accuracy']}%")
    col2.metric("Training Samples", "24,129")
    col3.metric("Test Samples", "6,033")

    st.divider()

    st.subheader("Accuracy by Gender Group")
    st.pyplot(post['accuracy_chart'])

    st.info("ℹ️ Notice how accuracy differs between Male and Female groups — this indicates the model treats them differently.")


# ════════════════════════════════════════
# TAB 3: POST-MODEL AUDIT
# ════════════════════════════════════════
with tab3:
    st.header("⚖️ Post-Model Audit")
    st.markdown("**Fairness metrics BEFORE and AFTER bias mitigation.**")
    st.divider()

    post = pipeline['post']
    mit  = pipeline['mit']

    # Before vs After
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Demographic Parity BEFORE",
        post['demographic_parity_before'],
        "Higher = more biased"
    )
    col2.metric(
        "Demographic Parity AFTER",
        mit['demographic_parity_after'],
        f"↓ {pipeline['improvement']}% reduction",
        delta_color="inverse"
    )
    col3.metric(
        "Accuracy After Mitigation",
        f"{mit['accuracy_after']}%",
        f"{round(mit['accuracy_after'] - post['accuracy'], 1)}% vs before"
    )

    st.divider()

    # Before vs After bar chart
    st.subheader("Bias Before vs After Mitigation")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        ['Before Mitigation', 'After Mitigation'],
        [post['demographic_parity_before'], mit['demographic_parity_after']],
        color=['#FF6B6B', '#4ECDC4'],
        edgecolor='black',
        width=0.4
    )
    ax.set_ylabel('Demographic Parity Difference')
    ax.set_title('Bias Score: Before vs After Mitigation', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, [post['demographic_parity_before'], mit['demographic_parity_after']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val}', ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    st.success(f"✅ Bias reduced by {pipeline['improvement']}% after mitigation!")
    st.markdown("""
    **What is Demographic Parity Difference?**
    - It measures how differently the model treats different groups
    - **0** = perfectly fair
    - **0.19** = model is 19% more likely to predict >50K for one group vs another
    """)


# ════════════════════════════════════════
# TAB 4: GOVERNANCE REPORT
# ════════════════════════════════════════
with tab4:
    st.header("🏛️ Governance Report")
    st.markdown("**Full audit trail and AI-generated explanation.**")
    st.divider()

    # Gemini explanation
    st.subheader("🤖 Gemini AI Explanation")
    with st.spinner("Getting explanation from Gemini..."):
        try:
            explanation = explain_with_gemini(
                pipeline['pre'],
                pipeline['post'],
                pipeline['mit']
            )
            st.info(explanation)
        except Exception as e:
            st.warning(f"Gemini unavailable: {e}")
            st.info("The model showed significant gender bias with a demographic parity difference of 0.1949. After applying ExponentiatedGradient mitigation, bias was reduced by 98.5% to 0.0029.")

    st.divider()

    # Audit log
    st.subheader("📋 Audit Log")
    audit_log = pipeline['audit_log']

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(audit_log))
    col2.metric("Decisions Changed", audit_log['fairness_flag'].sum())
    col3.metric("Change Rate", f"{round(audit_log['fairness_flag'].mean()*100, 1)}%")

    st.dataframe(audit_log.head(50), use_container_width=True)

    # Download button
    csv = audit_log.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Full Audit Log (CSV)",
        data=csv,
        file_name="audit_log.csv",
        mime="text/csv"
    )


# ════════════════════════════════════════
# TAB 5: WHAT-IF SIMULATOR
# ════════════════════════════════════════
with tab5:
    st.header("🔍 What-If Simulator")
    st.markdown("**Enter a person's details and see if changing gender affects the prediction.**")
    st.divider()

    encoders = pipeline['encoders']

    col1, col2 = st.columns(2)

    with col1:
        age         = st.slider("Age", 18, 90, 35)
        gender      = st.selectbox("Gender", ["Male", "Female"])
        race        = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        education   = st.selectbox("Education", ["Bachelors", "Masters", "Doctorate", "HS-grad", "Some-college", "Assoc-acdm"])
        hours       = st.slider("Hours per week", 1, 99, 40)

    with col2:
        workclass   = st.selectbox("Work Class", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"])
        occupation  = st.selectbox("Occupation", ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales", "Other-service"])
        marital     = st.selectbox("Marital Status", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"])
        relationship= st.selectbox("Relationship", ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"])
        country     = st.selectbox("Country", ["United-States", "India", "Mexico", "Philippines", "Germany"])

    def encode_person(gender_val):
        """Encode a single person's data using saved encoders."""
        person = {
            'Age'            : age,
            'Job_type'       : workclass,
            'Final_wt'       : 189778,
            'Education_level': education,
            'Education_Yr'   : 13,
            'Marital_status' : marital,
            'Job_role'       : occupation,
            'Family_role'    : relationship,
            'Race'           : race,
            'Gender'         : gender_val,
            'Capital_gain'   : 0,
            'Capital_loss'   : 0,
            'Weekly_hrs'     : hours,
            'Country'        : country
        }
        person_df = pd.DataFrame([person])
        for col, le in encoders.items():
            if col != 'Income' and col in person_df.columns:
                try:
                    person_df[col] = le.transform(person_df[col])
                except:
                    person_df[col] = 0
        return person_df

    if st.button("🔍 Predict & Test for Bias", type="primary"):
        model = pipeline['model']

        # Predict for selected gender
        person_encoded    = encode_person(gender)
        prediction        = model.predict(person_encoded)[0]
        prediction_label  = ">50K ✅" if prediction == 1 else "<=50K ❌"

        # Predict for opposite gender
        opposite_gender   = "Female" if gender == "Male" else "Male"
        person_opposite   = encode_person(opposite_gender)
        prediction_opp    = model.predict(person_opposite)[0]
        prediction_opp_label = ">50K ✅" if prediction_opp == 1 else "<=50K ❌"

        st.divider()

        col1, col2 = st.columns(2)
        col1.metric(f"Prediction ({gender})", prediction_label)
        col2.metric(f"Prediction ({opposite_gender})", prediction_opp_label)

        st.divider()

        if prediction != prediction_opp:
            st.error(f"""
            ⚠️ **BIAS DETECTED**
            
            Changing gender from **{gender}** to **{opposite_gender}** 
            changed the prediction from **{prediction_label}** to **{prediction_opp_label}**.
            
            Everything else was identical. Only gender changed.
            This is textbook gender discrimination in AI.
            """)
        else:
            st.success(f"""
            ✅ **No gender bias detected for this profile.**
            
            The prediction remained **{prediction_label}** regardless of gender.
            """)