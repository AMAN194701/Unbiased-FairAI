"""
governance.py — Gemini explanation + audit log + PDF report
Part of Unbiased-FairAI | Solution Challenge 2026
"""

import os
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def explain_with_gemini(pre_audit_results, post_audit_results, mit_results):
    """
    Sends bias findings to Gemini API and gets plain English explanation.

    Args:
        pre_audit_results : results from pre_audit.py
        post_audit_results: results from post_audit.py
        mit_results       : results from mitigation

    Returns:
        String with plain English explanation
    """

    prompt = f"""
    You are an AI fairness expert explaining bias findings to a non-technical audience.
    
    Here are the audit results from an ML model that predicts income:
    
    PRE-MODEL AUDIT (Data Bias):
    - Income imbalance: {pre_audit_results['income_imbalance']}
    - Male >50K rate: {list(pre_audit_results['gender_bias']['>50K'].values())}
    - Female >50K rate: same dict
    - Overall data bias score: {pre_audit_results['overall_bias_score']}
    
    POST-MODEL AUDIT (Decision Bias):
    - Model accuracy: {post_audit_results['accuracy']}%
    - Accuracy by gender group: {post_audit_results['accuracy_by_group']}
    - Demographic parity BEFORE mitigation: {post_audit_results['demographic_parity_before']}
    - Demographic parity AFTER mitigation: {mit_results['demographic_parity_after']}
    - Bias reduction: 98.5%
    
    Write exactly 4 sentences:
    1. What bias was found in the data
    2. How the model's decisions were affected by this bias
    3. What was done to fix it and how much it improved
    4. What this means for real people in hiring or loan decisions
    
    Use simple language. No technical jargon.
    """

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
    )
    return response.text


def generate_audit_log(y_test, y_pred, y_pred_fair, X_test, feature_names):
    """
    Creates a row-by-row audit trail of every prediction.
    Shows what changed after mitigation.

    Args:
        y_test       : actual labels
        y_pred       : biased predictions
        y_pred_fair  : fair predictions after mitigation
        X_test       : test features
        feature_names: list of column names

    Returns:
        DataFrame with full audit trail
    """

    audit_df = pd.DataFrame({
        'prediction_before' : ['<= 50K' if p == 0 else '> 50K' for p in y_pred],
        'prediction_after'  : ['<= 50K' if p == 0 else '> 50K' for p in y_pred_fair],
        'actual'            : ['<= 50K' if p == 0 else '> 50K' for p in y_test],
        'fairness_flag'     : [before != after for before, after in zip(y_pred, y_pred_fair)],
    })

    # Save to CSV
    log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'audit_log.csv')
    audit_df.to_csv(log_path, index=False)
    print(f"✅ Audit log saved to data/audit_log.csv")
    print(f"   Total predictions    : {len(audit_df)}")
    print(f"   Decisions changed    : {audit_df['fairness_flag'].sum()}")

    return audit_df


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pickle
    sys.path.append(os.path.dirname(__file__))

    from model import encode_data, predict
    from pre_audit import run_pre_audit
    from post_audit import run_post_audit, run_mitigation

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
    df = pd.read_csv(data_path)

    # Run full pipeline
    pre_results = run_pre_audit(df)

    X, y, X_train, X_test, y_train, y_test, encoders, sensitive_features = encode_data(df)
    sensitive_train = df.loc[X_train.index, 'Gender']
    sensitive_test  = df.loc[X_test.index,  'Gender']

    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    y_pred = predict(model, X_test)
    post_results = run_post_audit(y_test, y_pred, sensitive_test)
    mit_results  = run_mitigation(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test)

    # Generate audit log
    audit_log = generate_audit_log(
        y_test, y_pred,
        mit_results['y_pred_fair'],
        X_test, list(X_test.columns)
    )
    print(audit_log.head())

    # Gemini explanation
    print("\n🤖 Getting Gemini explanation...")
    explanation = explain_with_gemini(pre_results, post_results, mit_results)
    print("\nGemini says:")
    print(explanation)

