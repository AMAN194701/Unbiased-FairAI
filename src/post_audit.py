"""
post_audit.py — Audits model DECISIONS for bias AFTER prediction
Part of Unbiased-FairAI | Solution Challenge 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import RandomForestClassifier


def run_post_audit(y_test, y_pred, sensitive_features_test):
    """
    Audits model predictions for fairness AFTER the model has predicted.

    Args:
        y_test                 : actual correct answers
        y_pred                 : what model predicted
        sensitive_features_test: Gender column from test set only

    Returns:
        Dictionary with fairness metrics and charts
    """

    results = {}

    # 1. OVERALL ACCURACY
    accuracy = accuracy_score(y_test, y_pred)
    results['accuracy'] = round(accuracy * 100, 2)

    # 2. ACCURACY BY GROUP
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features_test
    )
    results['accuracy_by_group'] = (mf.by_group * 100).round(2).to_dict()

    # 3. DEMOGRAPHIC PARITY DIFFERENCE
    dpd = demographic_parity_difference(
        y_test, y_pred,
        sensitive_features=sensitive_features_test
    )
    results['demographic_parity_before'] = round(abs(dpd), 4)

    # 4. EQUALIZED ODDS DIFFERENCE
    eod = equalized_odds_difference(
        y_test, y_pred,
        sensitive_features=sensitive_features_test
    )
    results['equalized_odds_before'] = round(abs(eod), 4)

    # 5. ACCURACY BY GROUP CHART
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = list(mf.by_group.index)
    values = list(mf.by_group.values * 100)
    colors = ['#4ECDC4' if v > 80 else '#FF6B6B' for v in values]
    ax.bar(groups, values, color=colors, edgecolor='black')
    ax.set_title('Model Accuracy by Gender Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.axhline(y=results['accuracy'], color='gray', linestyle='--', label='Overall Accuracy')
    ax.legend()
    plt.tight_layout()
    results['accuracy_chart'] = fig

    print("=" * 50)
    print("POST-MODEL AUDIT RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy            : {results['accuracy']}%")
    print(f"Accuracy by Group           : {results['accuracy_by_group']}")
    print(f"Demographic Parity (before) : {results['demographic_parity_before']}")
    print(f"Equalized Odds (before)     : {results['equalized_odds_before']}")
    print("=" * 50)

    return results


def run_mitigation(X_train, y_train, X_test, y_test, sensitive_features_train, sensitive_features_test):
    """
    Fixes bias using ExponentiatedGradient + DemographicParity.

    Args:
        X_train, y_train          : training data
        X_test, y_test            : test data
        sensitive_features_train  : Gender column for training set
        sensitive_features_test   : Gender column for test set

    Returns:
        Dictionary with mitigated predictions and improvement metrics
    """

    print("🔄 Running bias mitigation...")

    mitigator = ExponentiatedGradient(
        RandomForestClassifier(n_estimators=50, random_state=42),
        DemographicParity()
    )

    # Train with fairness constraint using TRAIN sensitive features
    mitigator.fit(
        X_train, y_train,
        sensitive_features=sensitive_features_train
    )

    y_pred_fair = mitigator.predict(X_test)

    # Measure bias AFTER using TEST sensitive features
    dpd_after = demographic_parity_difference(
        y_test, y_pred_fair,
        sensitive_features=sensitive_features_test
    )
    dpd_after = round(abs(dpd_after), 4)

    print("✅ Mitigation complete")

    return {
        'y_pred_fair'              : y_pred_fair,
        'demographic_parity_after' : dpd_after,
        'accuracy_after'           : round(accuracy_score(y_test, y_pred_fair) * 100, 2)
    }


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    import sys
    sys.path.append(os.path.dirname(__file__))
    from model import encode_data, predict

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
    df = pd.read_csv(data_path)

    X, y, X_train, X_test, y_train, y_test, encoders, sensitive_features = encode_data(df)

    # Gender for train AND test separately
    sensitive_train = df.loc[X_train.index, 'Gender']
    sensitive_test  = df.loc[X_test.index,  'Gender']

    with open(os.path.join(os.path.dirname(__file__), '..', 'data', 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    y_pred = predict(model, X_test)

    # Post audit
    results = run_post_audit(y_test, y_pred, sensitive_test)

    # Mitigation
    mit_results = run_mitigation(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test)

    print(f"\nDemographic Parity BEFORE : {results['demographic_parity_before']}")
    print(f"Demographic Parity AFTER  : {mit_results['demographic_parity_after']}")
    print(f"Accuracy AFTER mitigation : {mit_results['accuracy_after']}%")

    improvement = (
        (results['demographic_parity_before'] - mit_results['demographic_parity_after'])
        / results['demographic_parity_before'] * 100
    )
    print(f"Bias Reduction            : {improvement:.1f}%")
    print("\n✅ post_audit.py working correctly")