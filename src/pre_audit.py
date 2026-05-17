"""
pre_audit.py — Audits raw data for bias BEFORE model training
Improved version (production-ready)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Required for Streamlit


def run_pre_audit(df):
    """
    Scans dataset for bias before model training

    Args:
        df: pandas DataFrame

    Returns:
        dict with metrics + charts
    """

    results = {}

    # ─────────────────────────────────────────
    # 0. VALIDATION (IMPORTANT)
    # ─────────────────────────────────────────
    required_cols = ['Income', 'Gender', 'Race']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # ─────────────────────────────────────────
    # 1. INCOME DISTRIBUTION
    # ─────────────────────────────────────────
    income_counts = df['Income'].value_counts(normalize=True) * 100
    income_counts = income_counts.reindex(['<=50K', '>50K'], fill_value=0)

    results['income_imbalance'] = income_counts.to_dict()

    # ─────────────────────────────────────────
    # 2. GENDER BIAS
    # ─────────────────────────────────────────
    gender_bias = pd.crosstab(
        df['Gender'], df['Income'], normalize='index'
    ) * 100

    gender_bias = gender_bias.reindex(columns=['<=50K', '>50K'], fill_value=0)

    results['gender_bias'] = gender_bias.to_dict()

    # ─────────────────────────────────────────
    # 3. RACE BIAS
    # ─────────────────────────────────────────
    race_bias = pd.crosstab(
        df['Race'], df['Income'], normalize='index'
    ) * 100

    race_bias = race_bias.reindex(columns=['<=50K', '>50K'], fill_value=0)

    # sort for better visualization
    if '>50K' in race_bias.columns:
        race_bias = race_bias.sort_values(by='>50K', ascending=False)

    results['race_bias'] = race_bias.to_dict()

    # ─────────────────────────────────────────
    # 4. BIAS SCORE
    # ─────────────────────────────────────────
    male_rate = gender_bias.loc['Male', '>50K'] if 'Male' in gender_bias.index else 0
    female_rate = gender_bias.loc['Female', '>50K'] if 'Female' in gender_bias.index else 0

    bias_score = round(abs(male_rate - female_rate) / 100, 4)

    # disparity ratio (important metric)
    disparity_ratio = round((female_rate / male_rate), 2) if male_rate != 0 else 0

    results['overall_bias_score'] = bias_score
    results['disparity_ratio'] = disparity_ratio

    # bias flag
    if bias_score > 0.1:
        results['bias_flag'] = "⚠️ Potential bias detected"
    else:
        results['bias_flag'] = "✅ Fair dataset"

    # ─────────────────────────────────────────
    # 5. GENDER CHART
    # ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    gender_bias.plot(
        kind='bar',
        ax=ax1,
        color=['#FF6B6B', '#4ECDC4'],
        edgecolor='black'
    )

    ax1.set_title('Gender-wise Income Distribution (Bias Analysis)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gender')
    ax1.set_ylabel('Percentage (%)')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(['<=50K', '>50K'])

    plt.tight_layout(pad=1.5)
    results['gender_chart'] = fig1

    # ─────────────────────────────────────────
    # 6. RACE CHART
    # ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    race_bias.plot(
        kind='bar',
        ax=ax2,
        color=['#FF6B6B', '#4ECDC4'],
        edgecolor='black'
    )

    ax2.set_title('Race-wise Income Distribution (Bias Analysis)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Race')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=30)

    plt.xticks(ha='right')
    ax2.legend(['<=50K', '>50K'])

    plt.tight_layout(pad=1.5)
    results['race_chart'] = fig2

    # ─────────────────────────────────────────
    # 7. PIE CHART (FIXED)
    # ─────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 4))

    income_counts.plot(
        kind='pie',
        ax=ax3,
        colors=['#FF6B6B', '#4ECDC4'],
        autopct='%1.1f%%',
        startangle=90
    )

    ax3.set_title('Income Class Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('')
    ax3.axis('equal')  # FIX: circle distortion

    plt.tight_layout(pad=1.5)
    results['income_chart'] = fig3

    # ─────────────────────────────────────────
    # LOG OUTPUT (DEBUGGING)
    # ─────────────────────────────────────────
    print("=" * 50)
    print("PRE-MODEL AUDIT RESULTS")
    print("=" * 50)
    print(f"Income Imbalance   : {income_counts.to_dict()}")
    print(f"Male >50K rate     : {male_rate:.1f}%")
    print(f"Female >50K rate   : {female_rate:.1f}%")
    print(f"Bias Score         : {bias_score}")
    print(f"Disparity Ratio    : {disparity_ratio}")
    print(f"Status             : {results['bias_flag']}")
    print("=" * 50)

    return results


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'clean_data.csv'
    )

    df = pd.read_csv(data_path)
    results = run_pre_audit(df)

    print(f"\nCharts generated: {[k for k in results if 'chart' in k]}")
    print("✅ pre_audit.py working correctly")