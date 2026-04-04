"""
pre_audit.py — Audits raw data for bias BEFORE model training
Part of Unbiased-FairAI | Solution Challenge 2026
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # prevents display issues in Streamlit


def run_pre_audit(df):
    """
    Scans raw dataset for bias before any model is trained.
    Think of it like a doctor checkup before surgery.

    Args:
        df: cleaned pandas DataFrame

    Returns:
        Dictionary with all bias findings and charts
    """

    results = {}

    # ─────────────────────────────────────────
    # 1. INCOME IMBALANCE
    # ─────────────────────────────────────────
    income_counts = df['Income'].value_counts(normalize=True) * 100
    results['income_imbalance'] = income_counts.to_dict()

    # ─────────────────────────────────────────
    # 2. GENDER BIAS
    # ─────────────────────────────────────────
    gender_bias = pd.crosstab(
        df['Gender'], df['Income'], normalize='index'
    ) * 100
    results['gender_bias'] = gender_bias.to_dict()

    # ─────────────────────────────────────────
    # 3. RACE BIAS
    # ─────────────────────────────────────────
    race_bias = pd.crosstab(
        df['Race'], df['Income'], normalize='index'
    ) * 100
    results['race_bias'] = race_bias.to_dict()

    # ─────────────────────────────────────────
    # 4. OVERALL BIAS SCORE (0 = fair, 1 = very biased)
    # Logic: difference between male and female >50K rates
    # divided by 100 to normalize between 0 and 1
    # ─────────────────────────────────────────
    male_rate = gender_bias.loc['Male', '>50K'] if 'Male' in gender_bias.index else 0
    female_rate = gender_bias.loc['Female', '>50K'] if 'Female' in gender_bias.index else 0
    bias_score = round(abs(male_rate - female_rate) / 100, 4)
    results['overall_bias_score'] = bias_score

    # ─────────────────────────────────────────
    # 5. GENDER BIAS CHART
    # ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    gender_bias.plot(
        kind='bar',
        ax=ax1,
        color=['#FF6B6B', '#4ECDC4'],
        edgecolor='black'
    )
    ax1.set_title('Income Distribution by Gender', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gender')
    ax1.set_ylabel('Percentage (%)')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(['<=50K', '>50K'])
    plt.tight_layout()
    results['gender_chart'] = fig1

    # ─────────────────────────────────────────
    # 6. RACE BIAS CHART
    # ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    race_bias.plot(
        kind='bar',
        ax=ax2,
        color=['#FF6B6B', '#4ECDC4'],
        edgecolor='black'
    )
    ax2.set_title('Income Distribution by Race', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Race')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(['<=50K', '>50K'])
    plt.tight_layout()
    results['race_chart'] = fig2

    # ─────────────────────────────────────────
    # 7. INCOME PIE CHART
    # ─────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    income_counts.plot(
        kind='pie',
        ax=ax3,
        colors=['#FF6B6B', '#4ECDC4'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax3.set_title('Income Class Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('')
    plt.tight_layout()
    results['income_chart'] = fig3

    # Print summary
    print("=" * 50)
    print("PRE-MODEL AUDIT RESULTS")
    print("=" * 50)
    print(f"Income Imbalance   : {income_counts.to_dict()}")
    print(f"Male >50K rate     : {male_rate:.1f}%")
    print(f"Female >50K rate   : {female_rate:.1f}%")
    print(f"Overall Bias Score : {bias_score} (0=fair, 1=biased)")
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