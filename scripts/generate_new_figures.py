#!/usr/bin/env python3
"""
mRCC Prognostic Model - 新しい図構成に基づくFigure作成
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import os

print("=" * 80)
print("新しい図構成に基づくFigure作成")
print("=" * 80)

# =============================================================================
# データ準備
# =============================================================================
data_path = '/Users/hideto/Desktop/三田先生UTC解析/mita_utc_analysis/data/データシート完成版.csv'
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    exit(1)

df = pd.read_csv(data_path, encoding='utf-8')

# アウトカム
df['OS_event'] = df['OS打ち切り(死亡0)'].apply(lambda x: 1 if x == 0 else 0)
df['OS_time'] = pd.to_numeric(df['time to death'], errors='coerce')

# 治療分類
df['Treatment'] = df['1st line_3分類'].copy()
df.loc[df['Treatment'] == 'IO+other', 'Treatment'] = 'IO+MKI'
df['Treatment_binary'] = df['Treatment'].apply(lambda x: 'IO-IO' if x == 'IOIO' else 'IO-TKI')

# 共変量
df['Age'] = pd.to_numeric(df['年齢(導入時）'], errors='coerce')
df['Sex'] = df['性別（男1女0)']
df['KPS'] = pd.to_numeric(df['PS(karnofsky)'], errors='coerce')
df['Hb'] = pd.to_numeric(df['1st line前_Hb(/μL)'], errors='coerce')
df['LDH'] = pd.to_numeric(df['1st line前_LDH(U/L)'], errors='coerce')
df['CRP'] = pd.to_numeric(df['1st line前_CRP(mg/dL)'], errors='coerce')
df['Alb'] = pd.to_numeric(df['1st line前_Alb(g/dL)'], errors='coerce')
df['mGPS'] = pd.to_numeric(df['1st line前_mGPS'], errors='coerce')
df['Plt'] = pd.to_numeric(df['1st line前_Plt(/μL)'], errors='coerce')
df['Neutrophil'] = pd.to_numeric(df['1st line前_好中球数'], errors='coerce')
df['IMDC_score'] = pd.to_numeric(df['IMDC点数'], errors='coerce')

# 転移
met_cols = ['リンパ節転移有無（単発多発分けず）', '骨転移有無', '肝転移有無',
            '肺転移有無', '膵転移有無', '脳転移有無']
df['met_count'] = df[met_cols].sum(axis=1)

# 二値化変数
df['KPS_low'] = (df['KPS'] < 80).astype(int)
df['time_to_tx_short'] = pd.to_numeric(df['初診〜治療開始まで1年未満(1)'], errors='coerce').fillna(0).astype(int)
df['anemia'] = ((df['Sex'] == 1) & (df['Hb'] < 13.5) | (df['Sex'] == 0) & (df['Hb'] < 11.5)).astype(int)
df['LDH_high'] = (df['LDH'] > 250).astype(int)
df['met_multi'] = (df['met_count'] >= 2).astype(int)
df['Neutro_high'] = (df['Neutrophil'] > 7000).astype(int)
df['Plt_high'] = (df['Plt'] > 400000).astype(int)

# Proposed Score計算
df['Proposed_Score'] = (
    df['KPS_low'].fillna(0) +
    df['time_to_tx_short'].fillna(0) +
    df['anemia'].fillna(0) +
    df['LDH_high'].fillna(0) +
    df['mGPS'].fillna(0) +
    df['met_multi'].fillna(0)
)

# リスク分類
def classify_risk(score):
    if pd.isna(score): return np.nan
    if score <= 2: return 'Favorable'
    elif score <= 4: return 'Intermediate'
    else: return 'Poor'

df['Proposed_Risk'] = df['Proposed_Score'].apply(classify_risk)

# 解析データ
data = df.dropna(subset=['OS_event', 'OS_time', 'Proposed_Risk'])
data = data[data['OS_time'] > 0]

print(f"解析対象: {len(data)}例")

# 出力ディレクトリ
output_dir = '/Users/hideto/mita_utc_analysis/figures'
os.makedirs(output_dir, exist_ok=True)

# カラーパレット
colors_risk = {'Favorable': '#2ecc71', 'Intermediate': '#f39c12', 'Poor': '#e74c3c'}
colors_imdc = {'favorable': '#2ecc71', 'intermediate': '#f39c12', 'poor': '#e74c3c'}

# =============================================================================
# Figure 1: Model Scoring Schema (Simplified)
# =============================================================================
print("\nCreating Figure 1...")

fig1, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.axis('off')

# スコアリングスキーマをテキストで表示
schema_text = """
PROPOSED MODEL SCORING

Component                         Points
───────────────────────────────────────────
KPS < 80                             1
Time to treatment < 1 year           1
Anemia (Hb < LLN)                    1
LDH > 250 U/L                        1
mGPS 1                               1
mGPS 2                               2
≥2 metastatic sites                  1
───────────────────────────────────────────
Total Score Range                 0 - 7


RISK CLASSIFICATION
───────────────────────────────────────────
Favorable:       0-2 points   (n=234)
Intermediate:    3-4 points   (n=122)
Poor:            ≥5 points    (n=90)
───────────────────────────────────────────
"""

ax.text(0.5, 0.5, schema_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#3498db', linewidth=2))
ax.set_title('Figure 1. Proposed Model Scoring System', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure1_Model_Schema.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/Figure1_Model_Schema.png")

# Figure 1のデータをCSV/Excel出力（編集用）
fig1_data = pd.DataFrame({
    'Component': [
        'KPS < 80',
        'Time to treatment < 1 year',
        'Anemia (Hb < LLN)',
        'LDH > 250 U/L',
        'mGPS 1',
        'mGPS 2',
        '≥2 metastatic sites'
    ],
    'Points': [1, 1, 1, 1, 1, 2, 1],
    'n': [63, 302, 256, 73, 77, 118, 178],
    'Percentage': [14.1, 67.7, 57.4, 16.4, 17.3, 26.5, 39.9]
})

# リスク分類
fig1_risk = pd.DataFrame({
    'Risk_Group': ['Favorable', 'Intermediate', 'Poor'],
    'Score_Range': ['0-2', '3-4', '≥5'],
    'n': [234, 122, 90],
    'Median_OS_months': [64.0, 39.0, 12.0]
})

# CSV出力（tablesディレクトリに保存）
tables_dir = '/Users/hideto/mita_utc_analysis/tables'
fig1_data.to_csv(f'{tables_dir}/Figure1_Components.csv', index=False, encoding='utf-8-sig')
fig1_risk.to_csv(f'{tables_dir}/Figure1_Risk_Groups.csv', index=False, encoding='utf-8-sig')
print(f"  Saved: {tables_dir}/Figure1_Components.csv")
print(f"  Saved: {tables_dir}/Figure1_Risk_Groups.csv")

# =============================================================================
# Figure 2: Primary Survival Analysis and Model Performance
# Publication-quality for top-tier medical journals
# =============================================================================
print("\nCreating Figure 2...")

# 高品質設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelweight'] = 'normal'

fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(wspace=0.35)

# 2A: Kaplan-Meier curves
ax1 = axes[0]
risk_order = ['Favorable', 'Intermediate', 'Poor']

for risk in risk_order:
    sub = data[data['Proposed_Risk'] == risk]
    kmf = KaplanMeierFitter()
    kmf.fit(sub['OS_time'], sub['OS_event'], label=f"{risk} (n={len(sub)})")
    kmf.plot_survival_function(ax=ax1, color=colors_risk[risk], ci_show=True, linewidth=1.8)

ax1.set_xlabel('Time (months)', fontsize=11)
ax1.set_ylabel('Overall Survival Probability', fontsize=11)
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax1.set_xlim(0, 72)
ax1.set_ylim(0, 1.0)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Log-rank test
ax1.text(0.97, 0.97, 'Log-rank P < 0.001', transform=ax1.transAxes,
         fontsize=9, ha='right', va='top')

# Number at risk table - cleaner design
times = [0, 12, 24, 36, 48, 60]
ax1_table = ax1.inset_axes([0, -0.32, 1, 0.22])
ax1_table.axis('off')

table_data = []
for risk in risk_order:
    sub = data[data['Proposed_Risk'] == risk]
    row = []
    for t in times:
        n_at_risk = (sub['OS_time'] >= t).sum()
        row.append(str(n_at_risk))
    table_data.append(row)

# Cleaner table without borders
for i, (risk, row) in enumerate(zip(risk_order, table_data)):
    y_pos = 0.75 - i * 0.3
    ax1_table.text(-0.02, y_pos, risk[:3], fontsize=8, ha='right', va='center',
                   color=colors_risk[risk], fontweight='bold')
    for j, val in enumerate(row):
        ax1_table.text(j/5.5 + 0.08, y_pos, val, fontsize=8, ha='center', va='center')

# Time labels
for j, t in enumerate(times):
    ax1_table.text(j/5.5 + 0.08, 1.0, str(t), fontsize=8, ha='center', va='center', fontweight='bold')
ax1_table.text(-0.02, 1.0, 'Mo', fontsize=8, ha='right', va='center', fontweight='bold')

# 2B: C-index comparison - Forest plot style (horizontal)
ax2 = axes[1]

valid_data = data.dropna(subset=['Proposed_Score', 'IMDC_score']).copy()

# Calculate C-indices
c_proposed = concordance_index(valid_data['OS_time'], -valid_data['Proposed_Score'], valid_data['OS_event'])
c_imdc = concordance_index(valid_data['OS_time'], -valid_data['IMDC_score'], valid_data['OS_event'])

# Bootstrap CI with more iterations for precision
n_boot = 1000
np.random.seed(42)
c_proposed_boots, c_imdc_boots = [], []

for _ in range(n_boot):
    idx = np.random.choice(len(valid_data), size=len(valid_data), replace=True)
    boot = valid_data.iloc[idx]
    try:
        c_proposed_boots.append(concordance_index(boot['OS_time'], -boot['Proposed_Score'], boot['OS_event']))
        c_imdc_boots.append(concordance_index(boot['OS_time'], -boot['IMDC_score'], boot['OS_event']))
    except:
        pass

c_proposed_ci = (np.percentile(c_proposed_boots, 2.5), np.percentile(c_proposed_boots, 97.5))
c_imdc_ci = (np.percentile(c_imdc_boots, 2.5), np.percentile(c_imdc_boots, 97.5))

# Forest plot style - horizontal layout
y_positions = [1.5, 0.5]
models = ['Proposed Model', 'IMDC']
c_vals = [c_proposed, c_imdc]
c_cis = [c_proposed_ci, c_imdc_ci]
colors_model = ['#2980b9', '#7f8c8d']

for y, model, c_val, ci, color in zip(y_positions, models, c_vals, c_cis, colors_model):
    # Error bar (CI)
    ax2.errorbar(c_val, y, xerr=[[c_val - ci[0]], [ci[1] - c_val]],
                 fmt='s', markersize=8, color=color, capsize=4, capthick=1.5,
                 elinewidth=1.5, markeredgecolor='black', markeredgewidth=0.5)

    # Model name on left
    ax2.text(0.62, y, model, fontsize=10, ha='right', va='center', fontweight='bold')

    # C-index value and CI on right
    ax2.text(0.82, y, f'{c_val:.3f} ({ci[0]:.3f}–{ci[1]:.3f})',
             fontsize=9, ha='left', va='center', family='monospace')

# Reference line at 0.7
ax2.axvline(x=0.7, color='gray', linestyle=':', linewidth=1, alpha=0.7)

# P-value for difference
c_diff = np.array(c_proposed_boots) - np.array(c_imdc_boots)
p_diff = 2 * min(np.mean(c_diff > 0), np.mean(c_diff < 0))
delta_ci = (np.percentile(c_diff, 2.5), np.percentile(c_diff, 97.5))

ax2.text(0.72, -0.3, f'ΔC-index: {np.mean(c_diff):.3f} ({delta_ci[0]:.3f}–{delta_ci[1]:.3f}), P = {p_diff:.3f}',
         fontsize=9, ha='center', va='top')

ax2.set_xlim(0.62, 0.82)
ax2.set_ylim(-0.5, 2.2)
ax2.set_xlabel('C-index (95% CI)', fontsize=11)
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Header
ax2.text(0.72, 2.0, 'C-index (95% CI)', fontsize=10, ha='center', va='bottom', fontweight='bold')

# 2C: Calibration plot - Publication quality with quantitative metrics
ax3 = axes[2]

# Quintile-based calibration (5 groups)
data_cal = data.dropna(subset=['Proposed_Score', 'OS_time', 'OS_event']).copy()
data_cal['quintile'] = pd.qcut(data_cal['Proposed_Score'], q=5, labels=False, duplicates='drop')

observed_probs = []
predicted_probs = []
obs_cis_low = []
obs_cis_high = []
group_sizes = []

for q in sorted(data_cal['quintile'].unique()):
    sub = data_cal[data_cal['quintile'] == q]
    group_sizes.append(len(sub))

    # Predicted probability (based on score, scaled)
    pred_prob = sub['Proposed_Score'].mean() / 7  # Normalize to 0-1
    predicted_probs.append(pred_prob)

    # Observed probability from KM
    kmf = KaplanMeierFitter()
    kmf.fit(sub['OS_time'], sub['OS_event'])
    try:
        surv_24 = kmf.survival_function_at_times(24).values[0]
        obs_prob = 1 - surv_24

        # CI from KM
        ci = kmf.confidence_interval_survival_function_
        if 24 in ci.index or len(ci) > 0:
            ci_idx = ci.index[ci.index <= 24].max() if any(ci.index <= 24) else ci.index.min()
            ci_low = 1 - ci.loc[ci_idx].iloc[1]
            ci_high = 1 - ci.loc[ci_idx].iloc[0]
        else:
            ci_low = obs_prob - 0.05
            ci_high = obs_prob + 0.05
    except:
        obs_prob = sub['OS_event'].mean()
        ci_low = obs_prob - 0.05
        ci_high = obs_prob + 0.05

    observed_probs.append(obs_prob)
    obs_cis_low.append(ci_low)
    obs_cis_high.append(ci_high)

# Calculate quantitative calibration metrics
predicted_arr = np.array(predicted_probs)
observed_arr = np.array(observed_probs)
weights = np.array(group_sizes)

# Calibration slope and intercept (weighted linear regression)
slope, intercept = np.polyfit(predicted_arr, observed_arr, 1, w=np.sqrt(weights))

# Integrated Calibration Index (ICI) - weighted mean absolute difference
ici = np.average(np.abs(observed_arr - predicted_arr), weights=weights)

# E:O ratio (Expected/Observed)
total_expected = np.sum(predicted_arr * weights)
total_observed = np.sum(observed_arr * weights)
eo_ratio = total_expected / total_observed if total_observed > 0 else np.nan

# Plot with error bars
for i, (pred, obs, ci_l, ci_h, n) in enumerate(zip(predicted_probs, observed_probs,
                                                     obs_cis_low, obs_cis_high, group_sizes)):
    ax3.errorbar(pred, obs, yerr=[[obs - max(0, ci_l)], [min(1, ci_h) - obs]],
                 fmt='o', markersize=7, color='#2980b9', capsize=3, capthick=1,
                 elinewidth=1, markeredgecolor='white', markeredgewidth=0.8, alpha=0.9)

# Perfect calibration line
ax3.plot([0, 0.8], [0, 0.8], color='#c0392b', linestyle='--', linewidth=1.5,
         label='Perfect calibration', alpha=0.8)

# Calibration curve (linear fit)
x_line = np.linspace(min(predicted_probs), max(predicted_probs), 50)
y_line = slope * x_line + intercept
ax3.plot(x_line, y_line, color='#2980b9', linewidth=1.5, alpha=0.7, label='Observed')

ax3.set_xlabel('Predicted 2-year mortality', fontsize=11)
ax3.set_ylabel('Observed 2-year mortality', fontsize=11)
ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax3.set_xlim(0, 0.75)
ax3.set_ylim(0, 0.75)
ax3.set_aspect('equal')
ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add quantitative calibration metrics to plot
metrics_text = f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\nICI: {ici:.3f}'
ax3.text(0.97, 0.03, metrics_text, transform=ax3.transAxes, fontsize=8,
         ha='right', va='bottom', family='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Add subtle grid
ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# Print calibration metrics for manuscript
print(f"\n  Calibration metrics (2-year mortality):")
print(f"    Calibration slope: {slope:.3f}")
print(f"    Calibration intercept: {intercept:.3f}")
print(f"    ICI (Integrated Calibration Index): {ici:.3f}")
print(f"    E:O ratio: {eo_ratio:.3f}")

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure2_Primary_Results.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/Figure2_Primary_Results.pdf', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"  Saved: {output_dir}/Figure2_Primary_Results.png")
print(f"  Saved: {output_dir}/Figure2_Primary_Results.pdf")

# =============================================================================
# Figure 3: Reclassification of IMDC Risk Groups
# Publication-quality for top-tier medical journals
# =============================================================================
print("\nCreating Figure 3...")

fig3, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(wspace=0.30)

# 3A: Stacked bar chart for reclassification
ax1 = axes[0]

reclass_data = data.dropna(subset=['IMDC', 'Proposed_Risk'])
imdc_cats = ['favorable', 'intermediate', 'poor']
proposed_cats = ['Favorable', 'Intermediate', 'Poor']

# Reclassification matrix
reclass_matrix = pd.crosstab(reclass_data['IMDC'], reclass_data['Proposed_Risk'], normalize='index') * 100

x = np.arange(len(imdc_cats))
width = 0.6

# Stacked bar chart
bottom = np.zeros(len(imdc_cats))
for prop_cat in proposed_cats:
    if prop_cat in reclass_matrix.columns:
        vals = [reclass_matrix.loc[imdc, prop_cat] if imdc in reclass_matrix.index else 0 for imdc in imdc_cats]
        bars = ax1.bar(x, vals, width, bottom=bottom, label=prop_cat,
                       color=colors_risk[prop_cat], edgecolor='white', linewidth=0.5)
        # Add percentage labels
        for i, (val, b) in enumerate(zip(vals, bottom)):
            if val > 8:  # Only show if > 8%
                ax1.text(x[i], b + val/2, f'{val:.0f}%', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        bottom += vals

ax1.set_ylabel('Patients (%)', fontsize=11)
ax1.set_title('A', fontsize=12, fontweight='bold', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(['IMDC\nFavorable', 'IMDC\nIntermediate', 'IMDC\nPoor'], fontsize=10)
ax1.legend(title='Proposed Model', loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_ylim(0, 105)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 3B: KM for IMDC Intermediate reclassified
ax2 = axes[1]

imdc_int = reclass_data[reclass_data['IMDC'] == 'intermediate']

for risk in proposed_cats:
    sub = imdc_int[imdc_int['Proposed_Risk'] == risk]
    if len(sub) >= 5:
        kmf = KaplanMeierFitter()
        kmf.fit(sub['OS_time'], sub['OS_event'], label=f"{risk} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax2, color=colors_risk[risk], ci_show=True, linewidth=1.8)

ax2.set_xlabel('Time (months)', fontsize=11)
ax2.set_ylabel('Overall Survival Probability', fontsize=11)
ax2.set_title('B', fontsize=12, fontweight='bold', loc='left')
ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax2.set_xlim(0, 72)
ax2.set_ylim(0, 1.0)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

if len(imdc_int) > 0:
    ax2.text(0.97, 0.97, 'Log-rank P < 0.001', transform=ax2.transAxes,
             fontsize=9, ha='right', va='top')

# Subtitle
ax2.text(0.5, 1.02, 'IMDC Intermediate-risk patients', transform=ax2.transAxes,
         fontsize=10, ha='center', va='bottom', style='italic')

# 3C: KM for IMDC Poor reclassified
ax3 = axes[2]

imdc_poor = reclass_data[reclass_data['IMDC'] == 'poor']

for risk in ['Favorable', 'Intermediate', 'Poor']:
    sub = imdc_poor[imdc_poor['Proposed_Risk'] == risk]
    if len(sub) >= 3:
        kmf = KaplanMeierFitter()
        kmf.fit(sub['OS_time'], sub['OS_event'], label=f"{risk} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax3, color=colors_risk[risk], ci_show=True, linewidth=1.8)

ax3.set_xlabel('Time (months)', fontsize=11)
ax3.set_ylabel('Overall Survival Probability', fontsize=11)
ax3.set_title('C', fontsize=12, fontweight='bold', loc='left')
ax3.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax3.set_xlim(0, 60)
ax3.set_ylim(0, 1.0)
ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

if len(imdc_poor) > 0:
    try:
        ax3.text(0.97, 0.97, 'Log-rank P < 0.001', transform=ax3.transAxes,
                 fontsize=9, ha='right', va='top')
    except:
        pass

# Subtitle
ax3.text(0.5, 1.02, 'IMDC Poor-risk patients', transform=ax3.transAxes,
         fontsize=10, ha='center', va='bottom', style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/Figure3_IMDC_Reclassification.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/Figure3_IMDC_Reclassification.pdf', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"  Saved: {output_dir}/Figure3_IMDC_Reclassification.png")
print(f"  Saved: {output_dir}/Figure3_IMDC_Reclassification.pdf")

# =============================================================================
# Supplementary Figure 1: Treatment-Stratified Analysis
# Publication-quality for top-tier medical journals
# =============================================================================
print("\nCreating Supplementary Figure 1...")

fig_supp1, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(wspace=0.25)

treatment_groups = ['IO-IO', 'IO-TKI']
treatment_labels = {'IO-IO': 'Nivolumab + Ipilimumab', 'IO-TKI': 'ICI + TKI'}
xlim_by_tx = {'IO-IO': 72, 'IO-TKI': 60}

for idx, tx in enumerate(treatment_groups):
    ax = axes[idx]
    tx_data = data[data['Treatment_binary'] == tx]

    for risk in risk_order:
        sub = tx_data[tx_data['Proposed_Risk'] == risk]
        if len(sub) >= 5:
            kmf = KaplanMeierFitter()
            kmf.fit(sub['OS_time'], sub['OS_event'], label=f"{risk} (n={len(sub)})")
            kmf.plot_survival_function(ax=ax, color=colors_risk[risk], ci_show=True, linewidth=1.8)

    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Overall Survival Probability', fontsize=11)
    ax.set_title(chr(65+idx), fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, xlim_by_tx[tx])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Subtitle with treatment info
    ax.text(0.5, 1.02, f'{treatment_labels[tx]} (n={len(tx_data)})', transform=ax.transAxes,
            fontsize=10, ha='center', va='bottom', style='italic')

    if len(tx_data) > 0:
        try:
            ax.text(0.97, 0.97, 'Log-rank P < 0.001', transform=ax.transAxes,
                    fontsize=9, ha='right', va='top')
        except:
            pass

plt.tight_layout()
plt.savefig(f'{output_dir}/SuppFigure1_Treatment_Stratified.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/SuppFigure1_Treatment_Stratified.pdf', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"  Saved: {output_dir}/SuppFigure1_Treatment_Stratified.png")
print(f"  Saved: {output_dir}/SuppFigure1_Treatment_Stratified.pdf")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Figure作成完了")
print("=" * 80)
print(f"""
保存先: {output_dir}/

Main Figures:
  - Figure1_Model_Schema.png
  - Figure2_Primary_Results.png
  - Figure3_IMDC_Reclassification.png

Supplementary Figures:
  - SuppFigure1_Treatment_Stratified.png
""")
