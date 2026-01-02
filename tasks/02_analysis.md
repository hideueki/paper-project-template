# 02. Statistical Analysis（統計解析）

## タスク概要

処理済みデータを用いて、記述統計・推論統計・生存解析を実施する。
論文に掲載する図表の元データを作成し、統計的妥当性を確保する。

**所要目安**: データ規模・解析複雑度により異なる
**担当者**: _______________
**開始日**: _______________
**完了日**: _______________

---

## 前提条件

### 必須チェックリスト
- [ ] [01_data_preparation.md](./01_data_preparation.md) 完了
- [ ] 処理済みデータが `data/processed/` に存在
- [ ] データ品質チェック完了
- [ ] 解析計画書（SAP）作成済み（該当する場合）

### 環境チェックリスト
- [ ] 必要パッケージインストール済み
  - [ ] pandas >= 2.0
  - [ ] numpy >= 1.24
  - [ ] scipy >= 1.10
  - [ ] lifelines >= 0.27（生存解析）
  - [ ] scikit-learn >= 1.2
  - [ ] tableone >= 0.8（Table 1作成）
  - [ ] statsmodels >= 0.14

---

## 使用データ

### 入力ファイル
| ファイル名 | パス | 説明 |
|-----------|------|------|
| 処理済みデータ | `data/processed/processed_data_YYYYMMDD.csv` | |
| データ辞書 | `data/data_dictionary.xlsx` | |

### データ確認
```
症例数: ___________
変数数: ___________
観察期間: ___________
イベント数: ___________
```

---

## 分析内容

### 1. 記述統計

> **注意**: 連続変数は **median (IQR)** を使用。mean ± SD は使用しない。

#### チェックリスト
- [ ] 連続変数: median (IQR) を算出
- [ ] カテゴリ変数: n (%) を算出
- [ ] パーセンテージ表記: n < 200 は整数、n >= 200 は小数点1桁
- [ ] 欠損値の報告

#### 記述統計サマリー
| 変数タイプ | 報告形式 | 例 |
|-----------|----------|-----|
| 連続変数 | median (IQR) | 65 (58-72) |
| カテゴリ変数 (n<200) | n (%) | 45 (38%) |
| カテゴリ変数 (n>=200) | n (%) | 145 (38.2%) |

#### コード例
```python
from tableone import TableOne

# 変数定義
columns = ['age', 'sex', 'bmi', 'stage', 'treatment']
categorical = ['sex', 'stage', 'treatment']
groupby = 'treatment_group'
nonnormal = ['age', 'bmi']  # 非正規分布変数

# Table 1 作成
table1 = TableOne(
    df,
    columns=columns,
    categorical=categorical,
    groupby=groupby,
    nonnormal=nonnormal,
    pval=False,  # P値は使用しない
    smd=True     # SMDを使用
)

print(table1.tabulate(tablefmt='github'))
```

---

### 2. グループ比較

> **注意**: Table 1ではP値ではなく **SMD（標準化平均差）** を使用。SMD < 0.1 をバランス良好の基準とする。

#### 検定選択フローチャート
```
連続変数
├── 2群比較
│   ├── 正規分布 → t検定（Welch）
│   └── 非正規分布 → Mann-Whitney U検定
└── 3群以上
    ├── 正規分布 → one-way ANOVA
    └── 非正規分布 → Kruskal-Wallis検定

カテゴリ変数
├── 2×2表 → カイ二乗検定 or Fisher正確検定
└── 2×k表 → カイ二乗検定
    └── 期待度数 < 5 のセルあり → Fisher正確検定
```

#### グループ比較記録表
| 変数名 | 検定法 | 検定統計量 | P値 | SMD | 備考 |
|--------|--------|-----------|-----|-----|------|
| | | | | | |
| | | | | | |

#### コード例
```python
from scipy import stats

# Mann-Whitney U検定
stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')

# カイ二乗検定
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# SMD計算
def calculate_smd(group1, group2):
    """標準化平均差を計算"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    smd = (group1.mean() - group2.mean()) / pooled_std
    return abs(smd)
```

---

### 3. 生存解析

> **注意**: HRだけでなく **Median OS**、**Events数** も必ず報告する。

#### 生存解析チェックリスト
- [ ] Time-to-event変数の定義確認
- [ ] イベントの定義確認
- [ ] 打ち切りの定義確認
- [ ] 追跡期間の記述（median follow-up）
- [ ] 比例ハザード性の確認（Schoenfeld残差）

#### 生存解析サマリー
| 群 | N | Events | Median OS (95% CI) | 1-yr OS (95% CI) | HR (95% CI) |
|----|---|--------|-------------------|------------------|-------------|
| 全体 | | | | | - |
| Group A | | | | | Reference |
| Group B | | | | | |

#### Kaplan-Meier解析
```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

kmf = KaplanMeierFitter()

# 全体の生存曲線
kmf.fit(df['time'], df['event'], label='Overall')
median_survival = kmf.median_survival_time_
ci = kmf.confidence_interval_survival_function_

# 群間比較（Log-rank検定）
results = logrank_test(
    durations_A=df[df['group']=='A']['time'],
    durations_B=df[df['group']=='B']['time'],
    event_observed_A=df[df['group']=='A']['event'],
    event_observed_B=df[df['group']=='B']['event']
)
print(f"Log-rank p-value: {results.p_value:.4f}")
```

#### Cox比例ハザードモデル
```python
from lifelines import CoxPHFitter

# 単変量解析
cph_uni = CoxPHFitter()
cph_uni.fit(df[['time', 'event', 'variable']], 'time', 'event')

# 多変量解析
cph_multi = CoxPHFitter()
cph_multi.fit(df[['time', 'event'] + covariates], 'time', 'event')

# 比例ハザード性の検定
cph_multi.check_assumptions(df, p_value_threshold=0.05)

# C-index
c_index = cph_multi.concordance_index_
print(f"C-index: {c_index:.3f}")
```

---

### 4. サブグループ解析

#### サブグループ定義
| サブグループ | 定義 | N | 事前指定 |
|-------------|------|---|---------|
| | | | Yes / No |
| | | | Yes / No |
| | | | Yes / No |

#### 交互作用検定
- [ ] 各サブグループと主要効果の交互作用を検定
- [ ] Forest plotで可視化

#### コード例
```python
# 交互作用項を含むCoxモデル
df['interaction'] = df['treatment'] * df['subgroup']
cph_interaction = CoxPHFitter()
cph_interaction.fit(
    df[['time', 'event', 'treatment', 'subgroup', 'interaction']],
    'time', 'event'
)
```

---

## 作成する図表リスト

### Table 1: 患者背景（Baseline Characteristics）

> **必須ルール**: P値ではなくSMDを使用

#### 構成要素チェックリスト
- [ ] 患者背景変数（年齢、性別、BMI等）
- [ ] 疾患特性（ステージ、グレード等）
- [ ] 治療関連（治療法、併用療法等）
- [ ] SMD列を追加（SMD < 0.1 で良好なバランス）
- [ ] 欠損値数を脚注に記載

#### 出力先
- `tables/table1_baseline.csv`

---

### Figure 1: CONSORTフローチャート

#### 構成要素チェックリスト
- [ ] スクリーニング症例数
- [ ] 除外症例数（理由別）
- [ ] 適格症例数
- [ ] 割付症例数（群別）
- [ ] 追跡脱落数（理由別）
- [ ] 解析対象症例数

#### 出力先
- `figures/figure1_consort.png`（300 dpi）

---

### Figure 2: Kaplan-Meier生存曲線

#### 必須要素チェックリスト
- [ ] 群別の生存曲線
- [ ] 95%信頼区間（シェード表示）
- [ ] Number at Risk テーブル
- [ ] Log-rank P値
- [ ] Median OS（各群）
- [ ] C-index（予後モデルの場合）

#### スタイル設定
```python
import matplotlib.pyplot as plt

# 推奨設定
plt.figure(figsize=(8, 6), dpi=300)

# カラーパレット（色覚多様性対応）
colors = {
    'Favorable': '#2ecc71',     # 緑
    'Intermediate': '#f39c12',  # オレンジ
    'Poor': '#e74c3c'           # 赤
}

# 軸ラベル
plt.xlabel('Time (months)', fontsize=12)
plt.ylabel('Overall Survival', fontsize=12)

# Number at Risk
from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(kmf_favorable, kmf_intermediate, kmf_poor, ax=ax)

# C-index表示（図中）
plt.text(0.95, 0.95, f'C-index: {c_index:.3f}',
         transform=ax.transAxes, ha='right', va='top')

plt.tight_layout()
plt.savefig('figures/figure2_km_curve.png', dpi=300, bbox_inches='tight')
```

#### 出力先
- `figures/figure2_km_curve.png`（300 dpi）

---

### Table 2: Cox回帰分析結果

> **必須ルール**: 単変量・多変量は別Tableに分離

#### 構成要素チェックリスト
- [ ] 変数名
- [ ] HR (95% CI)
- [ ] P値
- [ ] イベント数/総数
- [ ] 単変量・多変量を明確に区別

#### 変数選択の注意
> **禁止**: P値によるstepwise selection
> **推奨**: LASSO回帰、臨床的重要性に基づく選択

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# LASSO回帰による変数選択
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# 選択された変数
selected_vars = X.columns[lasso.coef_ != 0].tolist()
```

#### 出力先
- `tables/table2a_cox_univariate.csv`
- `tables/table2b_cox_multivariate.csv`

---

## 統計的検定の設定

### 基本設定
| 項目 | 設定値 |
|------|--------|
| 有意水準 | α = 0.05（両側検定） |
| 信頼区間 | 95% CI |
| 多重比較補正 | 下記参照 |

### 多重比較補正

#### 補正方法の選択
| 方法 | 適用場面 | 特徴 |
|------|----------|------|
| Bonferroni | 独立した検定、少数の比較 | 保守的 |
| Holm | 独立した検定、多数の比較 | Bonferroniより検出力高い |
| Benjamini-Hochberg (FDR) | 探索的解析、多数の比較 | False Discovery Rate制御 |

#### コード例
```python
from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.04, 0.03, 0.08, 0.001]

# Bonferroni補正
reject_bonf, p_adj_bonf, _, _ = multipletests(p_values, method='bonferroni')

# Benjamini-Hochberg (FDR)
reject_fdr, p_adj_fdr, _, _ = multipletests(p_values, method='fdr_bh')
```

### 多重比較補正記録
| 解析 | 比較数 | 補正方法 | 補正後α |
|------|--------|----------|---------|
| | | | |
| | | | |

---

## 再現性確保

### Random State / Seed 設定

> **必須**: すべての解析で random_state = 42 を固定

```python
import numpy as np
import random

# グローバル設定
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# sklearn
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, random_state=RANDOM_STATE)

# lifelines
from lifelines import CoxPHFitter
cph = CoxPHFitter(penalizer=0.1)
# lifelinesは内部でrandom_stateを使用しないが、
# ブートストラップ等で使用する場合は設定する
```

### 使用パッケージとバージョンの記録

#### チェックリスト
- [ ] `requirements.txt` を更新
- [ ] 解析に使用した全パッケージのバージョンを記録
- [ ] Python バージョンを記録

#### バージョン記録コマンド
```bash
# 現在の環境をエクスポート
pip freeze > requirements_frozen.txt

# 主要パッケージのバージョン確認
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import scipy; print(f'scipy: {scipy.__version__}')"
python -c "import lifelines; print(f'lifelines: {lifelines.__version__}')"
python -c "import sklearn; print(f'sklearn: {sklearn.__version__}')"
```

#### 環境情報記録表
| 項目 | バージョン |
|------|-----------|
| Python | |
| pandas | |
| numpy | |
| scipy | |
| lifelines | |
| scikit-learn | |
| statsmodels | |
| matplotlib | |
| seaborn | |

---

## 品質チェック項目

### 統計的妥当性チェック
- [ ] サンプルサイズが検定に十分か確認
- [ ] 検定の前提条件を確認（正規性、等分散性等）
- [ ] 比例ハザード性の仮定を検証（Cox回帰）
- [ ] 多重共線性を確認（VIF < 10）
- [ ] 外れ値・影響点の影響を確認

### 結果の妥当性チェック
- [ ] 効果の方向が臨床的に妥当か
- [ ] 効果の大きさが臨床的に意味があるか
- [ ] 信頼区間の幅が適切か
- [ ] 先行研究と比較して妥当か

### 報告基準準拠チェック
- [ ] STROBE（観察研究）準拠
- [ ] TRIPOD（予後モデル）準拠
- [ ] 該当するガイドラインのチェックリスト完了

### 再現性チェック
- [ ] 解析スクリプトを再実行して同じ結果が得られるか
- [ ] 中間結果のハッシュ値が一致するか
- [ ] 図表が再生成可能か

---

## 次のステップ

### 完了条件
- [ ] 全ての解析を完了
- [ ] 全ての図表を作成
- [ ] 品質チェック完了
- [ ] 結果を `results/` に保存
- [ ] スクリプトをコミット

### 次タスクへの引継ぎ
| 項目 | 内容 |
|------|------|
| 主要結果 | |
| 有意な変数 | |
| 特記事項 | |

### 次のタスク
→ [03. Visualization](./03_visualization.md)

---

## 変更履歴

| 日付 | 変更者 | 内容 |
|------|--------|------|
| | | 初版作成 |
