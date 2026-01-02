# 使用例

このドキュメントでは、テンプレートの具体的な使用方法を3つのシナリオで説明します。

---

## 例1: 臨床研究プロジェクト（生存解析）

### シナリオ
「がん患者の予後因子を解析し、リスクスコアを開発する」論文を執筆する。

### Step 1: プロジェクト初期化

```bash
# テンプレートをコピー
cp -r paper_project_template/ cancer_prognosis_study/
cd cancer_prognosis_study/

# Git初期化
rm -rf .git
git init
git add .
git commit -m "Initial commit: project setup"

# 環境構築
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: README.mdを編集

```markdown
# Cancer Prognosis Risk Score Study

## 研究目的
○○癌患者における予後予測リスクスコアを開発し、検証する。

## 研究デザイン
- 後ろ向きコホート研究
- 対象期間: 2015年1月 - 2022年12月
- 主要評価項目: 全生存期間（OS）
```

### Step 3: データ準備

```bash
# データを配置（PHI保護のためraw/はGit管理外）
cp /path/to/患者データ.xlsx data/raw/

# データプロファイリング
python scripts/data_profiler.py \
  --input data/raw/患者データ.xlsx \
  --output results/data_profile.md \
  --correlation

# tasks/01_data_preparation.md を開いてチェックリストを確認
```

### Step 4: 前処理スクリプト作成

```python
# scripts/01_preprocessing.py
import pandas as pd
import numpy as np

# 設定
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# データ読み込み
df = pd.read_excel('data/raw/患者データ.xlsx')

# 前処理
# - 欠損値処理（Multiple Imputation推奨）
# - 日付変換
# - 変数のリコーディング

# 保存
df.to_csv('data/processed/clean_data.csv', index=False)
```

### Step 5: Table 1作成

```bash
python scripts/create_table1.py \
  --input data/processed/clean_data.csv \
  --group risk_group \
  --output tables/table1.md \
  --format markdown
```

### Step 6: 生存解析

```python
# scripts/02_survival_analysis.py
from scripts.figure_style import set_paper_style, save_figure
from scripts.visualization_toolkit import plot_kaplan_meier
import pandas as pd

set_paper_style()

df = pd.read_csv('data/processed/clean_data.csv')

# Kaplan-Meier曲線
fig = plot_kaplan_meier(
    time=df['survival_time'].values,
    event=df['death'].values,
    groups=df['risk_group'].values,
    group_labels=['Low Risk', 'Intermediate Risk', 'High Risk'],
    save_path='figures/figure2_km_curve.png'
)
```

### Step 7: 論文執筆

- `tasks/03_manuscript_draft.md` のチェックリストに従って執筆
- 図表を `tables/` と `figures/` に整理
- 原稿を `manuscript/` に保存

### 使用するファイル一覧

| 段階 | 使用ファイル |
|------|-------------|
| データ準備 | `tasks/01_data_preparation.md`, `scripts/data_profiler.py` |
| 解析 | `tasks/02_analysis.md`, `scripts/create_table1.py`, `visualization_toolkit.py` |
| 執筆 | `tasks/03_manuscript_draft.md` |
| 投稿 | `tasks/04_submission.md` |

---

## 例2: 機械学習プロジェクト（画像分類）

### シナリオ
「医用画像からの疾患分類モデルを開発する」研究を行う。

### Step 1: プロジェクト初期化

```bash
cp -r paper_project_template/ image_classification/
cd image_classification/

rm -rf .git && git init

# 追加パッケージをインストール
echo "torch>=2.0" >> requirements.txt
echo "torchvision>=0.15" >> requirements.txt
pip install -r requirements.txt
```

### Step 2: 不要ファイルの削除（オプション）

```bash
# 臨床研究用のタスクファイルは使わないなら削除
# rm tasks/04_submission.md  # 論文投稿も行うなら残す
```

### Step 3: データ準備

```python
# scripts/01_prepare_dataset.py
import torch
from torchvision import datasets, transforms
import numpy as np

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# データ変換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# データセット分割（train/val/test）
# ...
```

### Step 4: 実験管理

```python
# scripts/02_train_model.py
from scripts.ml_experiment_tracker import ExperimentTracker
from scripts.figure_style import set_paper_style
from scripts.visualization_toolkit import plot_learning_curve, plot_confusion_matrix

# 実験トラッカー初期化
tracker = ExperimentTracker(
    project_name="image_classification",
    experiment_name="resnet50_baseline",
    description="ResNet50 baseline model"
)

# パラメータ記録
tracker.log_params({
    "model": "ResNet50",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "Adam",
    "pretrained": True
})

# モデル訓練
# history = train_model(...)

# 履歴記録
tracker.log_history({
    "loss": train_losses,
    "val_loss": val_losses,
    "accuracy": train_accs,
    "val_accuracy": val_accs
})

# 評価指標記録
tracker.calculate_classification_metrics(y_true, y_pred, y_prob, prefix="test_")

# 保存
tracker.save_experiment()
tracker.save_model(model, "best_model", framework="torch")
```

### Step 5: 可視化

```python
# scripts/03_visualize_results.py
from scripts.visualization_toolkit import (
    plot_learning_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve
)

set_paper_style()

# 学習曲線
plot_learning_curve(history, save_path="figures/figure1_learning_curve.png")

# Confusion Matrix
plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Disease'],
                      save_path="figures/figure2_confusion_matrix.png")

# ROC曲線
plot_roc_curve(y_true, y_prob, save_path="figures/figure3_roc_curve.png")
```

### Step 6: 実験比較

```python
from scripts.ml_experiment_tracker import compare_experiments, find_best_experiment

# 複数実験を比較
exp_dirs = [
    "results/experiments/image_classification/resnet50_baseline",
    "results/experiments/image_classification/resnet50_augmented",
    "results/experiments/image_classification/efficientnet_b0"
]

comparison = compare_experiments(exp_dirs, output_path="results/experiment_comparison.md")

# 最良モデルを検索
best = find_best_experiment(
    "results/experiments/image_classification",
    metric="test_accuracy",
    higher_is_better=True
)
```

### 使用するファイル一覧

| 段階 | 使用ファイル |
|------|-------------|
| データ準備 | `tasks/01_data_preparation.md`, `scripts/data_profiler.py` |
| モデル訓練 | `scripts/ml_experiment_tracker.py` |
| 可視化 | `scripts/visualization_toolkit.py`, `scripts/figure_style.py` |
| 品質確認 | `scripts/check_quality.py` |

---

## 例3: 軽量な探索的データ解析

### シナリオ
「簡単なデータ分析を行い、結果をチームに共有する」（論文執筆なし）

### Step 1: 最小構成で開始

```bash
cp -r paper_project_template/ quick_analysis/
cd quick_analysis/

# 不要ファイルを削除
rm -rf manuscript/
rm tasks/03_manuscript_draft.md
rm tasks/04_submission.md
rm LESSONS_LEARNED.md
rm EXAMPLES.md

# ML関連スクリプトも不要なら削除
rm scripts/ml_experiment_tracker.py
```

### Step 2: シンプルな構成

```
quick_analysis/
├── README.md              # 分析概要
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── figure_style.py
│   ├── data_profiler.py
│   └── analysis.py        # 分析スクリプト（新規作成）
├── tasks/
│   ├── 01_data_preparation.md
│   └── 02_analysis.md
├── figures/
└── results/
```

### Step 3: 分析実行

```bash
# データプロファイリング
python scripts/data_profiler.py \
  -i data/raw/data.csv \
  -o results/data_profile.md \
  --correlation

# 分析スクリプト実行
python scripts/analysis.py
```

### Step 4: 結果共有

```bash
# 結果をまとめてzip
zip -r analysis_results.zip results/ figures/ README.md

# またはGitHubで共有
git add .
git commit -m "Analysis complete"
git push
```

### 最小限の使用ファイル

| ファイル | 用途 |
|---------|------|
| `scripts/data_profiler.py` | データ品質確認 |
| `scripts/figure_style.py` | 図の作成 |
| `tasks/01_data_preparation.md` | チェックリスト |

---

## まとめ: プロジェクトタイプ別の推奨構成

| プロジェクトタイプ | 必須スクリプト | オプション |
|-------------------|---------------|-----------|
| 臨床研究（論文執筆） | 全て | - |
| 機械学習（論文執筆） | 全て | `create_table1.py` は任意 |
| 探索的分析 | `data_profiler.py`, `figure_style.py` | 他は削除可 |

---

## 補足: コマンドリファレンス

### データプロファイリング
```bash
python scripts/data_profiler.py -i INPUT -o OUTPUT [--correlation] [--outlier-method iqr|zscore]
```

### Table 1作成
```bash
python scripts/create_table1.py -i INPUT -g GROUP -o OUTPUT [--format markdown|latex|excel]
```

### 品質チェック
```bash
python scripts/check_quality.py [--project DIR] [--report OUTPUT]
```

各スクリプトの詳細は `--help` オプションで確認できます。
