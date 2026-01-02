# Quick Start Guide

このテンプレートを使って新しいプロジェクトを始める方法を説明します。

---

## 3ステップで始める

### Step 1: テンプレートをコピー

```bash
# 新しいプロジェクト用にコピー
cp -r paper_project_template/ my_new_project/
cd my_new_project/

# Gitを初期化（既存のGit履歴を削除）
rm -rf .git
git init
git add .
git commit -m "Initial commit from template"
```

### Step 2: プロジェクト情報を設定

1. **README.md を編集**
   - プロジェクト名・概要を記入
   - 研究目的・仮説を記載

2. **CLAUDE.md を確認**（必要に応じてカスタマイズ）
   - プロジェクト固有のルールがあれば追加

### Step 3: 作業を開始

```bash
# Python環境を作成
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# パッケージをインストール
pip install -r requirements.txt

# データを配置
# data/raw/ に元データを配置（自動的にGit管理外）
```

---

## ディレクトリの使い方

```
my_project/
├── data/
│   ├── raw/          # 元データ（Git管理外・編集禁止）
│   └── processed/    # 加工済みデータ
├── scripts/          # Pythonスクリプト
├── tasks/            # タスクチェックリスト（Markdown）
├── tables/           # 論文用テーブル（CSV、LaTeX）
├── figures/          # 論文用図（300 dpi以上）
├── results/          # 解析結果・中間ファイル
└── manuscript/       # 原稿
```

### 各ディレクトリの役割

| ディレクトリ | 役割 | Git管理 |
|-------------|------|---------|
| `data/raw/` | 元データ（PHI含む可能性あり） | **除外** |
| `data/processed/` | 前処理済みデータ | 任意 |
| `scripts/` | 解析スクリプト | 必須 |
| `tasks/` | 進捗管理チェックリスト | 必須 |
| `tables/` | 論文用テーブル | 必須 |
| `figures/` | 論文用図 | 必須 |
| `results/` | 中間結果・ログ | 任意 |
| `manuscript/` | 原稿ファイル | 必須 |

---

## よく使うスクリプト

### データプロファイリング
```bash
python scripts/data_profiler.py \
  --input data/raw/patients.csv \
  --output results/data_profile.md \
  --correlation
```

### Table 1（患者背景表）の作成
```bash
python scripts/create_table1.py \
  --input data/processed/clean_data.csv \
  --group treatment \
  --output tables/table1.md
```

### 品質チェック
```bash
python scripts/check_quality.py --report results/quality_report.md
```

### Pythonスクリプト内での使用
```python
# 図のスタイル設定
from scripts.figure_style import set_paper_style, save_figure
set_paper_style()

# 可視化
from scripts.visualization_toolkit import plot_kaplan_meier, plot_learning_curve

# ML実験管理
from scripts.ml_experiment_tracker import ExperimentTracker
tracker = ExperimentTracker("my_experiment")
```

---

## プロジェクトタイプ別の始め方

### 臨床研究・統計解析の場合

1. `tasks/01_data_preparation.md` を開いてチェックリストを確認
2. データを `data/raw/` に配置
3. `data_profiler.py` でデータ品質を確認
4. 前処理後、`create_table1.py` でTable 1を作成
5. `tasks/02_analysis.md` に沿って解析を進める

### 機械学習・AI研究の場合

1. `tasks/01_data_preparation.md` でデータ準備
2. `ml_experiment_tracker.py` で実験を管理
3. `visualization_toolkit.py` で学習曲線・評価結果を可視化
4. 結果を `results/experiments/` に保存

---

## 不要なファイルの削除

プロジェクトタイプに応じて、不要なファイルを削除してください。

### 統計解析のみの場合（MLスクリプト不要）
```bash
rm scripts/ml_experiment_tracker.py
# visualization_toolkit.py のML関連関数は無視でOK
```

### 軽量な探索的解析の場合
```bash
# タスクファイルを削減
rm tasks/03_manuscript_draft.md
rm tasks/04_submission.md

# manuscriptディレクトリを削除
rm -r manuscript/
```

### LESSONS_LEARNED.md について
このファイルはテンプレートの改善記録です。
新しいプロジェクトでは削除するか、自分用のメモとして活用してください。

---

## トラブルシューティング

### パッケージのインストールエラー
```bash
# 仮想環境を再作成
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### lifelinesのインストールに失敗
```bash
# 個別にインストール
pip install lifelines
```

### 図の解像度が低い
`figure_style.py` の `save_figure()` を使用してください。
デフォルトで300 dpi以上で保存されます。

---

## 次のステップ

1. [EXAMPLES.md](./EXAMPLES.md) - 具体的な使用例
2. [tasks/](./tasks/) - タスクチェックリスト
3. [scripts/](./scripts/) - 各スクリプトのdocstring

質問があれば、各スクリプトの `--help` オプションを参照してください：
```bash
python scripts/data_profiler.py --help
```
