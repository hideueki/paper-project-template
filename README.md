# Paper Project Template

医学研究論文・機械学習プロジェクト用の再利用可能なテンプレート。

統計解析（生存解析等）とAI研究（CNN、機械学習等）の両方に対応しています。

---

## 特徴

- 統計解析・AI研究の両方に対応する汎用設計
- 論文投稿に必要な図表の品質管理（300 dpi以上）
- CLAUDE.mdによる統計規約の自動チェック
- 再現性を確保するためのシード固定・環境記録
- チェックリスト形式のタスク管理

---

## クイックスタート

```bash
# 1. テンプレートをコピー
cp -r paper_project_template/ my_project/
cd my_project/

# 2. Git初期化
rm -rf .git && git init

# 3. 環境構築
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. 作業開始
# - data/raw/ にデータを配置
# - tasks/01_data_preparation.md を確認
```

詳細は [QUICKSTART.md](./QUICKSTART.md) を参照してください。

---

## プロジェクト構造

```
paper_project_template/
│
├── README.md                 # このファイル
├── QUICKSTART.md             # 使い方ガイド
├── EXAMPLES.md               # 使用例
├── LESSONS_LEARNED.md        # 改善記録
├── CLAUDE.md                 # 統計規約・コーディング規約
├── requirements.txt          # 必要パッケージ
├── .gitignore                # Git除外設定
│
├── data/                     # データ
│   ├── raw/                  # 元データ（Git管理外）
│   └── processed/            # 加工済みデータ
│
├── scripts/                  # Pythonスクリプト
│   ├── figure_style.py       # 図のスタイル設定
│   ├── data_profiler.py      # データプロファイリング
│   ├── create_table1.py      # Table 1自動生成
│   ├── ml_experiment_tracker.py  # ML実験管理
│   ├── visualization_toolkit.py  # 可視化ツール
│   └── check_quality.py      # 品質チェック
│
├── tasks/                    # タスクチェックリスト
│   ├── 01_data_preparation.md
│   ├── 02_analysis.md
│   ├── 03_manuscript_draft.md
│   └── 04_submission.md
│
├── tables/                   # 論文用テーブル
├── figures/                  # 論文用図（300 dpi以上）
├── results/                  # 解析結果・中間ファイル
│   └── experiments/          # ML実験結果
│
└── manuscript/               # 原稿
    ├── submitted/            # 投稿版
    └── revision/             # 改訂版
```

---

## ディレクトリ詳細

| ディレクトリ | 役割 | Git管理 |
|-------------|------|---------|
| `data/raw/` | 元データ（PHI保護のためGit管理外） | 除外 |
| `data/processed/` | 前処理済みデータ | 任意 |
| `scripts/` | 解析・可視化スクリプト | 必須 |
| `tasks/` | 進捗管理チェックリスト | 必須 |
| `tables/` | 論文用テーブル（CSV, LaTeX） | 必須 |
| `figures/` | 論文用図（PNG, PDF, TIFF） | 必須 |
| `results/` | 中間結果・ログ | 任意 |
| `manuscript/` | 原稿ファイル | 必須 |

---

## スクリプト一覧

### 全プロジェクト共通

| スクリプト | 機能 | 使用例 |
|-----------|------|--------|
| `figure_style.py` | 図のスタイル設定 | `from figure_style import set_paper_style` |
| `data_profiler.py` | データ品質分析 | `python data_profiler.py -i data.csv -o profile.md` |
| `visualization_toolkit.py` | 可視化テンプレート | `from visualization_toolkit import plot_roc_curve` |
| `check_quality.py` | 品質チェック | `python check_quality.py --report report.md` |

### 統計解析用

| スクリプト | 機能 | 使用例 |
|-----------|------|--------|
| `create_table1.py` | Table 1自動生成 | `python create_table1.py -i data.csv -g group -o table1.md` |

### AI研究用

| スクリプト | 機能 | 使用例 |
|-----------|------|--------|
| `ml_experiment_tracker.py` | 実験管理 | `from ml_experiment_tracker import ExperimentTracker` |

---

## ワークフロー

### 臨床研究の場合

```
01_data_preparation.md  →  02_analysis.md  →  03_manuscript_draft.md  →  04_submission.md
        ↓                       ↓                       ↓
   data_profiler.py        create_table1.py        manuscript/
                          visualization_toolkit.py
```

### 機械学習の場合

```
01_data_preparation.md  →  02_analysis.md  →  03_manuscript_draft.md
        ↓                       ↓
   data_profiler.py     ml_experiment_tracker.py
                        visualization_toolkit.py
```

---

## 統計規約（CLAUDE.md準拠）

- 連続変数: **median [IQR]** を使用（mean ± SD は補助的に）
- Table 1: **SMD（標準化平均差）** を使用（P値ではなく）
- 変数選択: **LASSO** を使用（stepwise selection は禁止）
- 欠損値: **Multiple Imputation** 推奨（ダミー変数化・LOCF禁止）
- 図の解像度: **300 dpi以上**
- カラーパレット: 色覚多様性対応

---

## 報告ガイドライン

| 研究タイプ | ガイドライン |
|-----------|-------------|
| 観察研究 | STROBE |
| 予後モデル | TRIPOD |
| RCT | CONSORT |
| 診断精度研究 | STARD |
| システマティックレビュー | PRISMA |

---

## ライセンス

このテンプレートはMITライセンスで公開されています。
自由にカスタマイズしてご利用ください。

---

## 関連ドキュメント

- [QUICKSTART.md](./QUICKSTART.md) - 使い始めガイド
- [EXAMPLES.md](./EXAMPLES.md) - 具体的な使用例
- [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) - 改善記録
- [CLAUDE.md](./CLAUDE.md) - 統計規約・コーディング規約
