# 01. Data Preparation（データ準備）

## タスク概要

臨床研究データの前処理を行い、解析可能な状態に整備する。
データの品質を確保し、再現性のある解析基盤を構築する。

**所要目安**: データ規模により異なる
**担当者**: _______________
**開始日**: _______________
**完了日**: _______________

---

## 前提条件

### 環境チェックリスト
- [ ] Python環境構築済み（requirements.txt参照）
- [ ] 必要パッケージインストール済み
  - [ ] pandas >= 2.0
  - [ ] numpy >= 1.24
  - [ ] scipy >= 1.10
- [ ] Git リポジトリ初期化済み
- [ ] .gitignore 設定済み（PHI保護）

### 倫理・法的要件
- [ ] IRB/倫理委員会承認取得済み
- [ ] データ使用許諾確認済み
- [ ] PHI（Protected Health Information）取り扱い規定確認済み
- [ ] データ保管場所のセキュリティ確認済み

---

## 入力データ

### データソース情報
| 項目 | 内容 |
|------|------|
| データ名 | |
| 取得元 | |
| 取得日 | |
| ファイル形式 | |
| 文字コード | UTF-8 / Shift-JIS / その他 |
| 対象期間 | YYYY/MM - YYYY/MM |
| 症例数（予定） | |

### ファイル配置チェックリスト
- [ ] 元データを `data/raw/` に配置
- [ ] ファイル名に日付を含める（例: `raw_data_20240101.xlsx`）
- [ ] データ辞書/コードブックを `data/` に配置
- [ ] `data/raw/` が .gitignore に含まれていることを確認

---

## 処理内容

### 1. データ読み込み

#### チェックリスト
- [ ] ファイルの読み込み成功
- [ ] 文字コード確認・変換
- [ ] 列名の確認・英語化
- [ ] データ型の確認

#### 確認項目
```
総行数: ___________
総列数: ___________
重複行数: ___________
```

#### コード例
```python
import pandas as pd

# データ読み込み
df = pd.read_excel('data/raw/raw_data.xlsx', encoding='utf-8')

# 基本情報確認
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.dtypes)
```

---

### 2. 欠損値処理

> **注意**: ダミー変数化（欠損を別カテゴリ化）は禁止。LOCFも使用しない。

#### 欠損値確認チェックリスト
- [ ] 各変数の欠損率を算出
- [ ] 欠損パターンの可視化（MCAR/MAR/MNAR判定）
- [ ] 欠損率 > 20% の変数をリストアップ
- [ ] 欠損理由の臨床的妥当性を検討

#### 欠損値記録表
| 変数名 | 欠損数 | 欠損率(%) | 欠損メカニズム | 対処方法 |
|--------|--------|-----------|----------------|----------|
| | | | MCAR/MAR/MNAR | |
| | | | MCAR/MAR/MNAR | |
| | | | MCAR/MAR/MNAR | |

#### 推奨対処法
| 方法 | 適用条件 |
|------|----------|
| Complete Case Analysis | 欠損率 < 5%、MCAR |
| Multiple Imputation | MAR、複数変数に欠損あり（**推奨**） |
| 変数除外 | 欠損率 > 40%、臨床的重要性低い |

#### コード例
```python
# 欠損値確認
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct
}).sort_values('missing_pct', ascending=False)

# Multiple Imputation（推奨）
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42, max_iter=10)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)
```

---

### 3. 外れ値処理

#### 外れ値検出チェックリスト
- [ ] 各連続変数の分布確認（ヒストグラム、箱ひげ図）
- [ ] IQR法またはZ-score法で外れ値検出
- [ ] 外れ値の臨床的妥当性を確認
- [ ] 生物学的にありえない値をリストアップ

#### 外れ値記録表
| 変数名 | 外れ値数 | 範囲 | 臨床的解釈 | 対処方法 |
|--------|----------|------|------------|----------|
| | | | | |
| | | | | |

#### 対処方法の選択肢
| 方法 | 適用条件 |
|------|----------|
| そのまま保持 | 臨床的に妥当な値 |
| 修正（確認後） | 入力ミスが明らか |
| Winsorization | 極端な値を端点に置換 |
| 除外 | 生物学的にありえない値 |

#### コード例
```python
import numpy as np

def detect_outliers_iqr(df, column):
    """IQR法による外れ値検出"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

# 生物学的妥当性チェック例
biological_limits = {
    'age': (0, 120),
    'bmi': (10, 60),
    'creatinine': (0.1, 20),
    'hemoglobin': (3, 25)
}
```

---

## 出力データ

### 出力ファイルチェックリスト
- [ ] 処理済みデータを `data/processed/` に保存
- [ ] ファイル名: `processed_data_YYYYMMDD.csv`
- [ ] 処理ログを `results/` に保存
- [ ] データ辞書を更新

### 出力データ仕様
```
最終行数: ___________
最終列数: ___________
除外症例数: ___________（除外理由別に記録）
```

### 除外症例の記録
| 除外理由 | 症例数 | 備考 |
|----------|--------|------|
| 重複データ | | |
| 必須項目欠損 | | |
| 適格基準外 | | |
| **合計** | | |

---

## 品質チェック項目

### データ整合性チェック
- [ ] 患者IDの一意性確認
- [ ] 日付の論理性（入院日 <= 退院日 など）
- [ ] カテゴリ変数の値が想定内
- [ ] 数値変数の範囲が臨床的に妥当
- [ ] 重複レコードなし

### 統計的チェック
- [ ] 連続変数の分布確認（正規性）
- [ ] カテゴリ変数の度数分布確認
- [ ] 変数間の相関確認
- [ ] 多重共線性チェック（VIF < 10）

### 再現性チェック
- [ ] random_state を固定（42推奨）
- [ ] 処理手順をスクリプト化
- [ ] 中間ファイルのハッシュ値記録

#### コード例
```python
import hashlib

def get_file_hash(filepath):
    """ファイルのMD5ハッシュを取得"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# 処理後データのハッシュ記録
hash_value = get_file_hash('data/processed/processed_data.csv')
print(f"Data hash: {hash_value}")
```

---

## 次のステップ

### 完了条件
- [ ] 全てのチェックリストを完了
- [ ] 処理済みデータを `data/processed/` に保存
- [ ] 処理内容をコミット

### 次タスクへの引継ぎ
| 項目 | 内容 |
|------|------|
| 最終症例数 | |
| 主要変数数 | |
| 特記事項 | |

### 次のタスク
→ [02. Exploratory Data Analysis](./02_exploratory_analysis.md)

---

## 変更履歴

| 日付 | 変更者 | 内容 |
|------|--------|------|
| | | 初版作成 |
