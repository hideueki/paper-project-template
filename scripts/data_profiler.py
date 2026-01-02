#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_profiler.py - データの基本情報を自動出力する汎用スクリプト

統計解析・AI研究の両方で使用可能なデータプロファイリングツール。
データ品質の評価に必要な情報を自動生成します。

機能:
    - データ読み込み（CSV、Excel、Parquet対応）
    - 基本統計量（平均、中央値、標準偏差、IQR）
    - 欠損値の割合と分布
    - データ型の確認
    - 外れ値の検出（IQR法、Z-score法）
    - 特徴量の相関行列（ヒートマップ出力）
    - 結果をMarkdown形式で出力

使用例:
    python scripts/data_profiler.py --input data/raw/patients.csv --output results/data_profile.md
    python scripts/data_profiler.py -i data/raw/data.parquet -o results/profile.md --correlation

著者: [Your Name]
作成日: [Date]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import numpy as np

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 再現性のためのシード固定
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    データファイルを読み込む（CSV、Excel、Parquet対応）

    Parameters
    ----------
    filepath : str
        入力ファイルパス
    encoding : str
        文字エンコーディング（CSVのみ）

    Returns
    -------
    pd.DataFrame
        読み込まれたデータフレーム
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

    logger.info(f"データを読み込み中: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(filepath, encoding=encoding)
    elif suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif suffix == '.json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"サポートされていないファイル形式: {suffix}")

    logger.info(f"読み込み完了: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df


def get_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """データの基本情報を取得"""
    info = {
        'n_rows': df.shape[0],
        'n_cols': df.shape[1],
        'n_duplicates': df.duplicated().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': df.columns.tolist(),
    }

    dtype_counts = df.dtypes.value_counts()
    info['dtype_summary'] = {str(k): int(v) for k, v in dtype_counts.items()}

    return info


def classify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """連続変数とカテゴリ変数を分類"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # ユニーク値が少ない数値型はカテゴリとして扱う
    for col in numeric_cols.copy():
        if df[col].nunique() <= 10:
            categorical_cols.append(col)
            numeric_cols.remove(col)

    return numeric_cols, categorical_cols


def calculate_statistics(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """連続変数の記述統計量を計算（median [IQR] 形式）"""
    stats_list = []

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1

        stats = {
            'Variable': col,
            'N': len(data),
            'Missing': df[col].isna().sum(),
            'Missing%': f"{df[col].isna().mean() * 100:.1f}%",
            'Mean': f"{data.mean():.2f}",
            'SD': f"{data.std():.2f}",
            'Median': f"{median:.2f}",
            'IQR': f"{q1:.2f}-{q3:.2f}",
            'Min': f"{data.min():.2f}",
            'Max': f"{data.max():.2f}",
            'Median [IQR]': f"{median:.1f} [{q1:.1f}-{q3:.1f}]",
        }
        stats_list.append(stats)

    return pd.DataFrame(stats_list)


def calculate_categorical_stats(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """カテゴリ変数の度数分布を計算"""
    cat_stats = {}

    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100

        cat_df = pd.DataFrame({
            'Category': counts.index.astype(str),
            'N': counts.values,
            'Percentage': [f"{p:.1f}%" for p in percentages.values]
        })
        cat_stats[col] = cat_df

    return cat_stats


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値を分析"""
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100

    missing_df = pd.DataFrame({
        'Variable': df.columns,
        'Missing_Count': missing_count.values,
        'Missing_Percent': [f"{p:.2f}%" for p in missing_percent.values],
        'Missing_Percent_Numeric': missing_percent.values
    }).sort_values('Missing_Percent_Numeric', ascending=False).reset_index(drop=True)

    return missing_df


def detect_outliers(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    外れ値を検出する（IQR法またはZ-score法）

    Parameters
    ----------
    method : str
        'iqr' または 'zscore'
    threshold : float
        IQR法: 乗数（デフォルト1.5）、Z-score法: 閾値（デフォルト3.0）
    """
    outlier_info = []

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        if method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        else:  # zscore
            mean = data.mean()
            std = data.std()
            lower = mean - threshold * std
            upper = mean + threshold * std

        n_low = (data < lower).sum()
        n_high = (data > upper).sum()
        n_total = n_low + n_high

        outlier_info.append({
            'Variable': col,
            'Method': method.upper(),
            'Lower_Bound': f"{lower:.2f}",
            'Upper_Bound': f"{upper:.2f}",
            'Outliers_Low': n_low,
            'Outliers_High': n_high,
            'Outliers_Total': n_total,
            'Outliers%': f"{(n_total / len(data)) * 100:.2f}%"
        })

    return pd.DataFrame(outlier_info)


def calculate_correlation(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = 'pearson'
) -> pd.DataFrame:
    """相関行列を計算"""
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    return df[numeric_cols].corr(method=method)


def save_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """相関行列のヒートマップを保存"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # figure_styleがあれば使用
        try:
            from figure_style import set_paper_style, save_figure
            set_paper_style()
        except ImportError:
            pass

        fig, ax = plt.subplots(figsize=figsize)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True if len(corr_matrix) <= 15 else False,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Correlation Matrix')
        plt.tight_layout()

        heatmap_path = output_path.parent / 'correlation_heatmap.png'
        fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"相関ヒートマップを保存: {heatmap_path}")
        return str(heatmap_path)

    except ImportError:
        logger.warning("matplotlib/seaborn がインストールされていないため、ヒートマップは生成されません")
        return ""


def generate_markdown_report(
    filepath: str,
    basic_info: Dict[str, Any],
    numeric_stats: pd.DataFrame,
    categorical_stats: Dict[str, pd.DataFrame],
    missing_analysis: pd.DataFrame,
    outlier_analysis: pd.DataFrame,
    corr_matrix: Optional[pd.DataFrame],
    heatmap_path: str,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> str:
    """Markdown形式のレポートを生成"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Data Profile Report

**Generated:** {now}
**Input File:** `{filepath}`

---

## 1. Basic Information

| Item | Value |
|------|-------|
| Total Rows | {basic_info['n_rows']:,} |
| Total Columns | {basic_info['n_cols']} |
| Duplicate Rows | {basic_info['n_duplicates']:,} |
| Memory Usage | {basic_info['memory_mb']:.2f} MB |
| Numeric Variables | {len(numeric_cols)} |
| Categorical Variables | {len(categorical_cols)} |

### Data Types Summary

| Data Type | Count |
|-----------|-------|
"""
    for dtype, count in basic_info['dtype_summary'].items():
        report += f"| {dtype} | {count} |\n"

    # 連続変数
    report += f"""
---

## 2. Continuous Variables (N = {len(numeric_cols)})

> **Note:** 論文では median [IQR] 形式を使用

"""
    if not numeric_stats.empty:
        display_cols = ['Variable', 'N', 'Missing%', 'Median [IQR]', 'Min', 'Max']
        report += numeric_stats[display_cols].to_markdown(index=False) + "\n"
    else:
        report += "*No continuous variables found.*\n"

    # カテゴリ変数
    report += f"""
---

## 3. Categorical Variables (N = {len(categorical_cols)})

"""
    if categorical_stats:
        for col, cat_df in list(categorical_stats.items())[:10]:  # 最大10変数
            report += f"### {col}\n\n"
            report += cat_df.head(10).to_markdown(index=False) + "\n\n"
    else:
        report += "*No categorical variables found.*\n"

    # 欠損値
    report += """
---

## 4. Missing Values Analysis

"""
    missing_vars = missing_analysis[missing_analysis['Missing_Count'] > 0]

    if not missing_vars.empty:
        report += f"**Variables with missing:** {len(missing_vars)} / {basic_info['n_cols']}\n\n"
        report += missing_vars[['Variable', 'Missing_Count', 'Missing_Percent']].to_markdown(index=False) + "\n"
    else:
        report += "**No missing values detected.**\n"

    # 外れ値
    report += """
---

## 5. Outlier Detection

"""
    if not outlier_analysis.empty:
        outliers_present = outlier_analysis[outlier_analysis['Outliers_Total'] > 0]
        if not outliers_present.empty:
            report += outliers_present.to_markdown(index=False) + "\n"
        else:
            report += "**No outliers detected.**\n"
    else:
        report += "*No continuous variables to analyze.*\n"

    # 相関行列
    if corr_matrix is not None and not corr_matrix.empty:
        report += """
---

## 6. Correlation Analysis

"""
        if heatmap_path:
            report += f"![Correlation Heatmap]({Path(heatmap_path).name})\n\n"

        # 高相関ペアを抽出
        high_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    high_corr.append({
                        'Variable 1': corr_matrix.index[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': f"{corr_val:.3f}"
                    })

        if high_corr:
            report += "### High Correlations (|r| >= 0.7)\n\n"
            report += pd.DataFrame(high_corr).to_markdown(index=False) + "\n"

    # チェックリスト
    report += """
---

## 7. Data Quality Checklist

- [ ] Review duplicate rows
- [ ] Verify data types are correct
- [ ] Investigate high-missing variables (>20%)
- [ ] Review outliers for validity
- [ ] Check high correlations for multicollinearity
- [ ] Confirm categorical variable levels

---

## 8. Recommendations

### For Statistical Analysis
- Use median [IQR] for non-normal distributions
- Consider Multiple Imputation for missing data
- Check assumptions before parametric tests

### For Machine Learning
- Consider feature scaling for numeric variables
- Handle missing values (imputation or removal)
- Address multicollinearity if using linear models
- Consider feature selection for high-dimensional data

---

*Report generated by data_profiler.py*
"""
    return report


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='データの基本情報を自動出力する汎用スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/data_profiler.py --input data/raw/patients.csv --output results/data_profile.md
    python scripts/data_profiler.py -i data/raw/data.parquet -o results/profile.md --correlation
    python scripts/data_profiler.py -i data.csv -o profile.md --outlier-method zscore --threshold 3.0
        """
    )

    parser.add_argument('-i', '--input', required=True, help='入力ファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力ファイルパス（Markdown）')
    parser.add_argument('--encoding', default='utf-8', help='文字エンコーディング')
    parser.add_argument('--outlier-method', default='iqr', choices=['iqr', 'zscore'],
                        help='外れ値検出方法')
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='外れ値検出閾値（IQR: 1.5, Z-score: 3.0）')
    parser.add_argument('--correlation', action='store_true', help='相関行列を計算')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細ログ')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # データ読み込み
        df = load_data(args.input, encoding=args.encoding)

        # 分析実行
        logger.info("分析を実行中...")
        basic_info = get_basic_info(df)
        numeric_cols, categorical_cols = classify_columns(df)
        numeric_stats = calculate_statistics(df, numeric_cols)
        categorical_stats = calculate_categorical_stats(df, categorical_cols)
        missing_analysis = analyze_missing_values(df)
        outlier_analysis = detect_outliers(df, numeric_cols, args.outlier_method, args.threshold)

        # 相関行列（オプション）
        corr_matrix = None
        heatmap_path = ""
        if args.correlation:
            logger.info("相関行列を計算中...")
            corr_matrix = calculate_correlation(df, numeric_cols)
            if not corr_matrix.empty:
                output_path = Path(args.output)
                heatmap_path = save_correlation_heatmap(corr_matrix, output_path)

        # レポート生成
        logger.info("レポートを生成中...")
        report = generate_markdown_report(
            filepath=args.input,
            basic_info=basic_info,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            missing_analysis=missing_analysis,
            outlier_analysis=outlier_analysis,
            corr_matrix=corr_matrix,
            heatmap_path=heatmap_path,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        )

        # 保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"レポートを保存: {output_path}")

        # サマリー表示
        print("\n" + "=" * 50)
        print("Data Profile Summary")
        print("=" * 50)
        print(f"Rows: {basic_info['n_rows']:,}")
        print(f"Columns: {basic_info['n_cols']} (Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)})")
        print(f"Missing values: {df.isnull().sum().sum():,} total")
        print(f"Duplicate rows: {basic_info['n_duplicates']:,}")
        print(f"\nReport saved to: {output_path}")

    except Exception as e:
        logger.error(f"エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
