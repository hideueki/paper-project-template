#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_table1.py - 患者背景表（Table 1: Baseline Characteristics）自動生成

臨床研究・疫学研究用の記述統計表を自動生成するスクリプト。
CLAUDE.mdのルールに準拠（SMD使用、median [IQR]形式）。

機能:
    - 連続変数: median [IQR] または mean ± SD
    - カテゴリ変数: n (%)
    - グループ比較: t検定、Mann-Whitney U、カイ二乗検定
    - SMD（標準化平均差）の計算
    - 出力形式: Markdown、LaTeX、Excel

使用例:
    python scripts/create_table1.py --input data/processed/clean_data.csv --group treatment --output tables/table1.md
    python scripts/create_table1.py -i data.csv -g arm --format latex -o tables/table1.tex

著者: [Your Name]
作成日: [Date]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class VariableConfig:
    """変数の設定"""
    name: str
    label: str = ""
    var_type: str = "auto"  # auto, continuous, categorical
    nonnormal: bool = True  # Trueならmedian[IQR]、Falseならmean±SD


def calculate_smd(group1: pd.Series, group2: pd.Series, var_type: str = 'continuous') -> float:
    """
    標準化平均差（SMD）を計算

    Parameters
    ----------
    group1, group2 : pd.Series
        比較する2群のデータ
    var_type : str
        'continuous' または 'categorical'

    Returns
    -------
    float
        SMD値（絶対値）
    """
    g1 = group1.dropna()
    g2 = group2.dropna()

    if len(g1) == 0 or len(g2) == 0:
        return np.nan

    if var_type == 'continuous':
        mean1, mean2 = g1.mean(), g2.mean()
        var1, var2 = g1.var(), g2.var()
        n1, n2 = len(g1), len(g2)

        # プール標準偏差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        smd = (mean1 - mean2) / pooled_std
    else:
        # カテゴリ変数の場合（比率の差）
        p1, p2 = g1.mean(), g2.mean()
        pooled_p = (p1 + p2) / 2

        if pooled_p == 0 or pooled_p == 1:
            return 0.0

        smd = (p1 - p2) / np.sqrt(pooled_p * (1 - pooled_p))

    return abs(smd)


def format_continuous(
    data: pd.Series,
    nonnormal: bool = True,
    decimals: int = 1
) -> str:
    """連続変数をフォーマット"""
    data = data.dropna()

    if len(data) == 0:
        return "N/A"

    if nonnormal:
        median = data.median()
        q1, q3 = data.quantile([0.25, 0.75])
        return f"{median:.{decimals}f} [{q1:.{decimals}f}-{q3:.{decimals}f}]"
    else:
        mean = data.mean()
        std = data.std()
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_categorical(
    data: pd.Series,
    total_n: int,
    show_percent: bool = True
) -> Dict[str, str]:
    """カテゴリ変数をフォーマット"""
    counts = data.value_counts(dropna=False)
    result = {}

    for category, count in counts.items():
        cat_str = str(category) if pd.notna(category) else "Missing"

        if show_percent:
            percent = (count / total_n) * 100
            # n<200では整数、n>=200では小数点1桁
            if total_n < 200:
                result[cat_str] = f"{count} ({percent:.0f}%)"
            else:
                result[cat_str] = f"{count} ({percent:.1f}%)"
        else:
            result[cat_str] = str(count)

    return result


def test_continuous(group1: pd.Series, group2: pd.Series, nonnormal: bool = True) -> Tuple[str, float]:
    """連続変数の検定"""
    g1 = group1.dropna()
    g2 = group2.dropna()

    if len(g1) < 3 or len(g2) < 3:
        return "N/A", np.nan

    if nonnormal:
        stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        return "Mann-Whitney U", p
    else:
        stat, p = stats.ttest_ind(g1, g2)
        return "t-test", p


def test_categorical(data: pd.DataFrame, var: str, group: str) -> Tuple[str, float]:
    """カテゴリ変数の検定"""
    contingency = pd.crosstab(data[var], data[group])

    # 期待度数が5未満のセルがあればFisher正確検定
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    if (expected < 5).any():
        if contingency.shape == (2, 2):
            _, p = stats.fisher_exact(contingency)
            return "Fisher", p
        else:
            return "Chi-square*", p  # *: 期待度数<5あり
    else:
        return "Chi-square", p


def format_pvalue(p: float) -> str:
    """p値をフォーマット"""
    if pd.isna(p):
        return "N/A"
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.3f}"


def create_table1(
    df: pd.DataFrame,
    group_var: str,
    variables: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None,
    nonnormal_vars: Optional[List[str]] = None,
    show_pvalue: bool = False,
    show_smd: bool = True,
    show_test: bool = False,
    decimals: int = 1
) -> pd.DataFrame:
    """
    Table 1を作成

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    group_var : str
        グループ変数名
    variables : list, optional
        含める変数のリスト（Noneなら自動検出）
    categorical_vars : list, optional
        カテゴリ変数として扱う変数
    nonnormal_vars : list, optional
        非正規分布の連続変数（median[IQR]で表示）
    show_pvalue : bool
        p値を表示するか（CLAUDE.mdではSMD推奨）
    show_smd : bool
        SMDを表示するか
    show_test : bool
        検定方法を表示するか
    decimals : int
        小数点以下桁数

    Returns
    -------
    pd.DataFrame
        Table 1のデータフレーム
    """
    if variables is None:
        variables = [c for c in df.columns if c != group_var]

    if categorical_vars is None:
        categorical_vars = []
        for var in variables:
            if df[var].dtype == 'object' or df[var].nunique() <= 10:
                categorical_vars.append(var)

    if nonnormal_vars is None:
        nonnormal_vars = variables  # デフォルトは全てmedian[IQR]

    # グループの取得
    groups = df[group_var].dropna().unique()
    if len(groups) != 2:
        logger.warning(f"グループ数が2ではありません: {len(groups)}。SMD/p値は計算されません。")
        show_pvalue = False
        show_smd = False

    # 結果格納用
    rows = []

    # 各グループのN
    header_row = {'Variable': 'N'}
    for g in groups:
        n = (df[group_var] == g).sum()
        header_row[str(g)] = str(n)
    if show_smd:
        header_row['SMD'] = ''
    if show_pvalue:
        header_row['P-value'] = ''
    if show_test:
        header_row['Test'] = ''
    rows.append(header_row)

    # 各変数の処理
    for var in variables:
        if var == group_var:
            continue

        is_categorical = var in categorical_vars
        is_nonnormal = var in nonnormal_vars

        if is_categorical:
            # カテゴリ変数
            # 変数名の行
            var_row = {'Variable': f'**{var}**, n (%)'}
            for g in groups:
                var_row[str(g)] = ''

            # SMD/p値（最初のカテゴリで計算）
            if len(groups) == 2:
                # 二値化してSMD計算
                first_cat = df[var].value_counts().index[0]
                g1 = (df[df[group_var] == groups[0]][var] == first_cat).astype(int)
                g2 = (df[df[group_var] == groups[1]][var] == first_cat).astype(int)
                smd = calculate_smd(g1, g2, 'categorical')

                if show_smd:
                    var_row['SMD'] = f"{smd:.3f}" if not pd.isna(smd) else "N/A"

                if show_pvalue:
                    test_name, p = test_categorical(df.dropna(subset=[var, group_var]), var, group_var)
                    var_row['P-value'] = format_pvalue(p)
                    if show_test:
                        var_row['Test'] = test_name
            else:
                if show_smd:
                    var_row['SMD'] = ''
                if show_pvalue:
                    var_row['P-value'] = ''
                if show_test:
                    var_row['Test'] = ''

            rows.append(var_row)

            # 各カテゴリの行
            for g in groups:
                group_data = df[df[group_var] == g][var]
                formatted = format_categorical(group_data, len(group_data))

                for cat, val in formatted.items():
                    cat_row = {'Variable': f'  {cat}'}
                    # 各グループの値を設定
                    for g2 in groups:
                        if g2 == g:
                            g_data = df[df[group_var] == g2][var]
                            g_formatted = format_categorical(g_data, len(g_data))
                            cat_row[str(g2)] = g_formatted.get(cat, '0 (0%)')
                        else:
                            g_data = df[df[group_var] == g2][var]
                            g_formatted = format_categorical(g_data, len(g_data))
                            cat_row[str(g2)] = g_formatted.get(cat, '0 (0%)')

                    if show_smd:
                        cat_row['SMD'] = ''
                    if show_pvalue:
                        cat_row['P-value'] = ''
                    if show_test:
                        cat_row['Test'] = ''

                    # 重複を避ける
                    if not any(r.get('Variable') == cat_row['Variable'] for r in rows):
                        rows.append(cat_row)

        else:
            # 連続変数
            format_type = "median [IQR]" if is_nonnormal else "mean ± SD"
            var_row = {'Variable': f'{var}, {format_type}'}

            for g in groups:
                group_data = df[df[group_var] == g][var]
                var_row[str(g)] = format_continuous(group_data, is_nonnormal, decimals)

            # SMD/p値
            if len(groups) == 2:
                g1 = df[df[group_var] == groups[0]][var]
                g2 = df[df[group_var] == groups[1]][var]

                smd = calculate_smd(g1, g2, 'continuous')
                if show_smd:
                    var_row['SMD'] = f"{smd:.3f}" if not pd.isna(smd) else "N/A"

                if show_pvalue:
                    test_name, p = test_continuous(g1, g2, is_nonnormal)
                    var_row['P-value'] = format_pvalue(p)
                    if show_test:
                        var_row['Test'] = test_name
            else:
                if show_smd:
                    var_row['SMD'] = ''
                if show_pvalue:
                    var_row['P-value'] = ''
                if show_test:
                    var_row['Test'] = ''

            rows.append(var_row)

    return pd.DataFrame(rows)


def export_table(
    table: pd.DataFrame,
    output_path: Path,
    format: str = 'markdown'
) -> None:
    """テーブルをエクスポート"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'markdown':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Table 1. Baseline Characteristics\n\n")
            f.write(table.to_markdown(index=False))
            f.write("\n\n")
            f.write("> SMD: Standardized Mean Difference. SMD < 0.1 indicates good balance.\n")
            f.write("> Continuous variables: median [IQR]\n")
            f.write("> Categorical variables: n (%)\n")

    elif format == 'latex':
        latex_str = table.to_latex(index=False, escape=False)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("% Table 1. Baseline Characteristics\n")
            f.write(latex_str)

    elif format == 'excel':
        table.to_excel(output_path, index=False, sheet_name='Table1')

    elif format == 'csv':
        table.to_csv(output_path, index=False)

    logger.info(f"Table saved to: {output_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Table 1（患者背景表）を自動生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/create_table1.py --input data/processed/clean_data.csv --group treatment --output tables/table1.md
    python scripts/create_table1.py -i data.csv -g arm --format latex -o tables/table1.tex
    python scripts/create_table1.py -i data.csv -g group --show-pvalue --output tables/table1.md
        """
    )

    parser.add_argument('-i', '--input', required=True, help='入力ファイルパス')
    parser.add_argument('-g', '--group', required=True, help='グループ変数名')
    parser.add_argument('-o', '--output', required=True, help='出力ファイルパス')
    parser.add_argument('--format', default='markdown',
                        choices=['markdown', 'latex', 'excel', 'csv'],
                        help='出力形式')
    parser.add_argument('--variables', nargs='+', help='含める変数（スペース区切り）')
    parser.add_argument('--categorical', nargs='+', help='カテゴリ変数（スペース区切り）')
    parser.add_argument('--normal', nargs='+', help='正規分布の変数（mean±SDで表示）')
    parser.add_argument('--show-pvalue', action='store_true',
                        help='p値を表示（SMD推奨）')
    parser.add_argument('--no-smd', action='store_true', help='SMDを非表示')
    parser.add_argument('--show-test', action='store_true', help='検定方法を表示')
    parser.add_argument('--decimals', type=int, default=1, help='小数点以下桁数')
    parser.add_argument('--encoding', default='utf-8', help='文字エンコーディング')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細ログ')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # データ読み込み
        filepath = Path(args.input)
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, encoding=args.encoding)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, encoding=args.encoding)

        logger.info(f"データ読み込み: {len(df)} 行")

        # グループ変数の確認
        if args.group not in df.columns:
            raise ValueError(f"グループ変数 '{args.group}' が見つかりません")

        # 正規分布変数の反転（nonnormalのリストを作成）
        variables = args.variables if args.variables else None
        if variables is None:
            variables = [c for c in df.columns if c != args.group]

        if args.normal:
            nonnormal_vars = [v for v in variables if v not in args.normal]
        else:
            nonnormal_vars = variables  # 全てnonnormal

        # Table 1 作成
        table = create_table1(
            df=df,
            group_var=args.group,
            variables=args.variables,
            categorical_vars=args.categorical,
            nonnormal_vars=nonnormal_vars,
            show_pvalue=args.show_pvalue,
            show_smd=not args.no_smd,
            show_test=args.show_test,
            decimals=args.decimals
        )

        # エクスポート
        export_table(table, Path(args.output), args.format)

        print(f"\nTable 1 created successfully!")
        print(f"Output: {args.output}")
        print(f"Format: {args.format}")
        print(f"Groups: {df[args.group].unique().tolist()}")

    except Exception as e:
        logger.error(f"エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
