#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization_toolkit.py - よく使う可視化をテンプレート化したモジュール

統計解析・AI研究の両方で使用可能な可視化ツールキット。
論文品質の図を簡単に作成できます。

機能:
    統計解析用:
        - Kaplan-Meier生存曲線
        - Forest plot
        - 散布図行列

    AI研究用:
        - 学習曲線（loss/accuracyの推移）
        - Confusion matrix
        - ROC曲線・PR曲線
        - Feature importance

    共通:
        - 分布プロット（ヒストグラム、箱ひげ図、バイオリンプロット）
        - 相関行列ヒートマップ

使用例:
    from visualization_toolkit import plot_learning_curve, plot_confusion_matrix
    plot_learning_curve(history, save_path="figures/learning_curve.png")

著者: [Your Name]
作成日: [Date]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# figure_styleからインポート（同じディレクトリにある場合）
try:
    from figure_style import (
        set_paper_style, save_figure, create_figure,
        get_color_palette, add_grid, RISK_COLORS, ML_COLORS,
        CATEGORICAL_PALETTE, format_pvalue
    )
except ImportError:
    # スタンドアロン使用時のフォールバック
    def set_paper_style():
        plt.style.use('seaborn-v0_8-whitegrid')

    def save_figure(fig, path, formats=None, dpi=300, **kwargs):
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return [str(path)]

    def create_figure(nrows=1, ncols=1, figsize=None, **kwargs):
        return plt.subplots(nrows, ncols, figsize=figsize or (8, 6), **kwargs)

    def get_color_palette(n, palette_type='categorical'):
        return plt.cm.tab10.colors[:n]

    def add_grid(ax, **kwargs):
        ax.grid(True, alpha=0.3)

    RISK_COLORS = {'favorable': '#2ecc71', 'intermediate': '#f39c12', 'poor': '#e74c3c'}
    ML_COLORS = {'train': '#3498db', 'validation': '#e74c3c', 'test': '#2ecc71'}
    CATEGORICAL_PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    def format_pvalue(p):
        return f'P < 0.001' if p < 0.001 else f'P = {p:.3f}'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# AI研究用: 学習曲線
# =============================================================================

def plot_learning_curve(
    history: Union[Dict[str, List[float]], Any],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    title: str = "Learning Curve"
) -> plt.Figure:
    """
    学習曲線をプロット

    Parameters
    ----------
    history : dict or keras.History
        学習履歴
    metrics : list, optional
        プロットする指標（Noneなら自動検出）
    figsize : tuple
        図のサイズ
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_paper_style()

    # Keras Historyの場合
    if hasattr(history, 'history'):
        history = history.history

    if metrics is None:
        # loss系とその他を分離
        loss_metrics = [k for k in history.keys() if 'loss' in k.lower()]
        other_metrics = [k for k in history.keys() if 'loss' not in k.lower()]
        metrics = loss_metrics + other_metrics[:2]

    n_plots = min(2, len(metrics))
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(list(history.values())[0]) + 1)

    # Loss plot
    ax = axes[0]
    for key in history.keys():
        if 'loss' in key.lower():
            is_val = 'val' in key.lower()
            color = ML_COLORS['validation'] if is_val else ML_COLORS['train']
            label = 'Validation' if is_val else 'Training'
            ax.plot(epochs, history[key], color=color, label=f'{label} Loss',
                    linestyle='--' if is_val else '-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    add_grid(ax)

    # Accuracy/other metrics plot
    if n_plots > 1:
        ax = axes[1]
        for key in history.keys():
            if 'loss' not in key.lower() and ('acc' in key.lower() or 'auc' in key.lower()):
                is_val = 'val' in key.lower()
                color = ML_COLORS['validation'] if is_val else ML_COLORS['train']
                label = 'Validation' if is_val else 'Training'
                ax.plot(epochs, history[key], color=color, label=label,
                        linestyle='--' if is_val else '-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric')
        ax.set_title('Performance')
        ax.legend()
        add_grid(ax)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# AI研究用: Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Confusion Matrixをプロット

    Parameters
    ----------
    y_true : array-like
        正解ラベル
    y_pred : array-like
        予測ラベル
    labels : list, optional
        クラスラベル
    normalize : bool
        正規化するか
    figsize : tuple
        図のサイズ
    cmap : str
        カラーマップ
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import confusion_matrix

    set_paper_style()

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='True Label',
           xlabel='Predicted Label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # セル内にテキスト
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# AI研究用: ROC曲線・PR曲線
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    ROC曲線をプロット

    Parameters
    ----------
    y_true : array-like
        正解ラベル
    y_prob : array-like
        予測確率
    figsize : tuple
        図のサイズ
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import roc_curve, auc

    set_paper_style()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color=CATEGORICAL_PALETTE[0], lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    add_grid(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve"
) -> plt.Figure:
    """
    Precision-Recall曲線をプロット
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    set_paper_style()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color=CATEGORICAL_PALETTE[0], lw=2,
            label=f'PR curve (AP = {ap:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    add_grid(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# AI研究用: Feature Importance
# =============================================================================

def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
) -> plt.Figure:
    """
    特徴量重要度をプロット

    Parameters
    ----------
    importance : array-like
        重要度スコア
    feature_names : list
        特徴量名
    top_n : int
        上位何件を表示
    figsize : tuple
        図のサイズ
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_paper_style()

    # 上位N件を選択
    indices = np.argsort(importance)[::-1][:top_n]
    top_importance = importance[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importance, color=CATEGORICAL_PALETTE[0])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# 統計解析用: Kaplan-Meier曲線
# =============================================================================

def plot_kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
    groups: Optional[np.ndarray] = None,
    group_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    show_ci: bool = True,
    show_at_risk: bool = True,
    time_label: str = "Time (months)",
    event_label: str = "Survival Probability",
    save_path: Optional[str] = None,
    title: str = "Kaplan-Meier Survival Curve"
) -> plt.Figure:
    """
    Kaplan-Meier生存曲線をプロット

    Parameters
    ----------
    time : array-like
        生存時間
    event : array-like
        イベント発生（1=イベント、0=打ち切り）
    groups : array-like, optional
        グループ変数
    group_labels : list, optional
        グループラベル
    figsize : tuple
        図のサイズ
    show_ci : bool
        95%信頼区間を表示
    show_at_risk : bool
        Number at riskを表示
    time_label : str
        X軸ラベル
    event_label : str
        Y軸ラベル
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logger.error("lifelinesがインストールされていません: pip install lifelines")
        raise

    set_paper_style()

    if show_at_risk:
        fig, (ax, ax_risk) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={'height_ratios': [4, 1]}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_risk = None

    kmf_list = []

    if groups is None:
        # 単一群
        kmf = KaplanMeierFitter()
        kmf.fit(time, event, label='Overall')
        kmf.plot_survival_function(ax=ax, ci_show=show_ci, color=CATEGORICAL_PALETTE[0])
        kmf_list.append(kmf)
    else:
        # 複数群
        unique_groups = np.unique(groups)
        colors = get_color_palette(len(unique_groups), 'risk')

        for i, g in enumerate(unique_groups):
            mask = groups == g
            label = group_labels[i] if group_labels else str(g)
            color = colors[i]

            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask], label=label)
            kmf.plot_survival_function(ax=ax, ci_show=show_ci, color=color)
            kmf_list.append(kmf)

        # Log-rank検定
        if len(unique_groups) == 2:
            mask1 = groups == unique_groups[0]
            mask2 = groups == unique_groups[1]
            result = logrank_test(
                time[mask1], time[mask2],
                event[mask1], event[mask2]
            )
            p_text = format_pvalue(result.p_value)
            ax.text(0.95, 0.95, f'Log-rank {p_text}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel(time_label)
    ax.set_ylabel(event_label)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    add_grid(ax)

    # Number at risk
    if show_at_risk and ax_risk is not None:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(*kmf_list, ax=ax_risk)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# 統計解析用: Forest Plot
# =============================================================================

def plot_forest(
    estimates: List[float],
    ci_lower: List[float],
    ci_upper: List[float],
    labels: List[str],
    null_value: float = 1.0,
    figsize: Tuple[int, int] = (10, 8),
    log_scale: bool = True,
    save_path: Optional[str] = None,
    title: str = "Forest Plot"
) -> plt.Figure:
    """
    Forest Plotをプロット（メタ分析・多変量解析用）

    Parameters
    ----------
    estimates : list
        効果量（HR、OR等）
    ci_lower : list
        信頼区間下限
    ci_upper : list
        信頼区間上限
    labels : list
        変数名
    null_value : float
        帰無値（通常1.0）
    figsize : tuple
        図のサイズ
    log_scale : bool
        対数スケールを使用
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    set_paper_style()

    n = len(estimates)
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(n)

    # 信頼区間
    for i in range(n):
        ax.plot([ci_lower[i], ci_upper[i]], [y_pos[i], y_pos[i]],
                color='black', linewidth=1.5)

    # 点推定
    ax.scatter(estimates, y_pos, s=100, color=CATEGORICAL_PALETTE[0], zorder=5)

    # 帰無線
    ax.axvline(x=null_value, color='gray', linestyle='--', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    if log_scale:
        ax.set_xscale('log')
        ax.set_xlabel('Hazard Ratio (log scale)')
    else:
        ax.set_xlabel('Estimate')

    ax.set_title(title)
    add_grid(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# 共通: 分布プロット
# =============================================================================

def plot_distribution(
    data: pd.DataFrame,
    column: str,
    group: Optional[str] = None,
    plot_type: str = 'histogram',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    分布プロット（ヒストグラム、箱ひげ図、バイオリンプロット）

    Parameters
    ----------
    data : pd.DataFrame
        データ
    column : str
        プロットする列名
    group : str, optional
        グループ変数
    plot_type : str
        'histogram', 'boxplot', 'violin'
    figsize : tuple
        図のサイズ
    save_path : str, optional
        保存先パス
    title : str, optional
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    import seaborn as sns

    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'histogram':
        if group:
            for i, g in enumerate(data[group].unique()):
                subset = data[data[group] == g][column]
                ax.hist(subset, alpha=0.6, label=str(g),
                        color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])
            ax.legend()
        else:
            ax.hist(data[column], color=CATEGORICAL_PALETTE[0], alpha=0.7)

    elif plot_type == 'boxplot':
        if group:
            sns.boxplot(data=data, x=group, y=column, ax=ax, palette=CATEGORICAL_PALETTE)
        else:
            sns.boxplot(data=data, y=column, ax=ax, color=CATEGORICAL_PALETTE[0])

    elif plot_type == 'violin':
        if group:
            sns.violinplot(data=data, x=group, y=column, ax=ax, palette=CATEGORICAL_PALETTE)
        else:
            sns.violinplot(data=data, y=column, ax=ax, color=CATEGORICAL_PALETTE[0])

    ax.set_title(title or f'Distribution of {column}')
    add_grid(ax)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# 共通: 相関行列ヒートマップ
# =============================================================================

def plot_correlation_heatmap(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    save_path: Optional[str] = None,
    title: str = "Correlation Matrix"
) -> plt.Figure:
    """
    相関行列のヒートマップをプロット

    Parameters
    ----------
    data : pd.DataFrame
        データ
    columns : list, optional
        対象列（Noneなら数値列全て）
    method : str
        相関係数の種類 ('pearson', 'spearman', 'kendall')
    figsize : tuple
        図のサイズ
    annot : bool
        値を表示
    save_path : str, optional
        保存先パス
    title : str
        タイトル

    Returns
    -------
    matplotlib.figure.Figure
    """
    import seaborn as sns

    set_paper_style()

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    corr = data[columns].corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot and len(columns) <= 15,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# テスト
# =============================================================================

if __name__ == '__main__':
    print("Visualization Toolkit - Demo")
    print("=" * 50)

    # ダミーデータ生成
    np.random.seed(42)
    n = 100

    # 学習曲線デモ
    history = {
        'loss': list(np.exp(-0.05 * np.arange(50)) + np.random.normal(0, 0.02, 50)),
        'val_loss': list(np.exp(-0.04 * np.arange(50)) + 0.05 + np.random.normal(0, 0.03, 50)),
        'accuracy': list(1 - np.exp(-0.05 * np.arange(50)) + np.random.normal(0, 0.02, 50)),
        'val_accuracy': list(1 - np.exp(-0.04 * np.arange(50)) - 0.05 + np.random.normal(0, 0.03, 50)),
    }

    fig = plot_learning_curve(history, title="Demo Learning Curve")
    plt.savefig('demo_learning_curve.png', dpi=150)
    plt.close()
    print("Created: demo_learning_curve.png")

    # Confusion Matrix デモ
    y_true = np.random.randint(0, 3, 100)
    y_pred = y_true.copy()
    y_pred[np.random.choice(100, 20, replace=False)] = np.random.randint(0, 3, 20)

    fig = plot_confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C'])
    plt.savefig('demo_confusion_matrix.png', dpi=150)
    plt.close()
    print("Created: demo_confusion_matrix.png")

    # ROC曲線デモ
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.clip(y_true + np.random.normal(0, 0.3, 100), 0, 1)

    fig = plot_roc_curve(y_true, y_prob)
    plt.savefig('demo_roc_curve.png', dpi=150)
    plt.close()
    print("Created: demo_roc_curve.png")

    print("\nDemo completed!")
