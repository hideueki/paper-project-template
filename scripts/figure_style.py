#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_style.py - 論文用図表の共通スタイル設定モジュール

このモジュールは、統計解析・AI研究の両方で使用できる
一貫したスタイルを提供します。

機能:
    - MatplotlibのrcParams設定
    - 色覚バリアフリー対応のカラーパレット
    - 論文用の高解像度保存関数
    - 統計解析用・AI研究用の両方に対応

使用例:
    from figure_style import set_paper_style, save_figure, COLORS

    set_paper_style()
    fig, ax = plt.subplots()
    ax.plot(x, y, color=COLORS['primary'])
    save_figure(fig, 'figures/my_figure.png')

著者: [Your Name]
作成日: [Date]
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging
import warnings

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# カラーパレット（色覚多様性対応）
# =============================================================================

# メインカラー（汎用）
COLORS = {
    'primary': '#3498db',        # 青（メイン）
    'secondary': '#e74c3c',      # 赤（対比）
    'tertiary': '#2ecc71',       # 緑（補助）
    'quaternary': '#9b59b6',     # 紫（追加）
    'accent': '#f39c12',         # オレンジ（アクセント）
}

# リスク群用カラー（生存解析等）
RISK_COLORS = {
    'favorable': '#2ecc71',      # 緑（良好群）
    'intermediate': '#f39c12',   # オレンジ（中間群）
    'poor': '#e74c3c',           # 赤（不良群）
    'very_poor': '#8e44ad',      # 紫（最不良群）
}

# 機械学習用カラー
ML_COLORS = {
    'train': '#3498db',          # 青（訓練）
    'validation': '#e74c3c',     # 赤（検証）
    'test': '#2ecc71',           # 緑（テスト）
    'baseline': '#7f8c8d',       # グレー（ベースライン）
}

# 色覚バリアフリーパレット（IBM Design Library準拠）
COLORBLIND_SAFE = [
    '#648FFF',  # 青
    '#DC267F',  # マゼンタ
    '#FE6100',  # オレンジ
    '#FFB000',  # 黄
    '#785EF0',  # 紫
]

# グレースケール（印刷用）
GRAYSCALE = ['#000000', '#4a4a4a', '#7a7a7a', '#a0a0a0', '#d0d0d0']

# カテゴリカルパレット（最大10色）
CATEGORICAL_PALETTE = [
    '#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
    '#1abc9c', '#e91e63', '#795548', '#607d8b', '#ff5722',
]


# =============================================================================
# デフォルト設定
# =============================================================================

DEFAULT_SETTINGS = {
    'figure_width': 8,
    'figure_height': 6,
    'dpi': 300,
    'font_size': 12,
    'title_size': 14,
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 10,
    'line_width': 1.5,
    'marker_size': 6,
}


# =============================================================================
# スタイル設定関数
# =============================================================================

def set_paper_style(
    style: str = 'default',
    font_family: str = 'sans-serif',
    use_tex: bool = False,
    context: str = 'paper',
    custom_settings: Optional[dict] = None
) -> None:
    """
    論文用のMatplotlibスタイルを設定する

    Parameters
    ----------
    style : str
        スタイルプリセット ('default', 'minimal', 'presentation')
    font_family : str
        フォントファミリー ('sans-serif', 'serif')
    use_tex : bool
        LaTeX レンダリングを使用するか
    context : str
        コンテキスト ('paper', 'poster', 'talk')
    custom_settings : dict, optional
        カスタム設定

    Examples
    --------
    >>> set_paper_style()
    >>> set_paper_style(style='presentation', context='talk')
    """
    settings = DEFAULT_SETTINGS.copy()

    # コンテキスト別スケーリング
    scale_factors = {'paper': 1.0, 'poster': 1.5, 'talk': 1.3}
    scale = scale_factors.get(context, 1.0)

    # スタイルプリセット
    if style == 'presentation':
        settings.update({
            'font_size': 14, 'title_size': 18, 'label_size': 14,
            'tick_size': 12, 'legend_size': 12, 'line_width': 2.0
        })
    elif style == 'minimal':
        settings.update({'line_width': 1.0, 'marker_size': 4})

    # カスタム設定をマージ
    if custom_settings:
        settings.update(custom_settings)

    # スケール適用
    for key in ['font_size', 'title_size', 'label_size', 'tick_size', 'legend_size']:
        settings[key] = int(settings[key] * scale)

    # rcParams設定
    plt.rcParams.update({
        'figure.figsize': (settings['figure_width'], settings['figure_height']),
        'figure.dpi': settings['dpi'],
        'figure.facecolor': 'white',
        'figure.autolayout': True,

        'font.family': font_family,
        'font.size': settings['font_size'],
        'text.usetex': use_tex,

        'axes.titlesize': settings['title_size'],
        'axes.labelsize': settings['label_size'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'axes.prop_cycle': plt.cycler('color', CATEGORICAL_PALETTE),

        'xtick.labelsize': settings['tick_size'],
        'ytick.labelsize': settings['tick_size'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        'legend.fontsize': settings['legend_size'],
        'legend.frameon': True,
        'legend.framealpha': 0.9,

        'lines.linewidth': settings['line_width'],
        'lines.markersize': settings['marker_size'],

        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        'savefig.dpi': settings['dpi'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })

    logger.info(f"Paper style applied: {style}, context: {context}")


def save_figure(
    fig: plt.Figure,
    filepath: Union[str, Path],
    formats: Optional[List[str]] = None,
    dpi: int = 300,
    transparent: bool = False
) -> List[str]:
    """
    図を高解像度で保存する

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        保存する図
    filepath : str or Path
        保存先パス
    formats : list, optional
        保存形式リスト（デフォルト: ['png', 'pdf']）
    dpi : int
        解像度
    transparent : bool
        背景を透明にするか

    Returns
    -------
    list
        保存されたファイルパスのリスト

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> saved = save_figure(fig, 'figures/plot')
    """
    if formats is None:
        formats = ['png', 'pdf']

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for fmt in formats:
        save_path = filepath.with_suffix(f'.{fmt}')

        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white' if not transparent else 'none',
            'transparent': transparent,
        }

        if fmt.lower() == 'tiff':
            save_kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}

        fig.savefig(save_path, format=fmt, **save_kwargs)
        saved_files.append(str(save_path))
        logger.info(f"Saved: {save_path} ({dpi} dpi)")

    return saved_files


def get_color_palette(n_colors: int, palette_type: str = 'categorical') -> List[str]:
    """
    カラーパレットを取得する

    Parameters
    ----------
    n_colors : int
        必要な色の数
    palette_type : str
        パレットタイプ ('categorical', 'risk', 'ml', 'colorblind', 'grayscale')

    Returns
    -------
    list
        カラーコードのリスト
    """
    palettes = {
        'categorical': CATEGORICAL_PALETTE,
        'risk': list(RISK_COLORS.values()),
        'ml': list(ML_COLORS.values()),
        'colorblind': COLORBLIND_SAFE,
        'grayscale': GRAYSCALE,
    }

    palette = palettes.get(palette_type, CATEGORICAL_PALETTE)

    if n_colors > len(palette):
        palette = (palette * ((n_colors // len(palette)) + 1))[:n_colors]

    return palette[:n_colors]


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    論文用の図を作成する

    Parameters
    ----------
    nrows, ncols : int
        行数、列数
    figsize : tuple, optional
        図のサイズ

    Returns
    -------
    tuple
        (Figure, Axes)
    """
    if figsize is None:
        width = DEFAULT_SETTINGS['figure_width'] * (0.6 if ncols > 1 else 1) * ncols
        height = DEFAULT_SETTINGS['figure_height'] * (0.6 if nrows > 1 else 1) * nrows
        figsize = (width, height)

    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)


def format_pvalue(p_value: float, threshold: float = 0.001) -> str:
    """p値をフォーマットする"""
    if p_value < threshold:
        return f'P < {threshold}'
    return f'P = {p_value:.3f}'


def add_grid(ax: plt.Axes, which: str = 'major', alpha: float = 0.3) -> None:
    """グリッドを追加する"""
    ax.grid(True, which=which, linestyle='--', alpha=alpha)
    ax.set_axisbelow(True)


# =============================================================================
# モジュールテスト
# =============================================================================

if __name__ == '__main__':
    import numpy as np

    print("Figure Style Module Test")
    print("=" * 50)

    set_paper_style()

    # テスト図の作成
    fig, axes = create_figure(nrows=1, ncols=2, figsize=(12, 5))

    # 左: 統計解析用（生存曲線風）
    ax1 = axes[0]
    x = np.linspace(0, 24, 100)
    colors = get_color_palette(3, 'risk')
    for i, (c, label) in enumerate(zip(colors, ['Low', 'Medium', 'High'])):
        y = np.exp(-0.05 * (i + 1) * x)
        ax1.plot(x, y, color=c, label=f'{label} Risk')
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Survival Probability')
    ax1.set_title('Survival Analysis Style')
    ax1.legend()
    add_grid(ax1)

    # 右: ML用（学習曲線風）
    ax2 = axes[1]
    epochs = np.arange(1, 51)
    train_loss = 1.0 * np.exp(-0.05 * epochs) + 0.1 + np.random.normal(0, 0.02, 50)
    val_loss = 1.0 * np.exp(-0.04 * epochs) + 0.15 + np.random.normal(0, 0.03, 50)
    ax2.plot(epochs, train_loss, color=ML_COLORS['train'], label='Train')
    ax2.plot(epochs, val_loss, color=ML_COLORS['validation'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Machine Learning Style')
    ax2.legend()
    add_grid(ax2)

    plt.tight_layout()
    saved = save_figure(fig, 'test_figure_style', formats=['png'])
    print(f"Test figures saved: {saved}")
    plt.close()

    print("Test completed!")
