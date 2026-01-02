#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_experiment_tracker.py - 機械学習実験の記録・管理モジュール

機械学習・深層学習プロジェクト用の実験管理ツール。
軽量でシンプルながら、再現性確保に必要な情報を記録します。

機能:
    - 実験設定（モデル、ハイパーパラメータ）の記録
    - 学習履歴（loss、accuracy等）の保存
    - 評価指標（AUC、F1-score、精度、再現率等）の自動算出
    - 結果をCSV・JSONで保存
    - 実験比較表の自動生成（Markdown形式）
    - モデルの重みとメタデータの保存

使用例:
    from ml_experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(project_name="image_classification")
    tracker.log_params({"model": "ResNet50", "lr": 0.001, "batch_size": 32})
    tracker.log_metrics({"val_accuracy": 0.95, "val_auc": 0.98})
    tracker.log_history(history)  # Kerasのhistoryオブジェクト等
    tracker.save_experiment()

著者: [Your Name]
作成日: [Date]
"""

import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import sys

import pandas as pd
import numpy as np

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


@dataclass
class ExperimentConfig:
    """実験設定"""
    experiment_id: str = ""
    project_name: str = ""
    experiment_name: str = ""
    description: str = ""
    created_at: str = ""
    random_seed: int = RANDOM_STATE
    tags: List[str] = field(default_factory=list)


@dataclass
class EnvironmentInfo:
    """実行環境情報"""
    python_version: str = ""
    platform: str = ""
    packages: Dict[str, str] = field(default_factory=dict)


class ExperimentTracker:
    """
    機械学習実験のトラッカー

    Attributes
    ----------
    project_name : str
        プロジェクト名
    experiment_name : str
        実験名
    base_dir : Path
        結果保存のベースディレクトリ

    Examples
    --------
    >>> tracker = ExperimentTracker("my_project")
    >>> tracker.log_params({"model": "CNN", "lr": 0.001})
    >>> tracker.log_metrics({"accuracy": 0.95})
    >>> tracker.save_experiment()
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: Optional[str] = None,
        base_dir: str = "results/experiments",
        description: str = "",
        tags: Optional[List[str]] = None
    ):
        """
        Parameters
        ----------
        project_name : str
            プロジェクト名
        experiment_name : str, optional
            実験名（Noneならタイムスタンプ）
        base_dir : str
            結果保存のベースディレクトリ
        description : str
            実験の説明
        tags : list, optional
            タグリスト
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir) / project_name / self.experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 実験ID生成
        self.experiment_id = self._generate_experiment_id()

        # 設定
        self.config = ExperimentConfig(
            experiment_id=self.experiment_id,
            project_name=project_name,
            experiment_name=self.experiment_name,
            description=description,
            created_at=datetime.now().isoformat(),
            random_seed=RANDOM_STATE,
            tags=tags or []
        )

        # 環境情報
        self.environment = self._get_environment_info()

        # データ格納
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = {}
        self.artifacts: List[str] = []
        self.notes: List[str] = []

        logger.info(f"Experiment initialized: {self.experiment_id}")
        logger.info(f"Results directory: {self.base_dir}")

    def _generate_experiment_id(self) -> str:
        """実験IDを生成"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{self.project_name}_{self.experiment_name}_{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{timestamp}_{short_hash}"

    def _get_environment_info(self) -> EnvironmentInfo:
        """実行環境情報を取得"""
        import platform

        packages = {}
        try:
            import pkg_resources
            for pkg in ['numpy', 'pandas', 'scikit-learn', 'torch', 'tensorflow', 'keras']:
                try:
                    packages[pkg] = pkg_resources.get_distribution(pkg).version
                except pkg_resources.DistributionNotFound:
                    pass
        except ImportError:
            pass

        return EnvironmentInfo(
            python_version=platform.python_version(),
            platform=platform.platform(),
            packages=packages
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        ハイパーパラメータを記録

        Parameters
        ----------
        params : dict
            パラメータの辞書

        Examples
        --------
        >>> tracker.log_params({
        ...     "model": "ResNet50",
        ...     "learning_rate": 0.001,
        ...     "batch_size": 32,
        ...     "epochs": 100
        ... })
        """
        self.params.update(params)
        logger.debug(f"Logged params: {list(params.keys())}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        評価指標を記録

        Parameters
        ----------
        metrics : dict
            指標の辞書

        Examples
        --------
        >>> tracker.log_metrics({
        ...     "val_accuracy": 0.95,
        ...     "val_auc": 0.98,
        ...     "val_f1": 0.94
        ... })
        """
        self.metrics.update(metrics)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")

    def log_history(
        self,
        history: Union[Dict[str, List[float]], Any],
        prefix: str = ""
    ) -> None:
        """
        学習履歴を記録

        Parameters
        ----------
        history : dict or keras.History
            学習履歴（Kerasのhistory.historyやdict）
        prefix : str
            キーのプレフィックス

        Examples
        --------
        >>> # Kerasの場合
        >>> history = model.fit(...)
        >>> tracker.log_history(history.history)

        >>> # PyTorchの場合
        >>> tracker.log_history({"loss": train_losses, "val_loss": val_losses})
        """
        # Keras Historyオブジェクトの場合
        if hasattr(history, 'history'):
            history = history.history

        for key, values in history.items():
            full_key = f"{prefix}{key}" if prefix else key
            self.history[full_key] = list(values)

        logger.debug(f"Logged history: {list(self.history.keys())}")

    def log_artifact(self, filepath: Union[str, Path], description: str = "") -> None:
        """
        アーティファクト（ファイル）を記録

        Parameters
        ----------
        filepath : str or Path
            ファイルパス
        description : str
            説明
        """
        filepath = Path(filepath)
        if filepath.exists():
            self.artifacts.append({
                'path': str(filepath),
                'name': filepath.name,
                'description': description,
                'size_bytes': filepath.stat().st_size
            })
            logger.debug(f"Logged artifact: {filepath.name}")
        else:
            logger.warning(f"Artifact not found: {filepath}")

    def add_note(self, note: str) -> None:
        """メモを追加"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.notes.append(f"[{timestamp}] {note}")

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        分類タスクの評価指標を計算

        Parameters
        ----------
        y_true : array-like
            正解ラベル
        y_pred : array-like
            予測ラベル
        y_prob : array-like, optional
            予測確率（AUC計算用）
        prefix : str
            指標名のプレフィックス

        Returns
        -------
        dict
            計算された指標
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        metrics = {
            f'{prefix}accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f'{prefix}f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # 二値分類
                    if y_prob.ndim == 2:
                        y_prob = y_prob[:, 1]
                    metrics[f'{prefix}auc'] = roc_auc_score(y_true, y_prob)
                else:
                    # 多クラス分類
                    metrics[f'{prefix}auc'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                logger.warning(f"AUC calculation failed: {e}")

        self.log_metrics(metrics)
        return metrics

    def save_experiment(self) -> Path:
        """
        実験結果を保存

        Returns
        -------
        Path
            保存先ディレクトリ
        """
        # メタデータ
        metadata = {
            'config': asdict(self.config),
            'environment': asdict(self.environment),
            'params': self.params,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'notes': self.notes
        }

        # JSON保存
        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # パラメータをCSV保存
        if self.params:
            params_df = pd.DataFrame([self.params])
            params_df.to_csv(self.base_dir / 'params.csv', index=False)

        # 指標をCSV保存
        if self.metrics:
            metrics_df = pd.DataFrame([self.metrics])
            metrics_df.to_csv(self.base_dir / 'metrics.csv', index=False)

        # 履歴をCSV保存
        if self.history:
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(self.base_dir / 'history.csv', index=False)

        # サマリーをMarkdown保存
        self._save_summary_markdown()

        logger.info(f"Experiment saved: {self.base_dir}")
        return self.base_dir

    def _save_summary_markdown(self) -> None:
        """サマリーをMarkdown形式で保存"""
        md_content = f"""# Experiment Summary

**Experiment ID:** {self.config.experiment_id}
**Project:** {self.config.project_name}
**Name:** {self.config.experiment_name}
**Created:** {self.config.created_at}
**Description:** {self.config.description or 'N/A'}
**Tags:** {', '.join(self.config.tags) if self.config.tags else 'N/A'}

---

## Environment

| Item | Value |
|------|-------|
| Python | {self.environment.python_version} |
| Platform | {self.environment.platform} |
| Random Seed | {self.config.random_seed} |

### Packages

| Package | Version |
|---------|---------|
"""
        for pkg, ver in self.environment.packages.items():
            md_content += f"| {pkg} | {ver} |\n"

        md_content += """
---

## Parameters

| Parameter | Value |
|-----------|-------|
"""
        for key, val in self.params.items():
            md_content += f"| {key} | {val} |\n"

        md_content += """
---

## Metrics

| Metric | Value |
|--------|-------|
"""
        for key, val in self.metrics.items():
            if isinstance(val, float):
                md_content += f"| {key} | {val:.4f} |\n"
            else:
                md_content += f"| {key} | {val} |\n"

        if self.notes:
            md_content += "\n---\n\n## Notes\n\n"
            for note in self.notes:
                md_content += f"- {note}\n"

        with open(self.base_dir / 'summary.md', 'w') as f:
            f.write(md_content)

    def save_model(
        self,
        model: Any,
        filename: str = "model",
        framework: str = "auto"
    ) -> Path:
        """
        モデルを保存

        Parameters
        ----------
        model : Any
            モデルオブジェクト
        filename : str
            ファイル名（拡張子なし）
        framework : str
            フレームワーク ('auto', 'keras', 'torch', 'sklearn', 'joblib')

        Returns
        -------
        Path
            保存先パス
        """
        model_dir = self.base_dir / 'models'
        model_dir.mkdir(exist_ok=True)

        if framework == 'auto':
            framework = self._detect_framework(model)

        if framework == 'keras':
            model_path = model_dir / f"{filename}.keras"
            model.save(model_path)
        elif framework == 'torch':
            import torch
            model_path = model_dir / f"{filename}.pt"
            torch.save(model.state_dict(), model_path)
        elif framework in ['sklearn', 'joblib']:
            import joblib
            model_path = model_dir / f"{filename}.joblib"
            joblib.dump(model, model_path)
        else:
            import pickle
            model_path = model_dir / f"{filename}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        self.log_artifact(model_path, f"Model saved with {framework}")
        logger.info(f"Model saved: {model_path}")
        return model_path

    def _detect_framework(self, model: Any) -> str:
        """モデルのフレームワークを自動検出"""
        model_type = str(type(model))

        if 'keras' in model_type or 'tensorflow' in model_type:
            return 'keras'
        elif 'torch' in model_type:
            return 'torch'
        elif 'sklearn' in model_type:
            return 'sklearn'
        else:
            return 'pickle'


def compare_experiments(
    experiment_dirs: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    複数の実験を比較

    Parameters
    ----------
    experiment_dirs : list
        実験ディレクトリのリスト
    output_path : str or Path, optional
        出力先（Markdown）

    Returns
    -------
    pd.DataFrame
        比較表
    """
    results = []

    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        metadata_path = exp_dir / 'metadata.json'

        if not metadata_path.exists():
            logger.warning(f"Metadata not found: {exp_dir}")
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        row = {
            'experiment_id': metadata['config']['experiment_id'],
            'experiment_name': metadata['config']['experiment_name'],
            'created_at': metadata['config']['created_at'],
        }

        # パラメータを追加
        for key, val in metadata.get('params', {}).items():
            row[f'param_{key}'] = val

        # 指標を追加
        for key, val in metadata.get('metrics', {}).items():
            row[f'metric_{key}'] = val

        results.append(row)

    comparison_df = pd.DataFrame(results)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Experiment Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))

        logger.info(f"Comparison saved: {output_path}")

    return comparison_df


def find_best_experiment(
    project_dir: Union[str, Path],
    metric: str,
    higher_is_better: bool = True
) -> Optional[Path]:
    """
    最良の実験を検索

    Parameters
    ----------
    project_dir : str or Path
        プロジェクトディレクトリ
    metric : str
        比較する指標名
    higher_is_better : bool
        大きいほど良いか

    Returns
    -------
    Path or None
        最良の実験のディレクトリ
    """
    project_dir = Path(project_dir)
    best_value = None
    best_dir = None

    for exp_dir in project_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        metadata_path = exp_dir / 'metadata.json'
        if not metadata_path.exists():
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        value = metadata.get('metrics', {}).get(metric)
        if value is None:
            continue

        if best_value is None:
            best_value = value
            best_dir = exp_dir
        elif higher_is_better and value > best_value:
            best_value = value
            best_dir = exp_dir
        elif not higher_is_better and value < best_value:
            best_value = value
            best_dir = exp_dir

    if best_dir:
        logger.info(f"Best experiment: {best_dir.name} ({metric}={best_value})")

    return best_dir


if __name__ == '__main__':
    print("ML Experiment Tracker - Demo")
    print("=" * 50)

    # デモ実験
    tracker = ExperimentTracker(
        project_name="demo_project",
        experiment_name="test_run",
        description="Demo experiment for testing"
    )

    # パラメータ記録
    tracker.log_params({
        "model": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    # ダミー履歴
    tracker.log_history({
        "loss": [0.5, 0.3, 0.2, 0.15, 0.1],
        "val_loss": [0.6, 0.4, 0.3, 0.25, 0.2],
        "accuracy": [0.7, 0.8, 0.85, 0.88, 0.9],
        "val_accuracy": [0.65, 0.75, 0.8, 0.82, 0.85]
    })

    # 指標記録
    tracker.log_metrics({
        "test_accuracy": 0.87,
        "test_f1": 0.85,
        "test_auc": 0.92
    })

    # メモ追加
    tracker.add_note("Initial experiment with default parameters")

    # 保存
    save_dir = tracker.save_experiment()

    print(f"\nExperiment saved to: {save_dir}")
    print("Files created:")
    for f in save_dir.iterdir():
        print(f"  - {f.name}")
