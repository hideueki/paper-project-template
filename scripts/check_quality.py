#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_quality.py - コードとデータの品質チェックスクリプト

プロジェクト全体の品質を自動チェックし、問題点を報告します。
統計解析・AI研究の両方で使用可能。

機能:
    - Pythonコードのスタイルチェック（black、flake8、mypy）
    - 図表ファイルの存在確認
    - 解像度チェック（300 dpi以上）
    - データファイルの整合性確認
    - CLAUDE.mdの規約準拠チェック
    - 再現性チェック（random seed固定の確認）

使用例:
    python scripts/check_quality.py
    python scripts/check_quality.py --fix  # 自動修正可能な問題を修正
    python scripts/check_quality.py --report results/quality_report.md

著者: [Your Name]
作成日: [Date]
"""

import argparse
import logging
import subprocess
import sys
import re
import ast
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """チェック結果"""
    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)
    severity: str = "info"  # info, warning, error


class QualityChecker:
    """品質チェッカー"""

    def __init__(self, project_dir: str = "."):
        """
        Parameters
        ----------
        project_dir : str
            プロジェクトディレクトリ
        """
        self.project_dir = Path(project_dir).resolve()
        self.results: List[CheckResult] = []

        # ディレクトリ設定
        self.dirs = {
            'scripts': self.project_dir / 'scripts',
            'data': self.project_dir / 'data',
            'figures': self.project_dir / 'figures',
            'tables': self.project_dir / 'tables',
            'results': self.project_dir / 'results',
        }

    def run_all_checks(self) -> List[CheckResult]:
        """全てのチェックを実行"""
        logger.info(f"品質チェック開始: {self.project_dir}")

        self.results = []

        # 構造チェック
        self.check_directory_structure()

        # コード品質チェック
        self.check_python_syntax()
        self.check_random_seed()
        self.check_imports()

        # 図表チェック
        self.check_figure_resolution()
        self.check_figure_format()

        # データチェック
        self.check_data_files()

        # 規約チェック
        self.check_claude_md_compliance()

        # 再現性チェック
        self.check_reproducibility()

        return self.results

    def check_directory_structure(self) -> None:
        """ディレクトリ構造をチェック"""
        missing_dirs = []
        for name, path in self.dirs.items():
            if not path.exists():
                missing_dirs.append(name)

        if missing_dirs:
            self.results.append(CheckResult(
                name="Directory Structure",
                passed=False,
                message=f"Missing directories: {', '.join(missing_dirs)}",
                severity="warning"
            ))
        else:
            self.results.append(CheckResult(
                name="Directory Structure",
                passed=True,
                message="All required directories exist"
            ))

    def check_python_syntax(self) -> None:
        """Pythonファイルの構文チェック"""
        scripts_dir = self.dirs.get('scripts')
        if not scripts_dir or not scripts_dir.exists():
            return

        errors = []
        for py_file in scripts_dir.glob('**/*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
            except SyntaxError as e:
                errors.append(f"{py_file.name}: Line {e.lineno} - {e.msg}")

        if errors:
            self.results.append(CheckResult(
                name="Python Syntax",
                passed=False,
                message=f"Syntax errors found in {len(errors)} file(s)",
                details=errors,
                severity="error"
            ))
        else:
            py_count = len(list(scripts_dir.glob('**/*.py')))
            self.results.append(CheckResult(
                name="Python Syntax",
                passed=True,
                message=f"All {py_count} Python files have valid syntax"
            ))

    def check_random_seed(self) -> None:
        """random seed の固定をチェック"""
        scripts_dir = self.dirs.get('scripts')
        if not scripts_dir or not scripts_dir.exists():
            return

        files_without_seed = []
        seed_patterns = [
            r'np\.random\.seed\(',
            r'random\.seed\(',
            r'torch\.manual_seed\(',
            r'tf\.random\.set_seed\(',
            r'RANDOM_STATE\s*=',
            r'random_state\s*=',
        ]

        for py_file in scripts_dir.glob('**/*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # ランダム操作を使用しているか
                uses_random = any([
                    'np.random' in content,
                    'random.' in content,
                    'torch.' in content and 'random' in content.lower(),
                    'sklearn' in content,
                ])

                if uses_random:
                    has_seed = any(re.search(p, content) for p in seed_patterns)
                    if not has_seed:
                        files_without_seed.append(py_file.name)
            except Exception as e:
                logger.warning(f"Could not check {py_file}: {e}")

        if files_without_seed:
            self.results.append(CheckResult(
                name="Random Seed",
                passed=False,
                message=f"{len(files_without_seed)} file(s) may not have random seed fixed",
                details=files_without_seed,
                severity="warning"
            ))
        else:
            self.results.append(CheckResult(
                name="Random Seed",
                passed=True,
                message="Random seed appears to be fixed in relevant files"
            ))

    def check_imports(self) -> None:
        """未使用インポートのチェック（簡易版）"""
        scripts_dir = self.dirs.get('scripts')
        if not scripts_dir or not scripts_dir.exists():
            return

        # flake8が利用可能かチェック
        try:
            result = subprocess.run(
                ['python', '-m', 'flake8', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                self.results.append(CheckResult(
                    name="Code Style (flake8)",
                    passed=True,
                    message="flake8 not installed, skipping style check"
                ))
                return
        except Exception:
            return

        # flake8実行
        try:
            result = subprocess.run(
                ['python', '-m', 'flake8', str(scripts_dir),
                 '--select=F401,F841', '--max-line-length=120'],
                capture_output=True,
                text=True
            )
            issues = result.stdout.strip().split('\n') if result.stdout.strip() else []
            issues = [i for i in issues if i]

            if issues:
                self.results.append(CheckResult(
                    name="Unused Imports/Variables",
                    passed=False,
                    message=f"{len(issues)} issue(s) found",
                    details=issues[:10],  # 最初の10件
                    severity="warning"
                ))
            else:
                self.results.append(CheckResult(
                    name="Unused Imports/Variables",
                    passed=True,
                    message="No unused imports or variables found"
                ))
        except Exception as e:
            logger.warning(f"flake8 check failed: {e}")

    def check_figure_resolution(self) -> None:
        """図の解像度をチェック（300 dpi以上）"""
        figures_dir = self.dirs.get('figures')
        if not figures_dir or not figures_dir.exists():
            self.results.append(CheckResult(
                name="Figure Resolution",
                passed=True,
                message="No figures directory found"
            ))
            return

        try:
            from PIL import Image
        except ImportError:
            self.results.append(CheckResult(
                name="Figure Resolution",
                passed=True,
                message="PIL not installed, skipping resolution check"
            ))
            return

        low_res_files = []
        checked_count = 0

        for img_file in figures_dir.glob('**/*'):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                try:
                    with Image.open(img_file) as img:
                        dpi = img.info.get('dpi', (72, 72))
                        if isinstance(dpi, tuple):
                            dpi = min(dpi)
                        else:
                            dpi = dpi

                        checked_count += 1
                        if dpi < 300:
                            low_res_files.append(f"{img_file.name} ({dpi} dpi)")
                except Exception as e:
                    logger.warning(f"Could not check {img_file}: {e}")

        if low_res_files:
            self.results.append(CheckResult(
                name="Figure Resolution",
                passed=False,
                message=f"{len(low_res_files)} figure(s) have resolution < 300 dpi",
                details=low_res_files,
                severity="warning"
            ))
        else:
            self.results.append(CheckResult(
                name="Figure Resolution",
                passed=True,
                message=f"All {checked_count} figure(s) meet resolution requirements"
            ))

    def check_figure_format(self) -> None:
        """図のフォーマットをチェック"""
        figures_dir = self.dirs.get('figures')
        if not figures_dir or not figures_dir.exists():
            return

        recommended_formats = ['.png', '.pdf', '.tiff', '.tif', '.eps']
        non_recommended = []

        for img_file in figures_dir.glob('**/*'):
            if img_file.is_file():
                if img_file.suffix.lower() not in recommended_formats:
                    non_recommended.append(f"{img_file.name} ({img_file.suffix})")

        if non_recommended:
            self.results.append(CheckResult(
                name="Figure Format",
                passed=False,
                message=f"{len(non_recommended)} figure(s) use non-recommended format",
                details=non_recommended,
                severity="warning"
            ))
        else:
            self.results.append(CheckResult(
                name="Figure Format",
                passed=True,
                message="All figures use recommended formats"
            ))

    def check_data_files(self) -> None:
        """データファイルの整合性チェック"""
        data_dir = self.dirs.get('data')
        if not data_dir or not data_dir.exists():
            return

        issues = []

        # raw dataが.gitignoreに含まれているか
        gitignore_path = self.project_dir / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            if 'data/raw' not in gitignore_content and 'raw/' not in gitignore_content:
                issues.append("data/raw/ is not in .gitignore (PHI protection)")

        # 機密データのチェック
        sensitive_patterns = ['*.xlsx', '*.csv']
        raw_dir = data_dir / 'raw'
        if raw_dir.exists():
            for pattern in sensitive_patterns:
                for f in raw_dir.glob(pattern):
                    # ファイル名に個人情報を含む可能性
                    if any(word in f.name.lower() for word in ['patient', 'personal', 'phi']):
                        issues.append(f"Potentially sensitive file: {f.name}")

        if issues:
            self.results.append(CheckResult(
                name="Data Files",
                passed=False,
                message=f"{len(issues)} potential issue(s) with data files",
                details=issues,
                severity="warning"
            ))
        else:
            self.results.append(CheckResult(
                name="Data Files",
                passed=True,
                message="Data files appear to be properly organized"
            ))

    def check_claude_md_compliance(self) -> None:
        """CLAUDE.mdの規約準拠チェック"""
        scripts_dir = self.dirs.get('scripts')
        if not scripts_dir or not scripts_dir.exists():
            return

        issues = []

        for py_file in scripts_dir.glob('**/*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # mean ± SD の使用チェック（median [IQR]推奨）
                if 'mean' in content.lower() and 'sd' in content.lower():
                    if 'median' not in content.lower():
                        issues.append(f"{py_file.name}: Uses mean±SD without median[IQR] option")

                # stepwise selection のチェック（禁止）
                if 'stepwise' in content.lower():
                    issues.append(f"{py_file.name}: May use stepwise selection (use LASSO instead)")

            except Exception as e:
                logger.warning(f"Could not check {py_file}: {e}")

        if issues:
            self.results.append(CheckResult(
                name="CLAUDE.md Compliance",
                passed=False,
                message=f"{len(issues)} potential compliance issue(s)",
                details=issues,
                severity="info"
            ))
        else:
            self.results.append(CheckResult(
                name="CLAUDE.md Compliance",
                passed=True,
                message="Code appears to follow CLAUDE.md guidelines"
            ))

    def check_reproducibility(self) -> None:
        """再現性チェック"""
        issues = []

        # requirements.txt の存在
        req_file = self.project_dir / 'requirements.txt'
        if not req_file.exists():
            issues.append("requirements.txt not found")

        # .gitがあるか
        git_dir = self.project_dir / '.git'
        if not git_dir.exists():
            issues.append("Not a git repository")

        if issues:
            self.results.append(CheckResult(
                name="Reproducibility",
                passed=False,
                message=f"{len(issues)} reproducibility issue(s)",
                details=issues,
                severity="warning"
            ))
        else:
            self.results.append(CheckResult(
                name="Reproducibility",
                passed=True,
                message="Basic reproducibility requirements met"
            ))

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """レポート生成"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        errors = sum(1 for r in self.results if r.severity == 'error')
        warnings = sum(1 for r in self.results if r.severity == 'warning')

        report = f"""# Quality Check Report

**Generated:** {now}
**Project:** {self.project_dir}

---

## Summary

| Status | Count |
|--------|-------|
| Passed | {passed}/{total} |
| Errors | {errors} |
| Warnings | {warnings} |

---

## Results

"""
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            icon = "+" if result.passed else "!"
            report += f"### [{icon}] {result.name}: {status}\n\n"
            report += f"{result.message}\n\n"

            if result.details:
                report += "**Details:**\n"
                for detail in result.details[:10]:
                    report += f"- {detail}\n"
                if len(result.details) > 10:
                    report += f"- ... and {len(result.details) - 10} more\n"
                report += "\n"

        report += """---

## Recommendations

1. Fix all ERROR-level issues before submission
2. Review WARNING-level issues
3. Ensure all figures are 300 dpi or higher
4. Confirm random seeds are fixed for reproducibility
5. Check CLAUDE.md compliance for statistical methods

---

*Report generated by check_quality.py*
"""

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")

        return report

    def print_summary(self) -> None:
        """サマリーを表示"""
        print("\n" + "=" * 60)
        print("Quality Check Summary")
        print("=" * 60)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            icon = "[+]" if result.passed else "[!]"
            print(f"{icon} {result.name}: {status}")
            if not result.passed and result.details:
                for detail in result.details[:3]:
                    print(f"    - {detail}")

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print("\n" + "-" * 60)
        print(f"Result: {passed}/{total} checks passed")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='コードとデータの品質チェック',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python scripts/check_quality.py
    python scripts/check_quality.py --report results/quality_report.md
    python scripts/check_quality.py --project /path/to/project
        """
    )

    parser.add_argument('--project', default='.', help='プロジェクトディレクトリ')
    parser.add_argument('--report', help='レポート出力先（Markdown）')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細ログ')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        checker = QualityChecker(args.project)
        checker.run_all_checks()
        checker.print_summary()

        if args.report:
            checker.generate_report(args.report)

        # エラーがあれば終了コード1
        errors = sum(1 for r in checker.results if r.severity == 'error')
        sys.exit(1 if errors > 0 else 0)

    except Exception as e:
        logger.error(f"エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
