# Lessons Learned

このドキュメントは、テンプレート作成過程で得られた知見と、将来の改善に向けた提言をまとめています。

---

## うまくいった点

### 1. 汎用性の高い設計

**統計解析とAI研究の両立**
- 最初から両方のユースケースを想定して設計したため、幅広いプロジェクトに対応可能
- `figure_style.py` や `visualization_toolkit.py` は両方の分野で再利用できる
- 不要なスクリプトは簡単に削除できる構成

**教訓**: 初期設計段階で複数のユースケースを想定することが重要

### 2. チェックリスト形式のタスク管理

**Markdown形式の利点**
- Git管理しやすい
- 進捗が可視化される
- チーム共有が容易

**教訓**: 論文執筆は複雑なプロセスなので、段階的なチェックリストが有効

### 3. CLAUDE.mdによる規約の明文化

**統計規約の自動チェック**
- SMD使用、median [IQR]形式など、投稿前に確認すべき規約を明文化
- `check_quality.py` で一部自動チェック可能

**教訓**: 規約は「書くだけ」でなく「チェック可能」にすることが重要

### 4. 図表の品質管理

**300 dpi以上の徹底**
- `figure_style.py` でデフォルト300 dpi
- `check_quality.py` で解像度チェック

**教訓**: 投稿時に却下される一般的な理由（低解像度）を事前に防げる

---

## 改善が必要だった点

### 1. 依存関係の複雑さ

**問題**
- `lifelines` や `tableone` など、特定分野のパッケージが必須になっている
- インストール時にエラーが発生する可能性

**改善案**
- スクリプト内でのインポートを遅延読み込みにする
- パッケージがない場合の graceful degradation を実装
- 分野別の requirements ファイルを用意（`requirements-clinical.txt`, `requirements-ml.txt`）

### 2. 大規模データへの対応

**問題**
- 現在のスクリプトはメモリに収まるデータを想定
- 大規模画像データセットや遺伝子発現データには不十分

**改善案**
- チャンク処理の実装
- Daskなどの分散処理ライブラリの導入
- データパイプラインの追加（`scripts/data_pipeline.py`）

### 3. 多施設研究への対応

**問題**
- 単施設研究を想定した構成
- 複数データソースの統合機能がない

**改善案**
- `data/site_a/`, `data/site_b/` などの構成を追加
- データ統合スクリプトの作成
- サイト間のバッチ効果補正機能

### 4. 自動化の不足

**問題**
- 多くの作業が手動
- CIパイプラインが未整備

**改善案**
- GitHub Actions による自動テスト
- 図表の自動生成パイプライン
- 論文のPDF自動ビルド（LaTeX連携）

---

## 次回のプロジェクトへの提言

### 短期的な改善

1. **requirements.txt の分割**
   ```
   requirements-base.txt      # 必須パッケージのみ
   requirements-clinical.txt  # 臨床研究用
   requirements-ml.txt        # 機械学習用
   requirements-full.txt      # 全て（開発用）
   ```

2. **Makefileの追加**
   ```makefile
   setup:
       python -m venv .venv
       pip install -r requirements.txt

   profile:
       python scripts/data_profiler.py -i $(INPUT) -o results/profile.md

   check:
       python scripts/check_quality.py

   clean:
       rm -rf results/temp/ __pycache__/
   ```

3. **テストの追加**
   ```
   tests/
   ├── test_data_profiler.py
   ├── test_create_table1.py
   └── test_visualization.py
   ```

### 中長期的な改善

1. **Cookiecutterテンプレート化**
   - プロジェクト名などを対話的に設定
   - より柔軟な初期化

2. **Webインターフェース**
   - Streamlitを使った簡易GUI
   - 非プログラマーでも使いやすく

3. **クラウド対応**
   - Google Colab用ノートブック
   - AWS/GCP でのスケーラブルな実行

---

## テンプレート更新履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|---------|
| 1.0.0 | 2024-XX-XX | 初版リリース |

---

## 参考資料

### 報告ガイドライン
- [STROBE Statement](https://www.strobe-statement.org/)
- [TRIPOD Statement](https://www.tripod-statement.org/)
- [CONSORT Statement](http://www.consort-statement.org/)

### コーディング規約
- [PEP 8](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

### 再現性
- [The Turing Way](https://the-turing-way.netlify.app/)
- [Reproducible Research](https://reproducibleresearch.net/)

---

## フィードバック

このテンプレートに関するフィードバックや改善提案は歓迎します。
Issueやプルリクエストでお知らせください。

---

*このファイルは新しいプロジェクトを始める際に削除しても構いません。*
*ただし、プロジェクト終了後に自分の学びを記録する場所として活用することを推奨します。*
