

# ネットカフェ動的価格提案システム PoC

## 概要

このリポジトリは、ネットカフェにおける席種別の動的価格設定を支援するシステムのProof of Concept（概念実証）です。需要予測モデルと価格最適化アルゴリズムを組み合わせ、収益を最大化する価格提案を行います。

## 特徴

- 日次・席種別の予約データに基づく需要予測
- イベント情報を考慮した予測精度の向上
- 価格弾力性を考慮した最適価格の算出
- 直感的なStreamlitインターフェースによる価格設定支援
- 将来日の予測と価格提案機能

## システム構成

```
├── data/                      # データファイル
│   ├── reservations.csv       # 日次・席種別予約実績
│   ├── sales.csv              # 日次・席種別売上
│   ├── events.csv             # イベント情報（date,event_type,popularity）
│   └── prices.csv             # 日次・席種別適用価格
├── models/                    # 学習済みモデル（.gitignoreで除外）
│   ├── model_baseline_*.pkl   # Prophet ベースライン需要予測モデル
│   └── model_price_*.pkl      # Prophet 需要–価格反応モデル
├── app.py                     # Streamlit アプリ本体
├── train.py                   # モデル学習スクリプト
├── evaluate.py                # モデル評価スクリプト
├── price_calc.py              # 価格最適化ロジック
├── utils.py                   # 共通ユーティリティ関数
└── requirements.txt           # 必要ライブラリ一覧
```

## 使用技術

- **Prophet**: Facebookが開発した時系列予測ライブラリ
- **Streamlit**: データアプリケーション構築フレームワーク
- **Pandas/NumPy**: データ処理・分析ライブラリ
- **Scikit-learn**: 機械学習ライブラリ（モデル評価）

## セットアップ方法

1. リポジトリのクローン
   ```bash
   git clone https://github.com/kai00514/dynamic_pricing_poc.git
   cd dynamic_pricing_poc
   ```

2. 仮想環境の作成と依存ライブラリのインストール
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. サンプルデータの生成（必要な場合）
   ```bash
   python generate_sample_data.py
   ```

4. モデルの学習
   ```bash
   python train.py
   ```

5. アプリケーションの起動
   ```bash
   streamlit run app.py
   ```

## 使用方法

1. Streamlitアプリにアクセス（デフォルト: http://localhost:8501）
2. 日付と席種を選択
3. システムが自動的に需要予測と最適価格を計算
4. 推奨価格を確認し、必要に応じて調整
5. 「価格を確定」ボタンで設定を保存

## モデル評価

モデルの予測精度を評価するには以下のコマンドを実行します：

```bash
python evaluate.py
```

このスクリプトは、テストデータに対するモデルの平均絶対誤差（MAE）を計算します。

## 今後の展望

- リアルタイムデータ連携による予測精度の向上
- 複数の席種間の相互影響を考慮したモデリング
- 顧客セグメント別の価格最適化
- 予約システムとの完全統合

## ライセンス

このプロジェクトは [MITライセンス](LICENSE) の下で公開されています。

## 貢献

バグ報告や機能リクエストは、GitHubのIssueを通じてお知らせください。プルリクエストも歓迎します。

---

**注意**: このシステムはPoC（概念実証）であり、実際の運用には追加の検証と調整が必要です。
