# VAE: Paper ↔ Code Alignment (PyTorch examples/vae × Kingma & Welling 2013/2014)

英語版: [README.en.md](./README.en.md)

このリポジトリは、**VAE の元論文**（Kingma & Welling, 2013/2014）と
**PyTorch examples の `vae/main.py`** をそのまま活用しながら理解・再現・拡張するためのワークフロー集です。

- 論文: Auto-Encoding Variational Bayes (arXiv:1312.6114)
- 参考実装: https://github.com/pytorch/examples/tree/main/vae

## What's inside
- PyTorch 公式 VAE 実装（`examples/vae/main.py`）をそのまま呼び出すラッパー
- ノートブック `notebooks/vae_workflow.ipynb`：公式実装を中心に解説・可視化をまとめたチェックリスト
- 公式スクリプトの入出力（`examples/vae/results/`）を見やすく解析する補助コード
- サンプル画像をモンタージュ化するユーティリティ (`scripts/make_samples_montage.py`)

## リポジトリ構成（主なエントリーポイント）
- `examples/` … PyTorch 公式リポジトリから取得したサンプルコード（BSD 3-Clause）。`examples/vae/main.py` を直接呼び出せます。
- `src/run_official.py` … 公式実装を CLI から実行するためのラッパー。
- `src/pytorch_examples/` … 公式コードを変更せずに再利用するためのヘルパー群。
- `reports/` … 学習ログや生成画像の出力先。
- `notebooks/vae_workflow.ipynb` … 公式実装の動作確認と可視化ができるハンズオンノート。

## 更新情報
- 2025-09-28: 独自トレーナー（`src.train`）を終了し、公式実装＋ノートブック中心の構成に移行。
- 2025-08-31: 実験を繰り返しやすいよう出力ディレクトリを「グループ/ラン」単位に整理。
	- 新オプション: `--project-dir`, `--group`, `--name`
	- ランメタデータ `run_meta.json` を保存、`<group>/latest` シンボリックリンクを作成
	- グリッド実行サンプル `scripts/run_grid.sh` を追加

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### PyTorch 公式 VAE を動かす

```bash
python -m src.run_official -- --epochs 5 --batch-size 64
```

- `src/run_official.py` が `examples/vae/main.py` を同一プロセス内で起動します。
- `./data` フォルダが存在する場合、`examples/data` へのシンボリックリンクを自動作成して公式実装と共有します（失敗した場合は `examples/data` にダウンロードされます）。
- 出力画像は `examples/vae/results/` に保存され、公式実装と同じレイアウトで確認できます。
- 公式側の追加オプションは、そのまま続けて指定してください（例: `--no-accel`）。

### ノートブックでチェックリストを進める

`notebooks/vae_workflow.ipynb` を開き、上から順にセルを実行してください。以下をサポートしています。

- 環境セットアップと MNIST の確認
- 公式スクリプトの参照・実行
- 生成された画像（再構成・サンプル）の一覧や最新結果の可視化
- 複数条件で公式スクリプトを連続実行するヘルパー

### 生成結果の整理

公式スクリプトは既定で `examples/vae/results/` に以下のようなファイルを生成します。

```
examples/vae/results/
	reconstruction_1.png
	reconstruction_2.png
	...
	sample_1.png
	sample_2.png
	...
```

`scripts/make_samples_montage.py` を使うと、生成サンプルをモンタージュ画像にまとめられます。

## Outputs
公式スクリプトはエポックごとに以下の PNG を保存します（デフォルト設定）。

```
examples/vae/results/
    reconstruction_<epoch>.png
    sample_<epoch>.png
```

ノートブックではこれらのファイルを自動検出し、最新の画像やエポックごとの遷移を表示するセルを用意しています。

## Paper ↔ Code quick map
PyTorch 公式実装（`examples/vae/main.py`）内の該当箇所:

- 近似事後 q_φ(z|x)=N(μ,diag(σ²)) → `VAE.encode`
- 再パラメータ化 z=μ+σ⊙ε, ε~N(0,I) → `VAE.reparameterize`
- 尤度 p_θ(x|z)（ベルヌーイ仮定）→ `loss_function` 内の BCE 項
- **ELBO** ＝ 再構成項 − KL → `loss_function` の合計（最小化 = −ELBO）

## License & Attribution
- `examples/` 以下は PyTorch 公式リポジトリから取得したコードで、BSD 3-Clause License（`examples/LICENSE`）に従います。
- 本体コード（`src/` など）は `LICENSE` (MIT License) の下で配布しています。
- 公式実装を再利用するヘルパー (`src/pytorch_examples/*`, `src/run_official.py`) は本リポジトリのライセンスで提供しています。

## Next steps
- 公式コードに最小限の変更を加えて学習曲線やチェックポイントを保存する
- ConvVAE 化や別データセット（CIFAR-10 など）への拡張を検討する
- β アニーリングや IWAE など、論文で言及される改良アイデアを上乗せして比較する

### Tips: サンプル画像のモンタージュ
各エポックの `examples/vae/results/sample_*.png` をまとめて 1 枚にできます。

```bash
python scripts/make_samples_montage.py \
	--input-dir examples/vae/results \
	--pattern "sample_*.png" \
	--output examples/vae/results/samples_montage.png \
	--cols 5 --stride 1 --font-size 14
```
