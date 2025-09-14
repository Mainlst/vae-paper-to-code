# VAE: Paper ↔ Code Alignment (PyTorch examples/vae × Kingma & Welling 2013/2014)

英語版: [README.en.md](./README.en.md)

このリポジトリは、**VAE の元論文**（Kingma & Welling, 2013/2014）と
**PyTorch examples の `vae/main.py`** を 1 対 1 で照合しながら理解・再現・拡張するための最小実装です。

- 論文: Auto-Encoding Variational Bayes (arXiv:1312.6114)
- 参考実装: https://github.com/pytorch/examples/tree/main/vae

## What's inside
- MLP VAE (MNIST)
- Loss: BCE / BCE-with-logits / MSE を選択可能
- β-VAE / KL アニーリング (linear / cyclic)
- 学習曲線 (ELBO 内訳: recon, kl, total) の CSV/PNG 出力
- 再構成・サンプル・潜在トラバーサルの保存 (z=2時)

## 更新情報
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

```bash
# 構造化出力（推奨）: reports/<group>/<name>/...
# name には自動で日付-通し番号プレフィックス (YYYYMMDD-XXX-) が付与されます
# （例: 20250831-001-test-run や 20250831-001-seed42）。既存ディレクトリと衝突しないよう自動インクリメントします。
python -m src.train \
	--epochs 20 --batch-size 128 --latent-dim 20 \
	--loss bce --beta 1.0 --beta-schedule linear \
	--lr 1e-3 --device auto --seed 42 \
	--project-dir reports --group mnist-bench --name test-run

# 互換モード（--save-dir を明示指定すると直下に出力）
python -m src.train --epochs 5 --save-dir reports_legacy
```

主な引数:
- `--loss {bce,bce_logits,mse}`
- `--beta 1.0` / `--beta-schedule {none,linear,cyclic}`
- `--latent-dim 2|8|32|...`
- `--reduction {mean,sum}` (既定: mean)
- `--save-dir` 出力先 (curves, reconstructions, samples, traversals)

### Weights & Biases 連携（任意）
学習ログを [Weights & Biases](https://wandb.ai/) に送ることができます。

主なオプション:
- `--wandb` … 有効化スイッチ
- `--wandb-project` … プロジェクト名（既定: `vae-paper-to-code`）
- `--wandb-entity` … チーム/ユーザー（任意）
- `--wandb-mode {online,offline,disabled}` … 既定は `disabled`
- `--wandb-run-name` … ラン名を上書き（既定は `--name` と同じ）
- `--wandb-tags` … カンマ区切りタグ

API キーを設定してから（初回のみ）実行してください:

```bash
export WANDB_API_KEY=<your_api_key>
```

使用例:

```bash
python -m src.train \
	--epochs 10 --batch-size 128 --latent-dim 20 \
	--loss bce --beta 1.0 --beta-schedule linear \
	--project-dir reports --group mnist-bench --name test-run \
	--wandb --wandb-mode online --wandb-project vae-paper-to-code \
	--wandb-tags mnist,mlp,beta1
```

記録内容:
- スカラー: `loss/recon`, `loss/kl`, `loss/total`, `beta`, `epoch`
- 画像: 各エポックの `reconstructions`, `samples`（`latent_dim==2` のとき `traversal` も）

### 保存ポリシー（チェックポイント）
デフォルトでは重み（.pt）は保存しません。必要な場合は `--save-weights` を付けてください。

関連オプション:
- `--save-weights` … エポックごとに `vae_epoch_XXXX.pt` を保存
- `--no-date-prefix` … ラン名の自動プレフィックス付与を無効化

## Outputs
構造化出力の例（推奨）:

```
reports/
	<group>/
		latest -> ./<name>          # 直近ランへのシンボリックリンク（失敗時は latest.txt）
		<name>/
			run_meta.json             # すべての引数と設定のスナップショット
			vae_epoch_0000.pt         # --save-weights 指定時のみ
			curves/
				train_log.csv           # epoch, beta, recon, kl, total
				losses.png              # 学習曲線
			reconstructions/epoch_XXXX.png
			samples/epoch_XXXX.png
			traversals/epoch_XXXX.png # z=2 のとき
```

## Paper ↔ Code quick map
- 近似事後 q_φ(z|x)=N(μ,diag(σ²)) → `VAE.encode`（`mu, logvar`）
- 再パラメータ化 z=μ+σ⊙ε, ε~N(0,I) → `VAE.reparameterize`
- 尤度 p_θ(x|z)（ベルヌーイ仮定）→ `loss=bce`（`Sigmoid` 出力）
- **ELBO** ＝ 再構成項 − KL → 実装では `total = recon + beta * kl`（最小化 = −ELBO）

## License & Attribution
- `LICENSE-THIRD-PARTY.md` に、PyTorch examples への帰属とライセンス注意点を記載。
- 本リポジトリのコードは MIT 互換を想定（必要に応じて変更してください）

## Next steps
- ConvVAE（CIFAR-10）/ Gaussian likelihood (MSE) 比較
- IWAE (importance weighted bound)
- posterior collapse 対策の比較（βアニーリング、free bits など）

### Tips: 複数条件の一括実行
グリッド実験のサンプルスクリプトを用意しています。

```bash
bash scripts/run_grid.sh reports my-mnist-group
```

実行後の最新ランは `reports/my-mnist-group/latest` から辿れます。

### Tips: サンプル画像のモンタージュ
各エポックの `samples/epoch_*.png` をまとめて 1 枚にできます。

```bash
python scripts/make_samples_montage.py \
	--input-dir reports/my-mnist-group/latest/samples \
	--output reports/my-mnist-group/latest/samples_montage.png \
	--cols 5 --stride 1 --font-size 14
```
