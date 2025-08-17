# VAE: Paper ↔ Code Alignment (PyTorch examples/vae × Kingma & Welling 2013/2014)

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

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python -m src.train       --epochs 20 --batch-size 128 --latent-dim 20       --loss bce --beta 1.0 --beta-schedule linear       --lr 1e-3 --device auto --seed 42       --save-dir reports
```

主な引数:
- `--loss {bce,bce_logits,mse}`
- `--beta 1.0` / `--beta-schedule {none,linear,cyclic}`
- `--latent-dim 2|8|32|...`
- `--reduction {mean,sum}` (既定: mean)
- `--save-dir` 出力先 (curves, reconstructions, samples, traversals)

## Outputs
- `reports/curves/train_log.csv` … epoch, beta, recon, kl, total
- `reports/reconstructions/epoch_XXXX.png`
- `reports/samples/epoch_XXXX.png`
- `reports/traversals/epoch_XXXX.png` (z=2 のとき)
- `reports/curves/losses.png` (おまけグラフ)

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
