# VAE: Paper ↔ Code Alignment (PyTorch examples/vae × Kingma & Welling 2013/2014)

このリポジトリは、**VAE の元論文**（Kingma & Welling, 2013/2014）と  
**PyTorch examples の `vae/main.py`** を 1 対 1 で照合しながら理解・再現・拡張するための最小実装です。

- 論文: *Auto-Encoding Variational Bayes* (arXiv:1312.6114)
- 参考実装: https://github.com/pytorch/examples/tree/main/vae

## What’s inside
- MLP VAE (MNIST)
- Loss: BCE / BCE-with-logits / MSE を選択可能
- β-VAE / KL アニーリング (linear / cyclic)
- 学習曲線 (ELBO 内訳: recon, kl, total) の CSV/PNG 出力
- 再構成・サンプル・潜在トラバーサルの保存

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
