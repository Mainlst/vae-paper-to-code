from __future__ import annotations
import argparse
import os
from dataclasses import asdict
from datetime import datetime
import json
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
try:
    from torchvision import datasets, transforms  # type: ignore
    _HAS_TORCHVISION = True
except Exception as e:  # pragma: no cover
    print(f"[warn] torchvision import failed: {e}. Using local MNIST loader.")
    _HAS_TORCHVISION = False

from .vae import VAE, VAEConfig, reconstruction_loss, kld_standard_normal
from .utils.vis import ensure_dir, save_reconstructions, save_samples, save_traversal


def beta_scheduler(kind: str, epoch: int, total_epochs: int, base_beta: float) -> float:
    if kind == "none":
        return base_beta
    if kind == "linear":
        t = (epoch + 1) / max(1, total_epochs)
        return base_beta * min(1.0, t)
    if kind == "cyclic":
        t = (epoch + 1) / max(1, total_epochs)
        val = 2.0 * (0.5 - abs((t % 1.0) - 0.5))  # 0→1→0
        return base_beta * val
    raise ValueError(f"Unknown beta schedule: {kind}")


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE (paper↔code alignment)")
    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=400)
    p.add_argument("--loss", type=str, default="bce", choices=["bce", "bce_logits", "mse"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-schedule", type=str, default="none", choices=["none", "linear", "cyclic"])
    p.add_argument("--reduction", type=str, default="mean", choices=["mean", "sum"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pin", action="store_true", help="Disable DataLoader pin_memory (auto-disabled on CPU)")
    p.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive across epochs (requires num_workers>0)")

    # Output (new structured layout)
    p.add_argument(
        "--project-dir",
        type=str,
        default="reports",
        help="Project root directory to store all experiments (default: reports)",
    )
    p.add_argument(
        "--group",
        type=str,
        default="",
        help="Experiment group name (e.g., mnist-lat20-b1.0-linear-bce). If empty, generated automatically.",
    )
    p.add_argument(
        "--name",
        type=str,
        default="",
        help="Run name under the group. If empty, generated as seed{seed} (prefixed with date-serial).",
    )
    p.add_argument(
        "--no-date-prefix",
        action="store_true",
        help="Disable auto prefixing run name with YYYYMMDD-XXX serial (per group/day).",
    )
    p.add_argument(
        "--save-weights",
        action="store_true",
        help="Save model checkpoints (default: do not save).",
    )
    # Deprecated but kept for compatibility: if explicitly set to non-default, overrides structured layout
    p.add_argument(
        "--save-dir",
        type=str,
        default="reports",
        help="[Deprecated] Direct output directory. If set to a non-default value, it overrides --project-dir/--group/--name.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)

    # Data: MNIST in [0,1], 28x28 → flatten
    if _HAS_TORCHVISION:
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
    else:
        from .utils.mnist import MNISTLocal
        transform = None  # MNISTLocal already returns tensors in [0,1]
        train_ds = MNISTLocal(root="./data", train=True, transform=transform)
        test_ds = MNISTLocal(root="./data", train=False, transform=transform)

    pin_mem = (device.type == "cuda") and (not args.no_pin)
    pw = args.persistent_workers and (args.num_workers > 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=pw,
    )

    # Model
    output_logits = (args.loss == "bce_logits")
    cfg = VAEConfig(
        input_dim=28 * 28,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_logits=output_logits,
    )
    model = VAE(cfg).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    # Build structured output directories
    def _auto_group() -> str:
        # Keep concise but informative
        return (
            f"mnist-lat{args.latent_dim}-"
            f"{args.loss}-"
            f"beta{args.beta:g}-{args.beta_schedule}-"
            f"{args.reduction}"
        )

    group = args.group or _auto_group()

    def _gen_prefixed_name(base: str) -> str:
        if args.no_date_prefix:
            return base
        today = datetime.now().strftime('%Y%m%d')
        group_dir = os.path.join(args.project_dir, group)
        ensure_dir(group_dir)
        # If base already starts with today's date-serial, keep as is
        import re
        if re.match(rf"^{today}-\\d{{3}}[\-_]", base):
            return base
        # Find next serial for today
        existing = []
        try:
            for entry in os.listdir(group_dir):
                if os.path.isdir(os.path.join(group_dir, entry)) and entry.startswith(f"{today}-"):
                    m = re.match(rf"^{today}-(\\d{{3}})", entry)
                    if m:
                        existing.append(int(m.group(1)))
        except Exception:
            pass
        next_id = (max(existing) + 1) if existing else 1
        prefix = f"{today}-{next_id:03d}"
        candidate = f"{prefix}-{base}" if base else prefix
        # Ensure uniqueness
        while os.path.exists(os.path.join(group_dir, candidate)):
            next_id += 1
            prefix = f"{today}-{next_id:03d}"
            candidate = f"{prefix}-{base}" if base else prefix
        return candidate

    base_name = args.name or f"seed{args.seed}"
    name = _gen_prefixed_name(base_name)

    # Backward-compat: if save-dir is explicitly set to non-default, use it as the final run dir
    use_legacy = ("--save-dir" in os.sys.argv) and (args.save_dir != "reports")
    run_dir = args.save_dir if use_legacy else os.path.join(args.project_dir, group, name)

    # Subdirs
    curves_dir = os.path.join(run_dir, "curves")
    recon_dir = os.path.join(run_dir, "reconstructions")
    sample_dir = os.path.join(run_dir, "samples")
    trav_dir = os.path.join(run_dir, "traversals")
    for d in [curves_dir, recon_dir, sample_dir, trav_dir]:
        ensure_dir(d)

    # Maintain a "latest" pointer per group for convenience
    try:
        group_dir = os.path.join(args.project_dir, group)
        ensure_dir(group_dir)
        latest_link = os.path.join(group_dir, "latest")
        # Try to create/refresh symlink; fall back to a text file
        try:
            if os.path.islink(latest_link) or os.path.isfile(latest_link):
                os.remove(latest_link)
            os.symlink(os.path.relpath(run_dir, group_dir), latest_link)
        except Exception:
            with open(os.path.join(group_dir, "latest.txt"), "w", encoding="utf-8") as f:
                f.write(run_dir + "\n")
    except Exception as e:
        print(f"[warn] failed to update latest pointer: {e}")

    # Save run metadata for reproducibility
    try:
        meta = {
            "group": group,
            "name": name,
            "run_dir": run_dir,
            "args": vars(args),
            "cfg": asdict(cfg),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[warn] failed to save run_meta.json: {e}")

    log_rows = []

    def evaluate(epoch: int, beta_val: float):
        model.eval()
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x.to(device).view(x.size(0), -1)
            x_hat, mu, logvar = model(x)
            recon = reconstruction_loss(x, x_hat, args.loss, args.reduction)
            kl = kld_standard_normal(mu, logvar, args.reduction)
            total = recon + beta_val * kl

            # Save reconstructions
            x_in = x.view(-1, 1, 28, 28)
            x_out = x_hat
            if args.loss == "bce_logits":
                x_out = torch.sigmoid(x_out)
            x_out = x_out.view(-1, 1, 28, 28)
            save_reconstructions(x_in, x_out, os.path.join(recon_dir, f"epoch_{epoch:04d}.png"), n=8)

            # Save random samples
            z = torch.randn(64, cfg.latent_dim, device=device)
            x_samp = model.decode(z)
            if args.loss == "bce_logits":
                x_samp = torch.sigmoid(x_samp)
            x_samp = x_samp.view(-1, 1, 28, 28)
            save_samples(x_samp, os.path.join(sample_dir, f"epoch_{epoch:04d}.png"), nrow=8)

            # Traversal for z=2
            if cfg.latent_dim == 2:
                save_traversal(model.decode, device, cfg.latent_dim, os.path.join(trav_dir, f"epoch_{epoch:04d}.png"))

        return recon.item(), kl.item(), total.item()

    for epoch in range(args.epochs):
        model.train()
        beta_val = beta_scheduler(args.beta_schedule, epoch, args.epochs, args.beta)

        for x, _ in train_loader:
            x = x.to(device).view(x.size(0), -1)
            x_hat, mu, logvar = model(x)

            recon = reconstruction_loss(x, x_hat, args.loss, args.reduction)
            kl = kld_standard_normal(mu, logvar, args.reduction)
            loss = recon + beta_val * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # evaluation + logging
        recon_val, kl_val, total_val = evaluate(epoch, beta_val)
        log_rows.append({"epoch": epoch, "beta": beta_val, "recon": recon_val, "kl": kl_val, "total": total_val})
        print(f"[{epoch+1:03d}/{args.epochs}] beta={beta_val:.3f} recon={recon_val:.3f} kl={kl_val:.3f} total={total_val:.3f}")

        # persist model
        if args.save_weights:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "args": vars(args),
                },
                os.path.join(run_dir, f"vae_epoch_{epoch:04d}.pt"),
            )

        # save curves CSV and quick plot
        # Use a non-interactive backend to avoid blocking in headless envs
        import matplotlib  # type: ignore
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore

        df = pd.DataFrame(log_rows)
        csv_path = os.path.join(curves_dir, "train_log.csv")
        df.to_csv(csv_path, index=False)

        try:
            plt.figure()
            plt.plot(df["epoch"], df["recon"], label="recon")
            plt.plot(df["epoch"], df["kl"], label="kl")
            plt.plot(df["epoch"], df["total"], label="total")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, "losses.png"))
            plt.close()
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
