#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ObservableNet trainer
---------------------
Dedicated neural emulator for global observables of Schwarzschild photon scattering.

Targets:
    1) delta_phi = Delta_phi - pi
    2) rho_min   = r_min / R_S

Derived quantity:
    N_wind = Delta_phi / (2*pi) = (delta_phi + pi) / (2*pi)

Validity:
    only for scattering trajectories (b > b_crit)

Main features:
- synthetic exact dataset from analytic formula
- CUDA training
- dynamic learning rate scheduler
- best-model checkpointing during training
- final benchmark and plots
- clean outputs for local runs or HPC

Example:
python3 train_observable_net.py \
  --outdir /home/kingsman/Escritorio/PhD/PINN_paper/observable_run001 \
  --nsamples 60000 \
  --epochs 200 \
  --width 256 \
  --depth 4 \
  --dropout 0.05 \
  --batch 256 \
  --init-lr 1e-3 \
  --weight-decay 1e-4 \
  --loss huber \
  --rs-min 0.5 \
  --rs-max 3.0 \
  --bratio-min 1.01 \
  --bratio-max 2.5 \
  --seed 123 \
  --amp
"""

import os
import json
import time
import random
import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.special import ellipk, ellipkinc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bcrit(Rs):
    """
    Critical impact parameter:
        b_c = (3*sqrt(3)/2) * R_S
    """
    return 1.5 * np.sqrt(3.0) * Rs


# ============================================================
# Exact analytic physics
# ============================================================

def cubic_roots_u(Rs, b):
    """
    Roots of:
        u^3 - u^2 + (Rs/b)^2 = 0

    For scattering trajectories:
        u1 < 0 < u2 < u3 < 1

    The closest-approach turning point is:
        u_turn = u2
    """
    coeff = [1.0, -1.0, 0.0, (Rs / b) ** 2]
    roots = np.roots(coeff)
    roots = np.real_if_close(roots, tol=1e5)

    if np.any(np.abs(np.imag(roots)) > 1e-10):
        return None

    roots = np.sort(np.real(roots))
    return roots


def scattering_observables_exact(Rs, b):
    """
    Exact scattering observables for Schwarzschild null geodesics.

    Returns
    -------
    dict or None
        Returns None if the input is not in the scattering regime.
    """
    roots = cubic_roots_u(Rs, b)
    if roots is None:
        return None

    u1, u2, u3 = roots

    # Valid scattering ordering
    if not (u1 < 0 < u2 < u3 < 1):
        return None

    m = (u2 - u1) / (u3 - u1)
    if not (0 <= m < 1):
        return None

    z = np.sqrt((-u1) / (u2 - u1))
    z = np.clip(z, 0.0, 1.0)

    K_complete = ellipk(m)
    F_incomplete = ellipkinc(np.arcsin(z), m)

    Delta_phi = 4.0 / np.sqrt(u3 - u1) * (K_complete - F_incomplete)
    delta_phi = Delta_phi - np.pi

    # Closest approach in scattering:
    u_turn = u2
    r_min = Rs / u_turn
    rho_min = r_min / Rs
    N_wind = Delta_phi / (2.0 * np.pi)

    return {
        "delta_phi": float(delta_phi),
        "Delta_phi": float(Delta_phi),
        "u1": float(u1),
        "u2": float(u2),
        "u3": float(u3),
        "u_turn": float(u_turn),
        "r_min": float(r_min),
        "rho_min": float(rho_min),
        "N_wind": float(N_wind),
    }


def deflection_weak_eq65(Rs, b):
    """
    Weak-deflection approximation.
    """
    x = Rs / b
    return float(2 * x + (15 * np.pi / 16) * (x ** 2) + (16 / 3) * (x ** 3))


def deflection_strong_eq66(Rs, b):
    """
    Strong-deflection logarithmic approximation near b_c.
    """
    denom = 1.0 - (3 * np.sqrt(3) * Rs / (2 * b)) ** 2
    if denom <= 0:
        return np.nan
    Delta_phi = np.log(432 * (2 - np.sqrt(3)) ** 2 / denom)
    return float(Delta_phi - np.pi)


# ============================================================
# Dataset generation
# ============================================================

def make_observables_dataset(
    n_samples: int,
    Rs_range=(0.5, 3.0),
    b_ratio_range=(1.01, 2.5),
    seed: int = 123,
) -> pd.DataFrame:
    """
    Build exact synthetic dataset for scattering trajectories only.

    Features:
        Rs, b, b_over_bc, Rs_over_b, log_gap

    Targets:
        delta_phi
        rho_min = r_min / Rs

    Derived:
        N_wind
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_samples):
        Rs = rng.uniform(*Rs_range)
        bc = bcrit(Rs)

        b_ratio = rng.uniform(*b_ratio_range)
        b = b_ratio * bc

        obs = scattering_observables_exact(Rs, b)
        if obs is None:
            continue

        rows.append({
            "Rs": Rs,
            "b": b,
            "bc": bc,
            "b_over_bc": b / bc,
            "Rs_over_b": Rs / b,
            "log_gap": np.log((b / bc) - 1.0),

            "delta_phi": obs["delta_phi"],
            "Delta_phi": obs["Delta_phi"],

            "u_turn": obs["u_turn"],
            "r_min": obs["r_min"],
            "rho_min": obs["rho_min"],

            "N_wind": obs["N_wind"],

            # targets for training
            "target_dphi": np.log1p(obs["delta_phi"]),
            "target_rmin": np.log(obs["rho_min"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# Torch dataset
# ============================================================

class ObservableDataset(Dataset):
    def __init__(self, df, features, targets, mu_x, std_x, mu_y, std_y):
        X = df[features].values.astype(np.float32)
        y = df[targets].values.astype(np.float32)

        self.X = (X - mu_x) / std_x
        self.y = (y - mu_y) / std_y

        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Model
# ============================================================

class ObservableNet(nn.Module):
    def __init__(self, in_dim, width=256, depth=4, dropout=0.05, out_dim=2):
        super().__init__()

        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [
                nn.Linear(d, width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = width

        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Evaluation helpers
# ============================================================

def inverse_targets(y_norm, mu_y, std_y):
    """
    y_norm shape: (N, 2)
    target 0: log1p(delta_phi)
    target 1: log(rho_min)
    """
    y = y_norm * std_y + mu_y

    delta_phi = np.expm1(y[:, 0])
    rho_min = np.exp(y[:, 1])
    N_wind = (delta_phi + np.pi) / (2.0 * np.pi)

    return {
        "delta_phi": delta_phi,
        "rho_min": rho_min,
        "N_wind": N_wind,
    }


def metrics_1d(true, pred):
    return {
        "mse": float(mean_squared_error(true, pred)),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)),
    }


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds, trues = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yhat = model(xb).cpu().numpy()
        preds.append(yhat)
        trues.append(yb.numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return preds, trues


def evaluate_observables(model, loader, mu_y, std_y, device):
    pred_z, true_z = predict_loader(model, loader, device)

    pred_phys = inverse_targets(pred_z, mu_y, std_y)
    true_phys = inverse_targets(true_z, mu_y, std_y)

    out = {
        "delta_phi": {
            **metrics_1d(true_phys["delta_phi"], pred_phys["delta_phi"]),
            "true": true_phys["delta_phi"],
            "pred": pred_phys["delta_phi"],
        },
        "rho_min": {
            **metrics_1d(true_phys["rho_min"], pred_phys["rho_min"]),
            "true": true_phys["rho_min"],
            "pred": pred_phys["rho_min"],
        },
        "N_wind": {
            **metrics_1d(true_phys["N_wind"], pred_phys["N_wind"]),
            "true": true_phys["N_wind"],
            "pred": pred_phys["N_wind"],
        },
    }
    return out


@torch.no_grad()
def predict_observables(model, norm_info, Rs, b, device):
    Rs = np.atleast_1d(Rs).astype(np.float32)
    b = np.atleast_1d(b).astype(np.float32)

    if len(Rs) == 1 and len(b) > 1:
        Rs = np.full_like(b, Rs.item(), dtype=np.float32)
    elif len(b) == 1 and len(Rs) > 1:
        b = np.full_like(Rs, b.item(), dtype=np.float32)
    elif len(Rs) != len(b):
        raise ValueError(f"Rs and b must have the same length, got len(Rs)={len(Rs)}, len(b)={len(b)}")

    bc = bcrit(Rs)

    feat_df = pd.DataFrame({
        "Rs": Rs,
        "b": b,
        "b_over_bc": b / bc,
        "Rs_over_b": Rs / b,
        "log_gap": np.log((b / bc) - 1.0),
    })

    features = norm_info["features"]
    X = feat_df[features].values.astype(np.float32)

    mu_x = np.asarray(norm_info["mu_x"], dtype=np.float32)
    std_x = np.asarray(norm_info["std_x"], dtype=np.float32)
    mu_y = np.asarray(norm_info["mu_y"], dtype=np.float32)
    std_y = np.asarray(norm_info["std_y"], dtype=np.float32)

    Xn = (X - mu_x) / std_x
    Xn = torch.from_numpy(Xn).to(device)

    model.eval()
    z = model(Xn).cpu().numpy()

    pred = inverse_targets(z, mu_y, std_y)
    return pred


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    outdir: str
    nsamples: int
    epochs: int
    width: int
    depth: int
    dropout: float
    batch: int
    init_lr: float
    weight_decay: float
    loss: str
    rs_min: float
    rs_max: float
    bratio_min: float
    bratio_max: float
    train_frac: float
    val_frac: float
    seed: int
    num_workers: int
    amp: bool
    scheduler: str
    min_lr: float
    patience: int


# ============================================================
# Training
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=60000)

    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch", type=int, default=256)

    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])

    parser.add_argument("--rs-min", type=float, default=0.5)
    parser.add_argument("--rs-max", type=float, default=3.0)
    parser.add_argument("--bratio-min", type=float, default=1.01)
    parser.add_argument("--bratio-max", type=float, default=2.5)

    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)

    args = parser.parse_args()
    cfg = TrainConfig(
        outdir=args.outdir,
        nsamples=args.nsamples,
        epochs=args.epochs,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        batch=args.batch,
        init_lr=args.init_lr,
        weight_decay=args.weight_decay,
        loss=args.loss,
        rs_min=args.rs_min,
        rs_max=args.rs_max,
        bratio_min=args.bratio_min,
        bratio_max=args.bratio_max,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
        scheduler=args.scheduler,
        min_lr=args.min_lr,
        patience=args.patience,
    )

    ensure_dir(cfg.outdir)
    ckpt_dir = os.path.join(cfg.outdir, "ckpt")
    fig_dir = os.path.join(cfg.outdir, "figs")
    tab_dir = os.path.join(cfg.outdir, "tables")
    data_dir = os.path.join(cfg.outdir, "data")
    for d in [ckpt_dir, fig_dir, tab_dir, data_dir]:
        ensure_dir(d)

    with open(os.path.join(cfg.outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")
    if device == "cuda":
        print(f"[INFO] gpu = {torch.cuda.get_device_name(0)}")

    # ----------------------------
    # Dataset
    # ----------------------------
    print("[INFO] building dataset...")
    df = make_observables_dataset(
        n_samples=cfg.nsamples,
        Rs_range=(cfg.rs_min, cfg.rs_max),
        b_ratio_range=(cfg.bratio_min, cfg.bratio_max),
        seed=cfg.seed,
    )
    print(f"[INFO] dataset rows = {len(df)}")

    data_csv = os.path.join(data_dir, "observables_dataset.csv")
    df.to_csv(data_csv, index=False)

    FEATURES = ["Rs", "b", "b_over_bc", "Rs_over_b", "log_gap"]
    TARGETS = ["target_dphi", "target_rmin"]

    test_frac = 1.0 - cfg.train_frac - cfg.val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1")

    train_df, temp_df = train_test_split(df, test_size=(1.0 - cfg.train_frac), random_state=cfg.seed)
    rel_test = test_frac / (test_frac + cfg.val_frac)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, random_state=cfg.seed)

    print(f"[INFO] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    mu_x = train_df[FEATURES].mean().values.astype(np.float32)
    std_x = train_df[FEATURES].std().replace(0, 1.0).values.astype(np.float32)

    mu_y = train_df[TARGETS].mean().values.astype(np.float32)
    std_y = train_df[TARGETS].std().replace(0, 1.0).values.astype(np.float32)
    std_y[std_y == 0] = 1.0

    norm_info = {
        "features": FEATURES,
        "targets": TARGETS,
        "mu_x": mu_x.tolist(),
        "std_x": std_x.tolist(),
        "mu_y": mu_y.tolist(),
        "std_y": std_y.tolist(),
    }
    with open(os.path.join(tab_dir, "normalization.json"), "w") as f:
        json.dump(norm_info, f, indent=2)

    train_ds = ObservableDataset(train_df, FEATURES, TARGETS, mu_x, std_x, mu_y, std_y)
    val_ds = ObservableDataset(val_df, FEATURES, TARGETS, mu_x, std_x, mu_y, std_y)
    test_ds = ObservableDataset(test_df, FEATURES, TARGETS, mu_x, std_x, mu_y, std_y)

    pin_memory = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1024, cfg.batch),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1024, cfg.batch),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(cfg.num_workers > 0),
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = ObservableNet(
        in_dim=len(FEATURES),
        width=cfg.width,
        depth=cfg.depth,
        dropout=cfg.dropout,
        out_dim=len(TARGETS),
    ).to(device)

    if cfg.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.init_lr,
        weight_decay=cfg.weight_decay,
    )

    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_lr,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device == "cuda"))

    # ----------------------------
    # Train loop
    # ----------------------------
    history = []
    best_val = np.inf
    best_epoch = -1
    t0 = time.perf_counter()

    print("[INFO] starting training...")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device == "cuda")):
                yhat = model(xb)
                loss = criterion(yhat, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * len(xb)
            train_n += len(xb)

        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(cfg.amp and device == "cuda")):
                    yhat = model(xb)
                    loss = criterion(yhat, yb)

                val_loss_sum += loss.item() * len(xb)
                val_n += len(xb)

        val_loss = val_loss_sum / max(val_n, 1)

        if cfg.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch

            best_ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": float(best_val),
                "config": asdict(cfg),
                "features": FEATURES,
                "targets": TARGETS,
                "mu_x": mu_x,
                "std_x": std_x,
                "mu_y": mu_y,
                "std_y": std_y,
            }
            torch.save(best_ckpt, os.path.join(ckpt_dir, "best.pt"))

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr_now),
            "best_val_so_far": float(best_val),
            "best_epoch_so_far": int(best_epoch),
        }
        history.append(row)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            print(
                f"[{epoch+1:4d}/{cfg.epochs}] "
                f"train={train_loss:.4e} "
                f"val={val_loss:.4e} "
                f"lr={lr_now:.3e} "
                f"best={best_val:.4e} @ ep {best_epoch}"
            )

    elapsed = time.perf_counter() - t0
    print(f"[INFO] training done in {elapsed/60:.2f} min")

    # save last checkpoint too
    last_ckpt = {
        "model_state": model.state_dict(),
        "epoch": cfg.epochs - 1,
        "config": asdict(cfg),
        "features": FEATURES,
        "targets": TARGETS,
        "mu_x": mu_x,
        "std_x": std_x,
        "mu_y": mu_y,
        "std_y": std_y,
    }
    torch.save(last_ckpt, os.path.join(ckpt_dir, "last.pt"))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(tab_dir, "history.csv"), index=False)

    # ----------------------------
    # Load best and evaluate
    # ----------------------------
    best_loaded = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_loaded["model_state"])
    model.eval()

    val_metrics = evaluate_observables(model, val_loader, mu_y, std_y, device)
    test_metrics = evaluate_observables(model, test_loader, mu_y, std_y, device)

    metrics = {
        "best_epoch": int(best_loaded["epoch"]),
        "best_val_loss": float(best_loaded["best_val_loss"]),

        "val_delta_phi": {
            "mse": val_metrics["delta_phi"]["mse"],
            "mae": val_metrics["delta_phi"]["mae"],
            "r2": val_metrics["delta_phi"]["r2"],
        },
        "test_delta_phi": {
            "mse": test_metrics["delta_phi"]["mse"],
            "mae": test_metrics["delta_phi"]["mae"],
            "r2": test_metrics["delta_phi"]["r2"],
        },

        "val_rho_min": {
            "mse": val_metrics["rho_min"]["mse"],
            "mae": val_metrics["rho_min"]["mae"],
            "r2": val_metrics["rho_min"]["r2"],
        },
        "test_rho_min": {
            "mse": test_metrics["rho_min"]["mse"],
            "mae": test_metrics["rho_min"]["mae"],
            "r2": test_metrics["rho_min"]["r2"],
        },

        "val_N_wind": {
            "mse": val_metrics["N_wind"]["mse"],
            "mae": val_metrics["N_wind"]["mae"],
            "r2": val_metrics["N_wind"]["r2"],
        },
        "test_N_wind": {
            "mse": test_metrics["N_wind"]["mse"],
            "mae": test_metrics["N_wind"]["mae"],
            "r2": test_metrics["N_wind"]["r2"],
        },

        "elapsed_sec": float(elapsed),
    }

    with open(os.path.join(tab_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] test metrics:")
    print(json.dumps(metrics, indent=2))

    # ----------------------------
    # Plots: learning curves
    # ----------------------------
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Learning curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(7.5, 4.0))
    plt.plot(history_df["epoch"], history_df["lr"])
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.title("LR schedule")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "lr_schedule.png"), dpi=220)
    plt.close()

    # ----------------------------
    # Test scatter plots
    # ----------------------------
    plt.figure(figsize=(5.8, 5.5))
    true_ = test_metrics["delta_phi"]["true"]
    pred_ = test_metrics["delta_phi"]["pred"]
    xymax = float(max(np.max(true_), np.max(pred_)))
    plt.scatter(true_, pred_, s=8, alpha=0.35)
    plt.plot([0, xymax], [0, xymax], "--")
    plt.xlabel(r"true $\delta\phi$")
    plt.ylabel(r"predicted $\delta\phi$")
    plt.title("Test set: deflection angle")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "test_scatter_delta_phi.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(5.8, 5.5))
    true_ = test_metrics["rho_min"]["true"]
    pred_ = test_metrics["rho_min"]["pred"]
    xymax = float(max(np.max(true_), np.max(pred_)))
    plt.scatter(true_, pred_, s=8, alpha=0.35)
    plt.plot([0, xymax], [0, xymax], "--")
    plt.xlabel(r"true $r_{\min}/R_S$")
    plt.ylabel(r"predicted $r_{\min}/R_S$")
    plt.title("Test set: minimum approach radius")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "test_scatter_rho_min.png"), dpi=220)
    plt.close()

    # ----------------------------
    # Benchmark grid for Rs=1
    # ----------------------------
    Rs_bench = 1.0
    bc_bench = bcrit(Rs_bench)
    b_grid = np.linspace(1.01 * bc_bench, 1.40 * bc_bench, 500)

    exact_rows = []
    for b in b_grid:
        exact_rows.append(scattering_observables_exact(Rs_bench, b))
    exact_df = pd.DataFrame(exact_rows)

    pred = predict_observables(model, norm_info, Rs_bench, b_grid, device=device)

    bench_df = pd.DataFrame({
        "Rs": Rs_bench,
        "b": b_grid,
        "b_over_bc": b_grid / bc_bench,

        "delta_exact": exact_df["delta_phi"].values,
        "delta_pred": pred["delta_phi"],

        "rho_min_exact": exact_df["rho_min"].values,
        "rho_min_pred": pred["rho_min"],

        "N_wind_exact": exact_df["N_wind"].values,
        "N_wind_pred": pred["N_wind"],

        "delta_weak": np.array([deflection_weak_eq65(Rs_bench, b) for b in b_grid]),
        "delta_strong": np.array([deflection_strong_eq66(Rs_bench, b) for b in b_grid]),
    })

    bench_df["abs_err_delta"] = np.abs(bench_df["delta_pred"] - bench_df["delta_exact"])
    bench_df["rel_err_delta"] = bench_df["abs_err_delta"] / np.abs(bench_df["delta_exact"])

    bench_df["abs_err_rho_min"] = np.abs(bench_df["rho_min_pred"] - bench_df["rho_min_exact"])
    bench_df["rel_err_rho_min"] = bench_df["abs_err_rho_min"] / np.abs(bench_df["rho_min_exact"])

    bench_df["abs_err_N_wind"] = np.abs(bench_df["N_wind_pred"] - bench_df["N_wind_exact"])
    bench_df["rel_err_N_wind"] = bench_df["abs_err_N_wind"] / np.abs(bench_df["N_wind_exact"])

    bench_df.to_csv(os.path.join(tab_dir, "benchmark_Rs1.csv"), index=False)

    # delta_phi
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(bench_df["b_over_bc"], bench_df["delta_exact"], label="Exact")
    plt.plot(bench_df["b_over_bc"], bench_df["delta_pred"], "--", label="ObservableNet")
    plt.plot(bench_df["b_over_bc"], bench_df["delta_weak"], ":", label="Weak-field")
    plt.plot(bench_df["b_over_bc"], bench_df["delta_strong"], "-.", label="Strong-field")
    plt.xlabel(r"$b/b_c$")
    plt.ylabel(r"$\delta\phi$ [rad]")
    plt.title("Deflection angle vs impact parameter")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "delta_vs_b.png"), dpi=220)
    plt.close()

    # rho_min
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(bench_df["b_over_bc"], bench_df["rho_min_exact"], label="Exact")
    plt.plot(bench_df["b_over_bc"], bench_df["rho_min_pred"], "--", label="ObservableNet")
    plt.xlabel(r"$b/b_c$")
    plt.ylabel(r"$r_{\min}/R_S$")
    plt.title("Minimum approach radius vs impact parameter")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rmin_vs_b.png"), dpi=220)
    plt.close()

    # winding
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(bench_df["b_over_bc"], bench_df["N_wind_exact"], label="Exact")
    plt.plot(bench_df["b_over_bc"], bench_df["N_wind_pred"], "--", label="ObservableNet")
    plt.xlabel(r"$b/b_c$")
    plt.ylabel(r"$N_{\rm wind}$")
    plt.title("Winding number vs impact parameter")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "winding_vs_b.png"), dpi=220)
    plt.close()

    # error plots
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(bench_df["b_over_bc"], bench_df["abs_err_delta"])
    plt.yscale("log")
    plt.xlabel(r"$b/b_c$")
    plt.ylabel(r"$|\delta\phi_{\rm NN}-\delta\phi_{\rm exact}|$")
    plt.title("Absolute error: deflection angle")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "abs_error_delta.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(bench_df["b_over_bc"], bench_df["abs_err_rho_min"])
    plt.yscale("log")
    plt.xlabel(r"$b/b_c$")
    plt.ylabel(r"$|(r_{\min}/R_S)_{\rm NN}-(r_{\min}/R_S)_{\rm exact}|$")
    plt.title("Absolute error: minimum approach radius")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "abs_error_rho_min.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(bench_df["b_over_bc"], bench_df["abs_err_N_wind"])
    plt.yscale("log")
    plt.xlabel(r"$b/b_c$")
    plt.ylabel(r"$|N_{{\rm wind,NN}}-N_{{\rm wind,exact}}|$")
    plt.title("Absolute error: winding number")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "abs_error_winding.png"), dpi=220)
    plt.close()

    # ----------------------------
    # Summary
    # ----------------------------
    summary = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "best_epoch": int(best_loaded["epoch"]),
        "best_val_loss": float(best_loaded["best_val_loss"]),

        "test_delta_phi_mse": float(test_metrics["delta_phi"]["mse"]),
        "test_delta_phi_mae": float(test_metrics["delta_phi"]["mae"]),
        "test_delta_phi_r2": float(test_metrics["delta_phi"]["r2"]),

        "test_rho_min_mse": float(test_metrics["rho_min"]["mse"]),
        "test_rho_min_mae": float(test_metrics["rho_min"]["mae"]),
        "test_rho_min_r2": float(test_metrics["rho_min"]["r2"]),

        "test_N_wind_mse": float(test_metrics["N_wind"]["mse"]),
        "test_N_wind_mae": float(test_metrics["N_wind"]["mae"]),
        "test_N_wind_r2": float(test_metrics["N_wind"]["r2"]),

        "benchmark_mean_abs_err_delta": float(bench_df["abs_err_delta"].mean()),
        "benchmark_mean_abs_err_rho_min": float(bench_df["abs_err_rho_min"].mean()),
        "benchmark_mean_abs_err_N_wind": float(bench_df["abs_err_N_wind"].mean()),
    }
    with open(os.path.join(tab_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] finished successfully")


if __name__ == "__main__":
    main()

    