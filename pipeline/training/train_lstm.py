# LSTM 하이브리드 회귀 — 60일 6채널 시퀀스 + 24개 보조 피처 → 15개 라벨 동시 예측
# BiLSTM + 4-way 풀링 + 그룹별 분리 head + uncertainty weighting + RobustScaler

import sys
import math
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, LSTM_LABEL_COLS, LSTM_FEAT_COLS,
    LSTM_INPUT_CHANNELS, LSTM_LABEL_GROUPS,
)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"device: {DEVICE}")

# 데이터 로드
X      = np.load(DATA_DIR / "lstm_X.npy")        # (N, 60, 6)
feat   = np.load(DATA_DIR / "lstm_feat.npy")     # (N, 24)
labels = np.load(DATA_DIR / "lstm_labels.npy")   # (N, 15)
meta   = pd.read_csv(DATA_DIR / "lstm_meta.csv")

print(f"X: {X.shape}  feat: {feat.shape}  labels: {labels.shape}")

# 시계열 기반 train/valid/test 분할
meta["date"] = pd.to_datetime(meta["date"])
dates = np.sort(meta["date"].unique())
n = len(dates)

GAP = pd.Timedelta(days=75)
train_end   = dates[int(n * 0.70)]
valid_start = train_end + GAP
remaining   = dates[dates >= valid_start]
valid_end   = remaining[int(len(remaining) * 0.5)]
test_start  = valid_end + GAP

train_mask = meta["date"] <= train_end
valid_mask = (meta["date"] >= valid_start) & (meta["date"] <= valid_end)
test_mask  = meta["date"] >= test_start

X_train, feat_train, y_train = X[train_mask], feat[train_mask], labels[train_mask]
X_valid, feat_valid, y_valid = X[valid_mask], feat[valid_mask], labels[valid_mask]
X_test,  feat_test,  y_test  = X[test_mask],  feat[test_mask],  labels[test_mask]

print(f"train: {len(X_train):,}  valid: {len(X_valid):,}  test: {len(X_test):,}")
print(f"train ~{pd.Timestamp(train_end).date()} | gap | "
      f"valid {pd.Timestamp(valid_start).date()}~{pd.Timestamp(valid_end).date()} | gap | "
      f"test {pd.Timestamp(test_start).date()}~")

# 라벨 정규화 — RobustScaler (median/IQR) 로 outlier 영향 완화
y_median = np.median(y_train, axis=0)
y_q25    = np.quantile(y_train, 0.25, axis=0)
y_q75    = np.quantile(y_train, 0.75, axis=0)
y_iqr    = (y_q75 - y_q25) + 1e-8

y_train_norm = (y_train - y_median) / y_iqr
y_valid_norm = (y_valid - y_median) / y_iqr
y_test_norm  = (y_test  - y_median) / y_iqr

# 피처 정규화 — train 통계 기준
feat_mean = feat_train.mean(axis=0)
feat_std  = feat_train.std(axis=0) + 1e-8

feat_train_norm = (feat_train - feat_mean) / feat_std
feat_valid_norm = (feat_valid - feat_mean) / feat_std
feat_test_norm  = (feat_test  - feat_mean) / feat_std


# 시퀀스 augmentation — 학습 시 가벼운 변형
def augment_sequence(x):
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.02
    if random.random() < 0.3:
        scale = 1.0 + (random.random() - 0.5) * 0.1
        x = x * scale
    return x


# PyTorch Dataset
class HybridDataset(Dataset):
    def __init__(self, X, feat, y, augment=False):
        self.X       = torch.tensor(X,    dtype=torch.float32)
        self.feat    = torch.tensor(feat, dtype=torch.float32)
        self.y       = torch.tensor(y,    dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment_sequence(x)
        return x, self.feat[idx], self.y[idx]


BATCH_SIZE = 512

train_loader = DataLoader(HybridDataset(X_train, feat_train_norm, y_train_norm, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(HybridDataset(X_valid, feat_valid_norm, y_valid_norm),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(HybridDataset(X_test,  feat_test_norm,  y_test_norm),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# Attention pooling — 시퀀스 시점별 가중합
class AttnPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        # seq: (B, T, H)
        scores  = self.proj(seq).squeeze(-1)            # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (seq * weights).sum(dim=1)


# 라벨 인덱스 — 그룹별 분리 head 라우팅용
LABEL_TO_IDX  = {col: i for i, col in enumerate(LSTM_LABEL_COLS)}
GROUP_INDICES = {g: [LABEL_TO_IDX[c] for c in cols] for g, cols in LSTM_LABEL_GROUPS.items()}
GROUP_NAMES   = list(GROUP_INDICES.keys())


# Hybrid BiLSTM — 4-way 풀링 + 보조 피처 + 그룹별 분리 head
class HybridBiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 feat_size=24, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM 출력 = 2H. 4-way 풀링 = last(2H) + max(2H) + avg(2H) + attn(2H) = 8H
        self.attn_pool = AttnPool(hidden_size * 2)
        self.seq_norm  = nn.LayerNorm(hidden_size * 8)

        # 보조 피처 브랜치
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_size, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
        )

        # 공통 백본 — 분리 head 직전까지
        backbone_in  = hidden_size * 8 + 96
        self.backbone = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 그룹별 분리 head
        self.heads = nn.ModuleDict()
        for g, cols in LSTM_LABEL_GROUPS.items():
            self.heads[g] = nn.Sequential(
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, len(cols)),
            )

    def forward(self, seq, feat):
        out, _ = self.lstm(seq)            # (B, T, 2H)
        last     = out[:, -1, :]
        first    = out[:,  0, :]           # bidirectional이라 첫 시점도 정보 있음
        max_pool = out.max(dim=1).values
        avg_pool = out.mean(dim=1)
        attn     = self.attn_pool(out)
        seq_repr = torch.cat([last, max_pool, avg_pool, attn], dim=1)
        seq_repr = self.seq_norm(seq_repr)

        feat_repr = self.feat_branch(feat)
        h = torch.cat([seq_repr, feat_repr], dim=1)
        h = self.backbone(h)

        # 그룹별 head 출력 → 원래 라벨 순서로 재배치
        outs = torch.zeros(seq.size(0), len(LSTM_LABEL_COLS), device=seq.device)
        for g, idxs in GROUP_INDICES.items():
            head_out = self.heads[g](h)
            for i, lbl_idx in enumerate(idxs):
                outs[:, lbl_idx] = head_out[:, i]
        return outs


# Uncertainty weighting — 그룹별 학습 가능 log_var
class UncertaintyLoss(nn.Module):
    def __init__(self, group_names):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            g: nn.Parameter(torch.zeros(1)) for g in group_names
        })

    def forward(self, pred, target):
        total = 0.0
        for g, idxs in GROUP_INDICES.items():
            p_g = pred[:,   idxs]
            t_g = target[:, idxs]
            # huber loss per group
            l_g = F.huber_loss(p_g, t_g, delta=1.0)
            log_var = self.log_vars[g]
            # 1 / (2 sigma^2) * loss + 0.5 * log_var
            precision = torch.exp(-log_var)
            total = total + precision * l_g + 0.5 * log_var
        return total.squeeze()


N_FEAT = len(LSTM_FEAT_COLS)
model = HybridBiLSTM(
    input_size=LSTM_INPUT_CHANNELS,
    hidden_size=128,
    num_layers=2,
    feat_size=N_FEAT,
    dropout=0.3,
).to(DEVICE)
loss_fn = UncertaintyLoss(GROUP_NAMES).to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in loss_fn.parameters()):,}")

# AdamW + warmup + cosine
EPOCHS    = 60
EARLY_STOP = 12
WARMUP_EP = 5
LR_BASE   = 1e-3

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=LR_BASE, weight_decay=1e-4,
)


def lr_lambda(epoch):
    if epoch < WARMUP_EP:
        return (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 학습 루프
MODEL_PATH = OUTPUT_DIR / "lstm_best.pt"

best_val_loss = float("inf")
no_improve    = 0
history       = {"train_loss": [], "val_loss": []}


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, fb, yb in loader:
            xb, fb, yb = xb.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb, fb)
            loss = loss_fn(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total      += len(yb)
    return total_loss / total


print("\n학습 시작")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(valid_loader, train=False)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"[{epoch:02d}/{EPOCHS}]  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}", end="")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save({"model": model.state_dict(), "loss_fn": loss_fn.state_dict()}, MODEL_PATH)
        print("  *saved*")
    else:
        no_improve += 1
        print(f"  (no improve {no_improve}/{EARLY_STOP})")
        if no_improve >= EARLY_STOP:
            print("early stopping")
            break

print(f"\nbest val loss: {best_val_loss:.6f}")

# 학습 곡선 저장
plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"],   label="valid")
plt.title("Uncertainty-weighted Multitask Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_curve.png", dpi=150)
plt.close()

# 테스트셋 평가 — 원래 스케일 역변환
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model"])
loss_fn.load_state_dict(ckpt["loss_fn"])
model.eval()

all_preds = []
with torch.no_grad():
    for xb, fb, _ in test_loader:
        pred = model(xb.to(DEVICE), fb.to(DEVICE)).cpu().numpy()
        all_preds.append(pred)

preds_norm = np.concatenate(all_preds, axis=0)
preds_orig = preds_norm * y_iqr + y_median
y_test_orig = y_test

# baseline — test set 평균
baseline_preds = np.tile(y_test_orig.mean(axis=0), (len(y_test_orig), 1))

# 직전 best 결과 로드 — outputs/lstm_*/evaluation_results.csv 중 최신
prev_eval = None
prev_dir  = None
lstm_dirs = sorted([d for d in (ROOT / "outputs").glob("lstm_*")
                    if d != OUTPUT_DIR
                    and not d.name.startswith("lstm_seq2seq")
                    and (d / "evaluation_results.csv").exists()])
if lstm_dirs:
    prev_dir  = lstm_dirs[-1]
    prev_eval = pd.read_csv(prev_dir / "evaluation_results.csv")

print(f"\n{'='*85}")
print(f"{'target':25s}  {'MAE':>8s}  {'baseline':>8s}  {'gain':>8s}  {'prev_gain':>10s}  {'Δ':>8s}")
print(f"{'='*85}")

results = []
for i, col in enumerate(LSTM_LABEL_COLS):
    mae      = float(np.abs(preds_orig[:, i] - y_test_orig[:, i]).mean())
    bl_mae   = float(np.abs(baseline_preds[:, i] - y_test_orig[:, i]).mean())
    gain_pct = (1 - mae / bl_mae) * 100 if bl_mae > 0 else 0

    prev_gain  = None
    delta_str  = ""
    prev_str   = "    n/a"
    if prev_eval is not None and col in prev_eval["target"].values:
        prev_row  = prev_eval[prev_eval["target"] == col].iloc[0]
        prev_gain = float(prev_row["gain_pct"])
        delta     = gain_pct - prev_gain
        prev_str  = f"{prev_gain:+7.1f}%"
        flag      = "↑" if delta > 0 else ("↓" if delta < -1 else "≈")
        delta_str = f"{flag}{delta:+5.1f}"
    print(f"  {col:25s}  {mae:8.4f}  {bl_mae:8.4f}  {gain_pct:+7.1f}%  {prev_str:>10s}  {delta_str:>8s}")
    results.append({"target": col, "mae": mae, "baseline_mae": bl_mae,
                    "gain_pct": gain_pct, "prev_gain": prev_gain})

avg_mae  = np.mean([r["mae"] for r in results])
avg_bl   = np.mean([r["baseline_mae"] for r in results])
avg_gain = (1 - avg_mae / avg_bl) * 100 if avg_bl > 0 else 0

prev_avg_gain = None
if prev_eval is not None:
    prev_avg_gain = float(prev_eval["gain_pct"].mean())
prev_avg_str = f"{prev_avg_gain:+7.1f}%" if prev_avg_gain is not None else "    n/a"
delta_avg    = (avg_gain - prev_avg_gain) if prev_avg_gain is not None else None
delta_avg_str = f"{delta_avg:+5.1f}" if delta_avg is not None else ""
print(f"  {'AVERAGE':25s}  {avg_mae:8.4f}  {avg_bl:8.4f}  {avg_gain:+7.1f}%  {prev_avg_str:>10s}  {delta_avg_str:>8s}")
print(f"{'='*85}")
if prev_dir is not None:
    print(f"  비교 대상: {prev_dir.name}")
    print(f"  주의: peak_softpos / log_growth / delta 라벨은 직전 버전과 정의가 달라 직접 비교 불가")

pd.DataFrame(results).to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

# 그룹별 학습된 log_var 저장
group_log_vars = {g: float(loss_fn.log_vars[g].detach().cpu().item()) for g in GROUP_NAMES}
print("\n--- 그룹별 학습된 log_var (낮을수록 모델이 자신 있음) ---")
for g, lv in group_log_vars.items():
    print(f"  {g:15s}  log_var={lv:+.4f}  sigma={math.exp(lv/2):.4f}")

# 라벨별 산점도
n_labels = len(LSTM_LABEL_COLS)
ncols = 5
nrows = (n_labels + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()
for i, col in enumerate(LSTM_LABEL_COLS):
    ax = axes[i]
    ax.scatter(y_test_orig[:, i], preds_orig[:, i], alpha=0.05, s=2)
    lo = min(y_test_orig[:, i].min(), preds_orig[:, i].min())
    hi = max(y_test_orig[:, i].max(), preds_orig[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
    gain_i = results[i]["gain_pct"]
    ax.set_title(f"{col}\ngain={gain_i:+.1f}%", fontsize=8)
    ax.set_xlabel("actual", fontsize=7)
    ax.set_ylabel("predicted", fontsize=7)
    ax.tick_params(labelsize=6)
for j in range(n_labels, len(axes)):
    axes[j].axis("off")
plt.suptitle(f"Predicted vs Actual (avg gain: {avg_gain:+.1f}%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_all.png", dpi=150)
plt.close()

# summary
summary = {
    "avg_gain":      float(avg_gain),
    "avg_mae":       float(avg_mae),
    "best_val_loss": float(best_val_loss),
    "epochs_trained": len(history["train_loss"]),
    "group_log_vars": group_log_vars,
    "prev_dir":       str(prev_dir.name) if prev_dir else None,
    "prev_avg_gain":  prev_avg_gain,
    "delta_avg_gain": delta_avg,
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nresults -> {OUTPUT_DIR}")
