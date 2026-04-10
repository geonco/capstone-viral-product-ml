# LSTM 하이브리드 회귀 — 60일 시퀀스 + 보조 피처 → 13개 라벨 동시 예측

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, LSTM_LABEL_COLS, LSTM_FEAT_COLS

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"device: {DEVICE}")

# 데이터 로드
X      = np.load(DATA_DIR / "lstm_X.npy")       # (N, 60, 4)
feat   = np.load(DATA_DIR / "lstm_feat.npy")     # (N, 19)
labels = np.load(DATA_DIR / "lstm_labels.npy")   # (N, 13)
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

# 타겟 z-score 정규화 — train 통계 기준
y_mean = y_train.mean(axis=0)
y_std  = y_train.std(axis=0) + 1e-8

y_train_norm = (y_train - y_mean) / y_std
y_valid_norm = (y_valid - y_mean) / y_std
y_test_norm  = (y_test  - y_mean) / y_std

# 피처 정규화 — train 통계 기준
feat_mean = feat_train.mean(axis=0)
feat_std  = feat_train.std(axis=0) + 1e-8

feat_train_norm = (feat_train - feat_mean) / feat_std
feat_valid_norm = (feat_valid - feat_mean) / feat_std
feat_test_norm  = (feat_test  - feat_mean) / feat_std


# PyTorch Dataset — 시퀀스 + 피처 + 라벨
class HybridDataset(Dataset):
    def __init__(self, X, feat, y):
        self.X    = torch.tensor(X,    dtype=torch.float32)
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.y    = torch.tensor(y,    dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.feat[idx], self.y[idx]


BATCH_SIZE = 512

train_loader = DataLoader(HybridDataset(X_train, feat_train_norm, y_train_norm),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(HybridDataset(X_valid, feat_valid_norm, y_valid_norm),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(HybridDataset(X_test,  feat_test_norm,  y_test_norm),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# 하이브리드 LSTM — 시퀀스 인코더 + 피처 브랜치 → 13 라벨 동시 예측
class HybridLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2,
                 feat_size=19, num_labels=13, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        # 피처 브랜치 — 스케일·채널관계·맥락 정보 인코딩
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_size, 64),
            nn.ReLU(),
        )
        # last hidden + max pooling = 256, feat = 64, total = 320
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, seq, feat):
        out, _ = self.lstm(seq)
        last     = out[:, -1, :]              # (batch, 128) 최근 상태
        max_pool = out.max(dim=1).values      # (batch, 128) 전체 최대 활성
        seq_repr = torch.cat([last, max_pool], dim=1)

        feat_repr = self.feat_proj(feat)
        combined  = torch.cat([seq_repr, feat_repr], dim=1)
        return self.head(combined)


N_FEAT   = len(LSTM_FEAT_COLS)
N_LABELS = len(LSTM_LABEL_COLS)

model = HybridLSTM(feat_size=N_FEAT, num_labels=N_LABELS).to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# 학습 루프
EPOCHS     = 50
EARLY_STOP = 10
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
            loss = criterion(pred, yb)
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
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"[{epoch:02d}/{EPOCHS}]  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.2e}", end="")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), MODEL_PATH)
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
plt.title("Huber Loss")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_curve.png", dpi=150)
plt.close()

# 테스트셋 최종 평가 — 원래 스케일로 역변환
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for xb, fb, _ in test_loader:
        pred = model(xb.to(DEVICE), fb.to(DEVICE)).cpu().numpy()
        all_preds.append(pred)

preds_norm = np.concatenate(all_preds, axis=0)

# 역변환 — train 통계 기준
preds_orig = preds_norm * y_std + y_mean
y_test_orig = y_test

# naive baseline — test set 평균
baseline_preds = np.tile(y_test_orig.mean(axis=0), (len(y_test_orig), 1))

print(f"\n{'='*65}")
print(f"{'target':25s}  {'MAE':>8s}  {'baseline':>8s}  {'gain':>8s}")
print(f"{'='*65}")

results = []
for i, col in enumerate(LSTM_LABEL_COLS):
    mae      = float(np.abs(preds_orig[:, i] - y_test_orig[:, i]).mean())
    bl_mae   = float(np.abs(baseline_preds[:, i] - y_test_orig[:, i]).mean())
    gain_pct = (1 - mae / bl_mae) * 100 if bl_mae > 0 else 0
    print(f"  {col:25s}  {mae:8.4f}  {bl_mae:8.4f}  {gain_pct:+7.1f}%")
    results.append({"target": col, "mae": mae, "baseline_mae": bl_mae, "gain_pct": gain_pct})

avg_mae  = np.mean([r["mae"] for r in results])
avg_bl   = np.mean([r["baseline_mae"] for r in results])
avg_gain = (1 - avg_mae / avg_bl) * 100 if avg_bl > 0 else 0
print(f"  {'AVERAGE':25s}  {avg_mae:8.4f}  {avg_bl:8.4f}  {avg_gain:+7.1f}%")
print(f"{'='*65}")

pd.DataFrame(results).to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

# 타겟별 예측 vs 실제 산점도
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(LSTM_LABEL_COLS):
    if i >= 13:
        break
    ax = axes[i]
    ax.scatter(y_test_orig[:, i], preds_orig[:, i], alpha=0.05, s=2)
    lo = min(y_test_orig[:, i].min(), preds_orig[:, i].min())
    hi = max(y_test_orig[:, i].max(), preds_orig[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
    ax.set_title(col, fontsize=8)
    ax.set_xlabel("actual", fontsize=7)
    ax.set_ylabel("predicted", fontsize=7)
    ax.tick_params(labelsize=6)

for j in range(len(LSTM_LABEL_COLS), len(axes)):
    axes[j].axis("off")
plt.suptitle(f"Predicted vs Actual (avg gain: {avg_gain:+.1f}%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_all.png", dpi=150)
plt.close()

print(f"\nresults -> {OUTPUT_DIR}")
