# seq2seq LSTM — BiLSTM 인코더 + multi-head attention 디코더 + residual output
# 입력: 60일 6채널 시퀀스 + 24개 보조 피처
# 출력: 15일 z-score 정규화 궤적

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
from pipeline.config import ROOT, LSTM_FEAT_COLS, LSTM_INPUT_CHANNELS

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / f"lstm_seq2seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"device: {DEVICE}")

# 데이터 로드
X     = np.load(DATA_DIR / "lstm_X.npy")        # (N, 60, 6)
feat  = np.load(DATA_DIR / "lstm_feat.npy")     # (N, 24)
seq_y = np.load(DATA_DIR / "lstm_seq_y.npy")    # (N, 15) z-score
meta  = pd.read_csv(DATA_DIR / "lstm_meta.csv")

print(f"X: {X.shape}  feat: {feat.shape}  seq_y: {seq_y.shape}")
print(f"seq_y  mean={seq_y.mean():.4f}  std={seq_y.std():.4f}  "
      f"min={seq_y.min():.4f}  max={seq_y.max():.4f}")

# 시계열 기반 분할
meta["date"] = pd.to_datetime(meta["date"])
dates = np.sort(meta["date"].unique())
n     = len(dates)

GAP = pd.Timedelta(days=75)
train_end   = dates[int(n * 0.70)]
valid_start = train_end + GAP
remaining   = dates[dates >= valid_start]
valid_end   = remaining[int(len(remaining) * 0.5)]
test_start  = valid_end + GAP

train_mask = meta["date"] <= train_end
valid_mask = (meta["date"] >= valid_start) & (meta["date"] <= valid_end)
test_mask  = meta["date"] >= test_start

X_train, feat_train, y_train = X[train_mask], feat[train_mask], seq_y[train_mask]
X_valid, feat_valid, y_valid = X[valid_mask], feat[valid_mask], seq_y[valid_mask]
X_test,  feat_test,  y_test  = X[test_mask],  feat[test_mask],  seq_y[test_mask]

# 역변환용 메타 (test set)
meta_test = meta[test_mask].reset_index(drop=True)

print(f"train: {len(X_train):,}  valid: {len(X_valid):,}  test: {len(X_test):,}")

# 피처 정규화 — train 통계 기준
feat_mean = feat_train.mean(axis=0)
feat_std  = feat_train.std(axis=0) + 1e-8

feat_train_n = (feat_train - feat_mean) / feat_std
feat_valid_n = (feat_valid - feat_mean) / feat_std
feat_test_n  = (feat_test  - feat_mean) / feat_std


# 시퀀스 augmentation — 학습 시에만 가벼운 변형
def augment_sequence(x):
    # 50% jitter sigma=0.02
    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.02
    # 30% magnitude scaling 0.95~1.05
    if random.random() < 0.3:
        scale = 1.0 + (random.random() - 0.5) * 0.1
        x = x * scale
    return x


# PyTorch Dataset
class SeqDataset(Dataset):
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

train_loader = DataLoader(SeqDataset(X_train, feat_train_n, y_train, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(SeqDataset(X_valid, feat_valid_n, y_valid),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(SeqDataset(X_test,  feat_test_n,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# Multi-head attention — 디코더 hidden과 인코더 출력 간 다중 어텐션
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.q_proj    = nn.Linear(hidden_size, hidden_size)
        self.k_proj    = nn.Linear(hidden_size, hidden_size)
        self.v_proj    = nn.Linear(hidden_size, hidden_size)
        self.out_proj  = nn.Linear(hidden_size, hidden_size)
        self.scale     = math.sqrt(self.head_dim)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: (B, H), enc_outputs: (B, T, H)
        B = dec_hidden.size(0)
        T = enc_outputs.size(1)
        H = enc_outputs.size(2)

        q = self.q_proj(dec_hidden).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc_outputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc_outputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, heads, 1, T)
        weights = F.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, H)
        return self.out_proj(ctx)


# seq2seq — BiLSTM 인코더 + multi-head attention + residual delta 디코더
class Seq2SeqBiAttn(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, enc_layers=2,
                 feat_size=24, out_steps=15, num_heads=2, dropout=0.3):
        super().__init__()
        self.out_steps   = out_steps
        self.hidden_size = hidden_size

        # 인코더 — BiLSTM. 출력 차원 2H를 H로 압축해서 attention/디코더와 호환
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=enc_layers, batch_first=True, bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.enc_out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_h_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_c_proj   = nn.Linear(hidden_size * 2, hidden_size)

        # 피처 주입 — 인코더 hidden + 디코더 init 양쪽에 사용
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 디코더 — autoregressive. residual delta 예측
        self.attention    = MultiHeadAttention(hidden_size, num_heads=num_heads)
        self.input_proj   = nn.Linear(1, hidden_size)
        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.delta_head   = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def _merge_directions(self, hc):
        # hc: (num_layers*2, B, H) → 양방향 마지막 레이어를 concat 후 H로 압축
        last_fwd = hc[-2]
        last_bwd = hc[-1]
        return torch.cat([last_fwd, last_bwd], dim=-1)  # (B, 2H)

    def forward(self, seq, feat, target=None, tf_ratio=0.5):
        B = seq.size(0)

        # 인코더
        enc_outputs, (h, c) = self.encoder(seq)
        enc_outputs = self.enc_out_proj(enc_outputs)  # (B, T, H)

        h_merged = self._merge_directions(h)  # (B, 2H)
        c_merged = self._merge_directions(c)
        dec_h    = self.enc_h_proj(h_merged)
        dec_c    = self.enc_c_proj(c_merged)

        # 피처 주입 — 디코더 초기 hidden에 가산
        feat_vec = self.feat_proj(feat)
        dec_h    = dec_h + feat_vec

        # 첫 입력 — 0 (z-score 평균)
        prev_value = torch.zeros(B, 1, device=seq.device)

        outputs = []
        for t in range(self.out_steps):
            ctx = self.attention(dec_h, enc_outputs)
            prev_proj = self.input_proj(prev_value)
            dec_in    = torch.cat([prev_proj, ctx], dim=1)

            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            delta = self.delta_head(self.dropout(dec_h))

            # residual — 이전 값 + delta
            pred = prev_value + delta

            outputs.append(pred)

            # teacher forcing
            if target is not None and random.random() < tf_ratio:
                prev_value = target[:, t:t+1]
            else:
                prev_value = pred.detach()

        return torch.cat(outputs, dim=1)


# 궤적 손실 — 포인트 + 형태 + 기울기
def trajectory_loss(pred, true, alpha=0.6, beta=0.25, gamma=0.15):
    point_loss = F.huber_loss(pred, true)

    cos_sim    = F.cosine_similarity(pred, true, dim=1)
    cos_sim    = torch.where(torch.isnan(cos_sim), torch.zeros_like(cos_sim), cos_sim)
    shape_loss = (1 - cos_sim).mean()

    pred_diff  = pred[:, 1:] - pred[:, :-1]
    true_diff  = true[:, 1:] - true[:, :-1]
    slope_loss = F.mse_loss(pred_diff, true_diff)

    return alpha * point_loss + beta * shape_loss + gamma * slope_loss


N_FEAT = len(LSTM_FEAT_COLS)
model  = Seq2SeqBiAttn(
    input_size=LSTM_INPUT_CHANNELS,
    hidden_size=128,
    enc_layers=2,
    feat_size=N_FEAT,
    out_steps=15,
    num_heads=2,
    dropout=0.3,
).to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# AdamW + warmup + cosine annealing
EPOCHS     = 60
EARLY_STOP = 12
WARMUP_EP  = 5
LR_BASE    = 1e-3

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=1e-4)


def lr_lambda(epoch):
    if epoch < WARMUP_EP:
        return (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

MODEL_PATH = OUTPUT_DIR / "seq2seq_best.pt"

best_val_loss = float("inf")
no_improve    = 0
history       = {"train_loss": [], "val_loss": []}


def run_epoch(loader, train=True, tf_ratio=0.5):
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, fb, yb in loader:
            xb, fb, yb = xb.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
            if train:
                pred = model(xb, fb, target=yb, tf_ratio=tf_ratio)
            else:
                pred = model(xb, fb, target=None, tf_ratio=0.0)
            loss = trajectory_loss(pred, yb)
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
    # scheduled sampling — 전 구간 선형 감소 (1.0 → 0.0)
    tf_ratio = max(0.0, 1.0 - (epoch - 1) / max(1, EPOCHS - 1))

    train_loss = run_epoch(train_loader, train=True,  tf_ratio=tf_ratio)
    val_loss   = run_epoch(valid_loader, train=False, tf_ratio=0.0)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"[{epoch:02d}/{EPOCHS}]  train={train_loss:.6f}  val={val_loss:.6f}  "
          f"tf={tf_ratio:.2f}  lr={lr_now:.2e}", end="")

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
plt.title("Trajectory Loss (huber + shape + slope)")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "train_curve.png", dpi=150)
plt.close()

# 테스트셋 최종 평가
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for xb, fb, _ in test_loader:
        pred = model(xb.to(DEVICE), fb.to(DEVICE), target=None, tf_ratio=0.0)
        all_preds.append(pred.cpu().numpy())

preds_z = np.concatenate(all_preds, axis=0)

# 원래 스케일 역변환
seq_y_mean = meta_test["seq_y_mean"].values[:, None]
seq_y_std  = meta_test["seq_y_std"].values[:, None]
preds_orig = preds_z * seq_y_std + seq_y_mean
y_test_orig = y_test * seq_y_std + seq_y_mean

# 평가 지표
mae_per_day  = np.abs(preds_orig - y_test_orig).mean(axis=0)
rmse_per_day = np.sqrt(((preds_orig - y_test_orig) ** 2).mean(axis=0))

correlations = []
for i in range(len(preds_orig)):
    if np.std(preds_orig[i]) > 1e-8 and np.std(y_test_orig[i]) > 1e-8:
        c = np.corrcoef(preds_orig[i], y_test_orig[i])[0, 1]
        correlations.append(c if not np.isnan(c) else 0.0)
    else:
        correlations.append(0.0)
correlations = np.array(correlations)
mean_corr    = float(correlations.mean())

# 분포 지표 — 잘 맞춘 비율, 반대 방향 비율
ratio_good     = float((correlations >  0.7).mean())
ratio_decent   = float(((correlations >  0.3) & (correlations <= 0.7)).mean())
ratio_neutral  = float((np.abs(correlations) <= 0.3).mean())
ratio_opposite = float((correlations < -0.3).mean())

# z-score cosine
cos_sims = []
for i in range(len(preds_z)):
    p, t = preds_z[i], y_test[i]
    np_, nt = np.linalg.norm(p), np.linalg.norm(t)
    if np_ > 1e-8 and nt > 1e-8:
        cos_sims.append(float(np.dot(p, t) / (np_ * nt)))
    else:
        cos_sims.append(0.0)
mean_cos_sim = float(np.mean(cos_sims))

print(f"\n--- 테스트셋 평가 ---")
print(f"MAE (원래 스케일 평균)  : {mae_per_day.mean():.4f}")
print(f"RMSE (원래 스케일 평균) : {rmse_per_day.mean():.4f}")
print(f"궤적 상관계수 (원래)    : {mean_corr:.4f}")
print(f"cosine similarity (z)   : {mean_cos_sim:.4f}")
print(f"\n--- 궤적 분포 ---")
print(f"잘 맞춤 (r >  0.7) : {ratio_good*100:5.1f}%")
print(f"적당   (0.3~0.7)  : {ratio_decent*100:5.1f}%")
print(f"무관   (|r|<=0.3) : {ratio_neutral*100:5.1f}%")
print(f"반대   (r < -0.3) : {ratio_opposite*100:5.1f}%")

# 직전 best 비교 — outputs/lstm_seq2seq_*/summary.json 중 최신
prev_summary = None
seq2seq_dirs = sorted([d for d in (ROOT / "outputs").glob("lstm_seq2seq_*")
                       if d != OUTPUT_DIR and (d / "summary.json").exists()])
if seq2seq_dirs:
    prev_path = seq2seq_dirs[-1] / "summary.json"
    try:
        prev_summary = json.loads(prev_path.read_text())
        print(f"\n--- 직전 best 비교 ({seq2seq_dirs[-1].name}) ---")
        for k in ["mae_mean", "rmse_mean", "mean_corr", "mean_cos_sim"]:
            if k in prev_summary:
                cur = {"mae_mean": float(mae_per_day.mean()),
                       "rmse_mean": float(rmse_per_day.mean()),
                       "mean_corr": mean_corr,
                       "mean_cos_sim": mean_cos_sim}[k]
                prev = float(prev_summary[k])
                if k in ["mae_mean", "rmse_mean"]:
                    delta = (cur - prev) / prev * 100
                    flag  = "↓ 개선" if delta < 0 else "↑ 회귀"
                else:
                    delta = cur - prev
                    flag  = "↑ 개선" if delta > 0 else "↓ 회귀"
                print(f"  {k:15s}  prev={prev:.4f}  cur={cur:.4f}  ({delta:+.2f}{'%' if k in ['mae_mean','rmse_mean'] else ''})  {flag}")
    except Exception as e:
        print(f"\n  prev summary load error: {e}")
else:
    print("\n  (이전 summary.json 없음, 비교 생략)")

print("\nday별 MAE:")
for d, (mae, rmse) in enumerate(zip(mae_per_day, rmse_per_day), 1):
    bar = "#" * int(mae / (mae_per_day.max() + 1e-8) * 25)
    print(f"  day{d:2d}: MAE={mae:.4f}  RMSE={rmse:.4f}  {bar}")

# 그래프
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 16), mae_per_day,  marker="o", label="MAE")
ax.plot(range(1, 16), rmse_per_day, marker="s", label="RMSE")
ax.set_xlabel("Forecast Day")
ax.set_ylabel("Error (original scale)")
ax.set_title("Error by Forecast Day")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "day_mae.png", dpi=150)
plt.close()

# 궤적 시각화
sorted_idx = np.argsort(correlations)
sample_indices = np.concatenate([sorted_idx[-3:], sorted_idx[:3]])

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, idx in enumerate(sample_indices):
    ax = axes[i]
    ax.plot(y_test_orig[idx], label="actual",    color="steelblue")
    ax.plot(preds_orig[idx],  label="predicted", color="tomato", linestyle="--")
    kw   = meta_test.iloc[idx]["keyword"]
    corr = correlations[idx]
    tag  = "BEST" if i < 3 else "WORST"
    ax.set_title(f"[{tag}] {kw}  r={corr:.3f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlabel("day")
plt.suptitle(f"Trajectory Prediction  (mean_corr={mean_corr:.3f}, cos_sim={mean_cos_sim:.3f})")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trajectory_samples.png", dpi=150)
plt.close()

# 결과 저장
eval_df = pd.DataFrame({"day": range(1, 16), "mae": mae_per_day, "rmse": rmse_per_day})
eval_df.to_csv(OUTPUT_DIR / "day_metrics.csv", index=False)

summary = {
    "mae_mean":       float(mae_per_day.mean()),
    "rmse_mean":      float(rmse_per_day.mean()),
    "mean_corr":      mean_corr,
    "mean_cos_sim":   mean_cos_sim,
    "ratio_good":     ratio_good,
    "ratio_decent":   ratio_decent,
    "ratio_neutral":  ratio_neutral,
    "ratio_opposite": ratio_opposite,
    "best_val_loss":  best_val_loss,
    "epochs_trained": len(history["train_loss"]),
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nresults -> {OUTPUT_DIR}")
