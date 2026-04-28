# seq2seq LSTM — attention 디코더 + teacher forcing → 미래 15일 궤적 예측
# 입력: 60일 z-score 시퀀스 + 보조 피처
# 출력: 15일 z-score 정규화 궤적

import sys
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
from pipeline.config import ROOT, LSTM_FEAT_COLS

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / f"lstm_seq2seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"device: {DEVICE}")

# 데이터 로드
X     = np.load(DATA_DIR / "lstm_X.npy")       # (N, 60, 4)
feat  = np.load(DATA_DIR / "lstm_feat.npy")     # (N, 19)
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


# PyTorch Dataset
class SeqDataset(Dataset):
    def __init__(self, X, feat, y):
        self.X    = torch.tensor(X,    dtype=torch.float32)
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.y    = torch.tensor(y,    dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.feat[idx], self.y[idx]


BATCH_SIZE = 512

train_loader = DataLoader(SeqDataset(X_train, feat_train_n, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(SeqDataset(X_valid, feat_valid_n, y_valid),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(SeqDataset(X_test,  feat_test_n,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# Bahdanau Attention — 디코더 hidden과 인코더 전체 출력 간 가중 결합
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: (batch, hidden), enc_outputs: (batch, seq_len, hidden)
        seq_len    = enc_outputs.size(1)
        dec_expand = dec_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        combined   = torch.cat([dec_expand, enc_outputs], dim=2)
        energy     = torch.tanh(self.W(combined))
        scores     = self.v(energy).squeeze(-1)           # (batch, seq_len)
        weights    = F.softmax(scores, dim=1)
        context    = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


# seq2seq LSTM — attention 디코더 + teacher forcing + 피처 주입
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2,
                 feat_size=19, out_steps=15, dropout=0.3):
        super().__init__()
        self.out_steps   = out_steps
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 인코더 — 과거 시퀀스 압축
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        # 피처 주입 — 인코더 hidden에 더해 스케일·맥락 정보 전달
        self.feat_proj = nn.Linear(feat_size, hidden_size)

        # 디코더 — autoregressive, 매 스텝 attention으로 인코더 참조
        self.attention   = Attention(hidden_size)
        self.input_proj  = nn.Linear(1, hidden_size)
        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, feat, target=None, tf_ratio=0.5):
        batch_size = seq.size(0)

        # 인코더
        enc_outputs, (h, c) = self.encoder(seq)

        # 피처 주입 — 마지막 레이어 hidden에 가산
        feat_vec = self.feat_proj(feat)
        h_mod    = h.clone()
        h_mod[-1] = h_mod[-1] + feat_vec

        # 디코더 초기 상태
        dec_h = h_mod[-1]
        dec_c = c[-1]

        # 첫 입력 — 0으로 시작 (z-score 기준 평균)
        prev_value = torch.zeros(batch_size, 1, device=seq.device)

        outputs = []
        for t in range(self.out_steps):
            # attention — 인코더 전체 hidden states 참조
            context, _ = self.attention(dec_h, enc_outputs)

            # 디코더 입력 — 이전 예측값 임베딩 + attention context
            prev_proj = self.input_proj(prev_value)
            dec_in    = torch.cat([prev_proj, context], dim=1)

            # 디코더 스텝
            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            pred = self.output_proj(self.dropout(dec_h))

            outputs.append(pred)

            # teacher forcing — 확률적으로 실제값 사용
            if target is not None and random.random() < tf_ratio:
                prev_value = target[:, t:t+1]
            else:
                prev_value = pred.detach()

        return torch.cat(outputs, dim=1)


# 궤적 손실 — 포인트 정확도 + 형태 유사도 + 기울기 매칭
def trajectory_loss(pred, true, alpha=0.7, beta=0.2, gamma=0.1):
    # 포인트 정확도 — 각 day 값이 맞는가
    point_loss = F.huber_loss(pred, true)

    # 형태 유사도 — cosine similarity (1=완벽, -1=반대)
    cos_sim    = F.cosine_similarity(pred, true, dim=1)
    cos_sim    = torch.where(torch.isnan(cos_sim), torch.zeros_like(cos_sim), cos_sim)
    shape_loss = (1 - cos_sim).mean()

    # 기울기 매칭 — 일별 변화 방향과 크기
    pred_diff  = pred[:, 1:] - pred[:, :-1]
    true_diff  = true[:, 1:] - true[:, :-1]
    slope_loss = F.mse_loss(pred_diff, true_diff)

    return alpha * point_loss + beta * shape_loss + gamma * slope_loss


N_FEAT = len(LSTM_FEAT_COLS)
model  = Seq2SeqAttention(feat_size=N_FEAT).to(DEVICE)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# 학습 루프
EPOCHS     = 50
EARLY_STOP = 10
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
    # scheduled sampling — teacher forcing 비율 점진 감소
    tf_ratio = max(0.1, 1.0 - epoch / (EPOCHS * 0.7))

    train_loss = run_epoch(train_loader, train=True,  tf_ratio=tf_ratio)
    val_loss   = run_epoch(valid_loader, train=False, tf_ratio=0.0)
    scheduler.step(val_loss)

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

preds_z = np.concatenate(all_preds, axis=0)   # z-score 스케일

# 원래 스케일 역변환 — 메타에 저장된 mean/std 사용
seq_y_mean = meta_test["seq_y_mean"].values[:, None]
seq_y_std  = meta_test["seq_y_std"].values[:, None]
preds_orig = preds_z * seq_y_std + seq_y_mean
y_test_orig = y_test * seq_y_std + seq_y_mean

# 평가 지표 — day별 MAE, RMSE, 궤적 상관계수
mae_per_day  = np.abs(preds_orig - y_test_orig).mean(axis=0)
rmse_per_day = np.sqrt(((preds_orig - y_test_orig) ** 2).mean(axis=0))

correlations = []
for i in range(len(preds_orig)):
    if np.std(preds_orig[i]) > 1e-8 and np.std(y_test_orig[i]) > 1e-8:
        c = np.corrcoef(preds_orig[i], y_test_orig[i])[0, 1]
        correlations.append(c if not np.isnan(c) else 0.0)
    else:
        correlations.append(0.0)
mean_corr = float(np.mean(correlations))

# z-score 스케일에서의 shape metrics
cos_sims = []
for i in range(len(preds_z)):
    p, t = preds_z[i], y_test[i]
    norm_p, norm_t = np.linalg.norm(p), np.linalg.norm(t)
    if norm_p > 1e-8 and norm_t > 1e-8:
        cos_sims.append(float(np.dot(p, t) / (norm_p * norm_t)))
    else:
        cos_sims.append(0.0)
mean_cos_sim = float(np.mean(cos_sims))

print(f"\n--- 테스트셋 평가 ---")
print(f"MAE (원래 스케일 평균)  : {mae_per_day.mean():.4f}")
print(f"RMSE (원래 스케일 평균) : {rmse_per_day.mean():.4f}")
print(f"궤적 상관계수 (원래)    : {mean_corr:.4f}")
print(f"cosine similarity (z)   : {mean_cos_sim:.4f}")

print("\nday별 MAE:")
for d, (mae, rmse) in enumerate(zip(mae_per_day, rmse_per_day), 1):
    bar = "#" * int(mae / (mae_per_day.max() + 1e-8) * 25)
    print(f"  day{d:2d}: MAE={mae:.4f}  RMSE={rmse:.4f}  {bar}")

# 그래프 — day별 MAE/RMSE
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

# 궤적 시각화 — 상관 상위/하위/중간 각 2개씩
corr_arr   = np.array(correlations)
sorted_idx = np.argsort(corr_arr)

sample_indices = np.concatenate([
    sorted_idx[-3:],                         # best 3
    sorted_idx[:3],                          # worst 3
])

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, idx in enumerate(sample_indices):
    ax = axes[i]
    ax.plot(y_test_orig[idx],  label="actual",    color="steelblue")
    ax.plot(preds_orig[idx],   label="predicted", color="tomato", linestyle="--")
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
eval_df = pd.DataFrame({
    "day": range(1, 16),
    "mae": mae_per_day,
    "rmse": rmse_per_day,
})
eval_df.to_csv(OUTPUT_DIR / "day_metrics.csv", index=False)

summary = {
    "mae_mean": float(mae_per_day.mean()),
    "rmse_mean": float(rmse_per_day.mean()),
    "mean_corr": mean_corr,
    "mean_cos_sim": mean_cos_sim,
    "best_val_loss": best_val_loss,
    "epochs_trained": len(history["train_loss"]),
}
pd.Series(summary).to_json(OUTPUT_DIR / "summary.json")

print(f"\nresults -> {OUTPUT_DIR}")
