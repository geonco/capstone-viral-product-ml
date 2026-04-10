# seq2seq 결과 시각화 — trajectory_samples.png, day_mae.png 생성
# train_lstm_seq2seq.py의 Seq2SeqAttention 모델을 로드해 test set에서 예측 수행

import sys
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, LSTM_FEAT_COLS


# Bahdanau Attention — train_lstm_seq2seq.py와 동일한 구조
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        seq_len    = enc_outputs.size(1)
        dec_expand = dec_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        combined   = torch.cat([dec_expand, enc_outputs], dim=2)
        energy     = torch.tanh(self.W(combined))
        scores     = self.v(energy).squeeze(-1)
        weights    = F.softmax(scores, dim=1)
        context    = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights


# Seq2SeqAttention — train_lstm_seq2seq.py와 동일한 구조 (학습 코드 실행 방지용 재정의)
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2,
                 feat_size=19, out_steps=15, dropout=0.3):
        super().__init__()
        self.out_steps   = out_steps
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        self.feat_proj = nn.Linear(feat_size, hidden_size)

        self.attention    = Attention(hidden_size)
        self.input_proj   = nn.Linear(1, hidden_size)
        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.output_proj  = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, feat, target=None, tf_ratio=0.0):
        batch_size = seq.size(0)

        enc_outputs, (h, c) = self.encoder(seq)

        h_mod     = h.clone()
        feat_vec  = self.feat_proj(feat)
        h_mod[-1] = h_mod[-1] + feat_vec

        dec_h = h_mod[-1]
        dec_c = c[-1]

        prev_value = torch.zeros(batch_size, 1, device=seq.device)

        outputs = []
        for t in range(self.out_steps):
            context, _ = self.attention(dec_h, enc_outputs)
            prev_proj  = self.input_proj(prev_value)
            dec_in     = torch.cat([prev_proj, context], dim=1)
            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            pred = self.output_proj(self.dropout(dec_h))
            outputs.append(pred)

            if target is not None and random.random() < tf_ratio:
                prev_value = target[:, t:t+1]
            else:
                prev_value = pred.detach()

        return torch.cat(outputs, dim=1)


DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"

# 가장 최근 seq2seq 출력 폴더
seq_dirs = sorted(glob.glob(str(ROOT / "outputs" / "lstm_seq2seq_*")))
if not seq_dirs:
    print("seq2seq 출력 폴더 없음 — train_lstm_seq2seq.py 먼저 실행 필요")
    sys.exit(1)
OUT_DIR = Path(seq_dirs[-1])
print(f"output dir: {OUT_DIR}")

# 데이터 로드
X     = np.load(DATA_DIR / "lstm_X.npy")
feat  = np.load(DATA_DIR / "lstm_feat.npy")
seq_y = np.load(DATA_DIR / "lstm_seq_y.npy")
meta  = pd.read_csv(DATA_DIR / "lstm_meta.csv")

# 시계열 기반 분할 — train_lstm_seq2seq.py와 동일한 로직
meta["date"] = pd.to_datetime(meta["date"])
dates = np.sort(meta["date"].unique())
n     = len(dates)
GAP   = pd.Timedelta(days=75)
train_end   = dates[int(n * 0.70)]
valid_start = train_end + GAP
remaining   = dates[dates >= valid_start]
valid_end   = remaining[int(len(remaining) * 0.5)]
test_start  = valid_end + GAP
test_mask   = meta["date"] >= test_start

X_test      = X[test_mask]
feat_test   = feat[test_mask]
y_test_z    = seq_y[test_mask]
meta_test   = meta[test_mask].reset_index(drop=True)

# 피처 정규화 — train 통계 기준
train_mask  = meta["date"] <= train_end
feat_train  = feat[train_mask]
feat_mean   = feat_train.mean(axis=0)
feat_std    = feat_train.std(axis=0) + 1e-8
feat_test_n = (feat_test - feat_mean) / feat_std

print(f"test samples: {len(X_test):,}")

# 모델 로드
N_FEAT = len(LSTM_FEAT_COLS)
model  = Seq2SeqAttention(feat_size=N_FEAT).to(DEVICE)
model.load_state_dict(torch.load(OUT_DIR / "seq2seq_best.pt", map_location=DEVICE, weights_only=True))
model.eval()

# 예측
loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(feat_test_n, dtype=torch.float32),
    ),
    batch_size=512, shuffle=False,
)

all_preds = []
with torch.no_grad():
    for xb, fb in loader:
        pred = model(xb.to(DEVICE), fb.to(DEVICE), target=None, tf_ratio=0.0)
        all_preds.append(pred.cpu().numpy())

preds_z = np.concatenate(all_preds, axis=0)

# 원래 스케일 역변환 — 메타에 저장된 mean/std 사용
seq_y_mean  = meta_test["seq_y_mean"].values[:, None]
seq_y_std   = meta_test["seq_y_std"].values[:, None]
preds_orig  = preds_z * seq_y_std + seq_y_mean
y_test_orig = y_test_z * seq_y_std + seq_y_mean

# day별 MAE / RMSE
mae_per_day  = np.abs(preds_orig - y_test_orig).mean(axis=0)
rmse_per_day = np.sqrt(((preds_orig - y_test_orig) ** 2).mean(axis=0))

print("\nday별 MAE:")
for d, (mae, rmse) in enumerate(zip(mae_per_day, rmse_per_day), 1):
    print(f"  day{d:2d}: MAE={mae:.4f}  RMSE={rmse:.4f}")

# 궤적 상관계수
correlations = []
for i in range(len(preds_orig)):
    if np.std(preds_orig[i]) > 1e-8 and np.std(y_test_orig[i]) > 1e-8:
        c = np.corrcoef(preds_orig[i], y_test_orig[i])[0, 1]
        correlations.append(c if not np.isnan(c) else 0.0)
    else:
        correlations.append(0.0)
mean_corr = float(np.mean(correlations))

# cosine similarity (z-score 스케일)
cos_sims = []
for i in range(len(preds_z)):
    p, t = preds_z[i], y_test_z[i]
    norm_p, norm_t = np.linalg.norm(p), np.linalg.norm(t)
    if norm_p > 1e-8 and norm_t > 1e-8:
        cos_sims.append(float(np.dot(p, t) / (norm_p * norm_t)))
    else:
        cos_sims.append(0.0)
mean_cos_sim = float(np.mean(cos_sims))

print(f"\n궤적 상관계수 (원래 스케일): {mean_corr:.4f}")
print(f"cosine similarity (z-score): {mean_cos_sim:.4f}")

# 그래프 1 — day별 MAE/RMSE
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 16), mae_per_day,  marker="o", label="MAE")
ax.plot(range(1, 16), rmse_per_day, marker="s", label="RMSE")
ax.set_xlabel("Forecast Day")
ax.set_ylabel("Error (original scale)")
ax.set_title("Error by Forecast Day")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "day_mae.png", dpi=150)
plt.close()
print("saved: day_mae.png")

# 그래프 2 — 상관 상위 3개 + 하위 3개 궤적
corr_arr   = np.array(correlations)
sorted_idx = np.argsort(corr_arr)

sample_indices = np.concatenate([
    sorted_idx[-3:],    # best 3
    sorted_idx[:3],     # worst 3
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
plt.savefig(OUT_DIR / "trajectory_samples.png", dpi=150)
plt.close()
print("saved: trajectory_samples.png")

print(f"\nresults -> {OUT_DIR}")
