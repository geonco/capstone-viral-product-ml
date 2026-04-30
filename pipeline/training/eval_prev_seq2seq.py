# 이전 seq2seq 모델 분포 평가 — 정확한 비교용
# 이전 아키텍처(단방향 LSTM, single Bahdanau, absolute output) + 백업 데이터(4채널, 19피처)
# 평가만 수행, 학습 없음

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT

DEVICE = "cpu"
DATA_DIR = ROOT / "data" / "processed"

# 이전 백업 데이터 로드
X     = np.load(DATA_DIR / "lstm_X.npy.backup")        # (N, 60, 4)
feat  = np.load(DATA_DIR / "lstm_feat.npy.backup")     # (N, 19)
seq_y = np.load(DATA_DIR / "lstm_seq_y.npy.backup")    # (N, 15)
meta  = pd.read_csv(DATA_DIR / "lstm_meta.csv.backup")

print(f"X: {X.shape}  feat: {feat.shape}  seq_y: {seq_y.shape}")

# 동일한 시계열 분할
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
test_mask  = meta["date"] >= test_start

X_train, feat_train = X[train_mask], feat[train_mask]
X_test,  feat_test, y_test = X[test_mask], feat[test_mask], seq_y[test_mask]
meta_test = meta[test_mask].reset_index(drop=True)

# train 통계로 피처 정규화
feat_mean = feat_train.mean(axis=0)
feat_std  = feat_train.std(axis=0) + 1e-8
feat_test_n = (feat_test - feat_mean) / feat_std

print(f"test: {len(X_test):,}")


# 이전 아키텍처 — Bahdanau attention
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

    def forward(self, seq, feat):
        batch_size = seq.size(0)
        enc_outputs, (h, c) = self.encoder(seq)

        feat_vec = self.feat_proj(feat)
        h_mod    = h.clone()
        h_mod[-1] = h_mod[-1] + feat_vec

        dec_h = h_mod[-1]
        dec_c = c[-1]

        prev_value = torch.zeros(batch_size, 1, device=seq.device)
        outputs = []
        for t in range(self.out_steps):
            context, _ = self.attention(dec_h, enc_outputs)
            prev_proj = self.input_proj(prev_value)
            dec_in    = torch.cat([prev_proj, context], dim=1)
            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            pred = self.output_proj(self.dropout(dec_h))
            outputs.append(pred)
            prev_value = pred.detach()
        return torch.cat(outputs, dim=1)


# 이전 모델 가중치 로드
WEIGHTS = ROOT / "outputs" / "lstm_seq2seq_20260410_160735" / "seq2seq_best.pt"
model = Seq2SeqAttention(input_size=4, hidden_size=128, num_layers=2,
                         feat_size=19, out_steps=15, dropout=0.3).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE, weights_only=True))
model.eval()
print(f"loaded: {WEIGHTS}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")


# 추론
class SeqDataset(Dataset):
    def __init__(self, X, feat, y):
        self.X    = torch.tensor(X,    dtype=torch.float32)
        self.feat = torch.tensor(feat, dtype=torch.float32)
        self.y    = torch.tensor(y,    dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.feat[idx], self.y[idx]


loader = DataLoader(SeqDataset(X_test, feat_test_n, y_test),
                    batch_size=512, shuffle=False, num_workers=0)

all_preds = []
with torch.no_grad():
    for xb, fb, _ in loader:
        pred = model(xb.to(DEVICE), fb.to(DEVICE)).cpu().numpy()
        all_preds.append(pred)
preds_z = np.concatenate(all_preds, axis=0)

# 역변환
seq_y_mean = meta_test["seq_y_mean"].values[:, None]
seq_y_std  = meta_test["seq_y_std"].values[:, None]
preds_orig  = preds_z * seq_y_std + seq_y_mean
y_test_orig = y_test * seq_y_std + seq_y_mean

# 지표
mae_per_day  = np.abs(preds_orig - y_test_orig).mean(axis=0)
rmse_per_day = np.sqrt(((preds_orig - y_test_orig) ** 2).mean(axis=0))

corrs = []
for i in range(len(preds_orig)):
    if np.std(preds_orig[i]) > 1e-8 and np.std(y_test_orig[i]) > 1e-8:
        c = np.corrcoef(preds_orig[i], y_test_orig[i])[0, 1]
        corrs.append(c if not np.isnan(c) else 0.0)
    else:
        corrs.append(0.0)
corrs = np.array(corrs)
mean_corr = float(corrs.mean())

cos_sims = []
for i in range(len(preds_z)):
    p, t = preds_z[i], y_test[i]
    np_, nt = np.linalg.norm(p), np.linalg.norm(t)
    if np_ > 1e-8 and nt > 1e-8:
        cos_sims.append(float(np.dot(p, t) / (np_ * nt)))
    else:
        cos_sims.append(0.0)
mean_cos = float(np.mean(cos_sims))

ratio_good     = float((corrs >  0.7).mean())
ratio_decent   = float(((corrs >  0.3) & (corrs <= 0.7)).mean())
ratio_neutral  = float((np.abs(corrs) <= 0.3).mean())
ratio_opposite = float((corrs < -0.3).mean())

print(f"\n=== 이전 모델 (lstm_seq2seq_20260410_160735) 분포 평가 ===")
print(f"MAE_mean  : {mae_per_day.mean():.4f}")
print(f"RMSE_mean : {rmse_per_day.mean():.4f}")
print(f"mean_corr : {mean_corr:.4f}")
print(f"cos_sim   : {mean_cos:.4f}")
print(f"")
print(f"잘 맞춤 (r >  0.7) : {ratio_good*100:5.2f}%")
print(f"적당   (0.3~0.7)  : {ratio_decent*100:5.2f}%")
print(f"무관   (|r|<=0.3) : {ratio_neutral*100:5.2f}%")
print(f"반대   (r < -0.3) : {ratio_opposite*100:5.2f}%")

# 결과 저장
out = ROOT / "outputs" / "lstm_seq2seq_20260410_160735" / "summary_recomputed.json"
data = {
    "mae_mean":       float(mae_per_day.mean()),
    "rmse_mean":      float(rmse_per_day.mean()),
    "mean_corr":      mean_corr,
    "mean_cos_sim":   mean_cos,
    "ratio_good":     ratio_good,
    "ratio_decent":   ratio_decent,
    "ratio_neutral":  ratio_neutral,
    "ratio_opposite": ratio_opposite,
}
with open(out, "w") as f:
    json.dump(data, f, indent=2)
print(f"\nsaved -> {out}")
