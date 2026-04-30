# 앙상블 평가 — 새 모델 + 이전 모델 예측을 평균하여 단일 모델 대비 성능 비교

import sys
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, LSTM_FEAT_COLS, LSTM_INPUT_CHANNELS

DEVICE = "cpu"
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "ensemble_seq2seq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 데이터 로드 — 새 데이터(6채널, 24피처) + 백업(4채널, 19피처) =====
X_new     = np.load(DATA_DIR / "lstm_X.npy")
feat_new  = np.load(DATA_DIR / "lstm_feat.npy")
seq_y     = np.load(DATA_DIR / "lstm_seq_y.npy")
meta      = pd.read_csv(DATA_DIR / "lstm_meta.csv")

X_old     = np.load(DATA_DIR / "lstm_X.npy.backup")
feat_old  = np.load(DATA_DIR / "lstm_feat.npy.backup")

print(f"new X: {X_new.shape}  feat: {feat_new.shape}")
print(f"old X: {X_old.shape}  feat: {feat_old.shape}")

# 동일 분할
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

# 새 모델 입력 (정규화)
feat_new_train = feat_new[train_mask]
feat_new_test  = feat_new[test_mask]
fn_mean = feat_new_train.mean(axis=0); fn_std = feat_new_train.std(axis=0) + 1e-8
feat_new_test_n = (feat_new_test - fn_mean) / fn_std

# 이전 모델 입력 (정규화)
feat_old_train = feat_old[train_mask]
feat_old_test  = feat_old[test_mask]
fo_mean = feat_old_train.mean(axis=0); fo_std = feat_old_train.std(axis=0) + 1e-8
feat_old_test_n = (feat_old_test - fo_mean) / fo_std

X_new_test = X_new[test_mask]
X_old_test = X_old[test_mask]
y_test     = seq_y[test_mask]
meta_test  = meta[test_mask].reset_index(drop=True)

print(f"test: {len(y_test):,}")


# ===== 새 모델 정의 (현재 학습본과 동일) =====
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, dec_hidden, enc_outputs):
        B = dec_hidden.size(0); T = enc_outputs.size(1); H = enc_outputs.size(2)
        q = self.q_proj(dec_hidden).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc_outputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc_outputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        w = F.softmax(s, dim=-1)
        ctx = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, H)
        return self.out_proj(ctx)


class Seq2SeqBiAttn(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, enc_layers=2,
                 feat_size=24, out_steps=15, num_heads=2, dropout=0.3):
        super().__init__()
        self.out_steps = out_steps; self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=enc_layers, batch_first=True,
                               bidirectional=True, dropout=dropout if enc_layers > 1 else 0.0)
        self.enc_out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_h_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_c_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.feat_proj    = nn.Sequential(nn.Linear(feat_size, hidden_size),
                                          nn.LayerNorm(hidden_size))
        self.attention    = MultiHeadAttention(hidden_size, num_heads=num_heads)
        self.input_proj   = nn.Linear(1, hidden_size)
        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.delta_head   = nn.Sequential(nn.Linear(hidden_size, 64), nn.LayerNorm(64),
                                          nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.dropout = nn.Dropout(dropout)

    def _merge(self, hc):
        return torch.cat([hc[-2], hc[-1]], dim=-1)

    def forward(self, seq, feat):
        B = seq.size(0)
        enc_out, (h, c) = self.encoder(seq)
        enc_out = self.enc_out_proj(enc_out)
        dec_h = self.enc_h_proj(self._merge(h)) + self.feat_proj(feat)
        dec_c = self.enc_c_proj(self._merge(c))
        prev = torch.zeros(B, 1, device=seq.device)
        outs = []
        for t in range(self.out_steps):
            ctx = self.attention(dec_h, enc_out)
            din = torch.cat([self.input_proj(prev), ctx], dim=1)
            dec_h, dec_c = self.decoder_cell(din, (dec_h, dec_c))
            delta = self.delta_head(self.dropout(dec_h))
            pred = prev + delta
            outs.append(pred)
            prev = pred.detach()
        return torch.cat(outs, dim=1)


# ===== 이전 모델 정의 =====
class OldAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        T = enc_outputs.size(1)
        de = dec_hidden.unsqueeze(1).expand(-1, T, -1)
        cb = torch.cat([de, enc_outputs], dim=2)
        s  = self.v(torch.tanh(self.W(cb))).squeeze(-1)
        w  = F.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), enc_outputs).squeeze(1)


class Seq2SeqOld(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2,
                 feat_size=19, out_steps=15, dropout=0.3):
        super().__init__()
        self.out_steps = out_steps
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout)
        self.feat_proj = nn.Linear(feat_size, hidden_size)
        self.attention = OldAttention(hidden_size)
        self.input_proj = nn.Linear(1, hidden_size)
        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.output_proj = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, feat):
        B = seq.size(0)
        enc_out, (h, c) = self.encoder(seq)
        feat_vec = self.feat_proj(feat)
        h_mod = h.clone(); h_mod[-1] = h_mod[-1] + feat_vec
        dec_h = h_mod[-1]; dec_c = c[-1]
        prev = torch.zeros(B, 1, device=seq.device)
        outs = []
        for t in range(self.out_steps):
            ctx = self.attention(dec_h, enc_out)
            din = torch.cat([self.input_proj(prev), ctx], dim=1)
            dec_h, dec_c = self.decoder_cell(din, (dec_h, dec_c))
            pred = self.output_proj(self.dropout(dec_h))
            outs.append(pred)
            prev = pred.detach()
        return torch.cat(outs, dim=1)


# ===== 모델 로드 =====
new_dirs = sorted([d for d in (ROOT / "outputs").glob("lstm_seq2seq_*")
                   if (d / "seq2seq_best.pt").exists()
                   and not d.name.endswith("_backup")])
NEW_PATH = new_dirs[-1] / "seq2seq_best.pt"
OLD_PATH = ROOT / "outputs" / "lstm_seq2seq_20260410_160735" / "seq2seq_best.pt"
print(f"new: {NEW_PATH}")
print(f"old: {OLD_PATH}")

new_model = Seq2SeqBiAttn(input_size=LSTM_INPUT_CHANNELS, hidden_size=128, enc_layers=2,
                          feat_size=len(LSTM_FEAT_COLS), out_steps=15,
                          num_heads=2, dropout=0.3).to(DEVICE)
new_model.load_state_dict(torch.load(NEW_PATH, map_location=DEVICE, weights_only=True))
new_model.eval()

old_model = Seq2SeqOld(input_size=4, hidden_size=128, num_layers=2,
                       feat_size=19, out_steps=15, dropout=0.3).to(DEVICE)
old_model.load_state_dict(torch.load(OLD_PATH, map_location=DEVICE, weights_only=True))
old_model.eval()


# ===== 추론 =====
def infer(model, X, feat):
    Xt = torch.tensor(X, dtype=torch.float32)
    Ft = torch.tensor(feat, dtype=torch.float32)
    out = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            xb = Xt[i:i+bs].to(DEVICE)
            fb = Ft[i:i+bs].to(DEVICE)
            out.append(model(xb, fb).cpu().numpy())
    return np.concatenate(out, axis=0)

print("\nrunning new model inference...")
preds_new = infer(new_model, X_new_test, feat_new_test_n)
print("running old model inference...")
preds_old = infer(old_model, X_old_test, feat_old_test_n)


# ===== 평가 함수 =====
seq_y_mean = meta_test["seq_y_mean"].values[:, None]
seq_y_std  = meta_test["seq_y_std"].values[:, None]
y_orig = y_test * seq_y_std + seq_y_mean


def evaluate(preds_z, label):
    preds_o = preds_z * seq_y_std + seq_y_mean
    mae = np.abs(preds_o - y_orig).mean()
    rmse = np.sqrt(((preds_o - y_orig) ** 2).mean())
    corrs = []
    for i in range(len(preds_o)):
        if np.std(preds_o[i]) > 1e-8 and np.std(y_orig[i]) > 1e-8:
            c = np.corrcoef(preds_o[i], y_orig[i])[0, 1]
            corrs.append(c if not np.isnan(c) else 0.0)
        else:
            corrs.append(0.0)
    corrs = np.array(corrs)
    cos = []
    for i in range(len(preds_z)):
        p, t = preds_z[i], y_test[i]
        np_, nt = np.linalg.norm(p), np.linalg.norm(t)
        cos.append(float(np.dot(p, t) / (np_ * nt + 1e-12)) if np_ > 1e-8 and nt > 1e-8 else 0.0)
    return {
        "label":          label,
        "mae":            float(mae),
        "rmse":           float(rmse),
        "mean_corr":      float(corrs.mean()),
        "cos_sim":        float(np.mean(cos)),
        "ratio_good":     float((corrs >  0.7).mean()),
        "ratio_decent":   float(((corrs > 0.3) & (corrs <= 0.7)).mean()),
        "ratio_neutral":  float((np.abs(corrs) <= 0.3).mean()),
        "ratio_opposite": float((corrs < -0.3).mean()),
    }


# 단일 + 다양한 가중평균 시도
results = []
results.append(evaluate(preds_new, "new only"))
results.append(evaluate(preds_old, "old only"))
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blended = w * preds_new + (1 - w) * preds_old
    results.append(evaluate(blended, f"ensemble new={w:.1f}"))

print(f"\n{'='*100}")
print(f"{'label':22s}  {'MAE':>8s}  {'corr':>7s}  {'cos':>7s}  {'good%':>6s}  {'decent%':>7s}  {'neutral%':>8s}  {'opposite%':>9s}")
print(f"{'='*100}")
for r in results:
    print(f"{r['label']:22s}  {r['mae']:8.2f}  {r['mean_corr']:7.4f}  {r['cos_sim']:7.4f}  "
          f"{r['ratio_good']*100:5.2f}  {r['ratio_decent']*100:7.2f}  "
          f"{r['ratio_neutral']*100:8.2f}  {r['ratio_opposite']*100:9.2f}")
print(f"{'='*100}")

with open(OUTPUT_DIR / "ensemble_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {OUTPUT_DIR / 'ensemble_results.json'}")
