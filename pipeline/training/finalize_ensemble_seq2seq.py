# 앙상블 최종 figure 생성 — trajectory samples, day MAE, weight sweep, ablation
# 0.5/0.5 가중치를 최종으로 확정하고 포스터/논문용 그림 일괄 생성

import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, LSTM_FEAT_COLS, LSTM_INPUT_CHANNELS

DEVICE = "cpu"
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "ensemble_seq2seq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 데이터 로드 =====
X_new    = np.load(DATA_DIR / "lstm_X.npy")
feat_new = np.load(DATA_DIR / "lstm_feat.npy")
seq_y    = np.load(DATA_DIR / "lstm_seq_y.npy")
meta     = pd.read_csv(DATA_DIR / "lstm_meta.csv")
X_old    = np.load(DATA_DIR / "lstm_X.npy.backup")
feat_old = np.load(DATA_DIR / "lstm_feat.npy.backup")

meta["date"] = pd.to_datetime(meta["date"])
dates = np.sort(meta["date"].unique())
n = len(dates)
GAP = pd.Timedelta(days=75)
train_end = dates[int(n * 0.70)]
remaining = dates[dates >= (train_end + GAP)]
valid_end = remaining[int(len(remaining) * 0.5)]
test_start = valid_end + GAP
train_mask = meta["date"] <= train_end
test_mask  = meta["date"] >= test_start

# 정규화
fn_mean = feat_new[train_mask].mean(axis=0); fn_std = feat_new[train_mask].std(axis=0) + 1e-8
fo_mean = feat_old[train_mask].mean(axis=0); fo_std = feat_old[train_mask].std(axis=0) + 1e-8
feat_new_test_n = (feat_new[test_mask] - fn_mean) / fn_std
feat_old_test_n = (feat_old[test_mask] - fo_mean) / fo_std

X_new_test = X_new[test_mask]
X_old_test = X_old[test_mask]
y_test     = seq_y[test_mask]
meta_test  = meta[test_mask].reset_index(drop=True)


# ===== 모델 클래스 (eval_ensemble_seq2seq와 동일) =====
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=2):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = hidden_size // num_heads
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
        self.out_steps = out_steps
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=enc_layers, batch_first=True,
                               bidirectional=True, dropout=dropout if enc_layers > 1 else 0.0)
        self.enc_out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_h_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_c_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.feat_proj = nn.Sequential(nn.Linear(feat_size, hidden_size), nn.LayerNorm(hidden_size))
        self.attention = MultiHeadAttention(hidden_size, num_heads=num_heads)
        self.input_proj = nn.Linear(1, hidden_size)
        self.decoder_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.delta_head = nn.Sequential(nn.Linear(hidden_size, 64), nn.LayerNorm(64),
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
            pred = prev + self.delta_head(self.dropout(dec_h))
            outs.append(pred); prev = pred.detach()
        return torch.cat(outs, dim=1)


class OldAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        T = enc_outputs.size(1)
        de = dec_hidden.unsqueeze(1).expand(-1, T, -1)
        cb = torch.cat([de, enc_outputs], dim=2)
        s = self.v(torch.tanh(self.W(cb))).squeeze(-1)
        w = F.softmax(s, dim=1)
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
        h_mod = h.clone(); h_mod[-1] = h_mod[-1] + self.feat_proj(feat)
        dec_h = h_mod[-1]; dec_c = c[-1]
        prev = torch.zeros(B, 1, device=seq.device)
        outs = []
        for t in range(self.out_steps):
            ctx = self.attention(dec_h, enc_out)
            din = torch.cat([self.input_proj(prev), ctx], dim=1)
            dec_h, dec_c = self.decoder_cell(din, (dec_h, dec_c))
            pred = self.output_proj(self.dropout(dec_h))
            outs.append(pred); prev = pred.detach()
        return torch.cat(outs, dim=1)


# 모델 로드
new_dirs = sorted([d for d in (ROOT / "outputs").glob("lstm_seq2seq_*")
                   if (d / "seq2seq_best.pt").exists() and not d.name.endswith("_backup")])
NEW_PATH = new_dirs[-1] / "seq2seq_best.pt"
OLD_PATH = ROOT / "outputs" / "lstm_seq2seq_20260410_160735" / "seq2seq_best.pt"

new_model = Seq2SeqBiAttn(input_size=LSTM_INPUT_CHANNELS, feat_size=len(LSTM_FEAT_COLS)).to(DEVICE)
new_model.load_state_dict(torch.load(NEW_PATH, map_location=DEVICE, weights_only=True))
new_model.eval()
old_model = Seq2SeqOld().to(DEVICE)
old_model.load_state_dict(torch.load(OLD_PATH, map_location=DEVICE, weights_only=True))
old_model.eval()


def infer(model, X, feat, bs=512):
    Xt = torch.tensor(X, dtype=torch.float32); Ft = torch.tensor(feat, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            out.append(model(Xt[i:i+bs].to(DEVICE), Ft[i:i+bs].to(DEVICE)).cpu().numpy())
    return np.concatenate(out, axis=0)


print("inference: new model"); preds_new = infer(new_model, X_new_test, feat_new_test_n)
print("inference: old model"); preds_old = infer(old_model, X_old_test, feat_old_test_n)

# 0.5/0.5 앙상블
preds_ens = 0.5 * preds_new + 0.5 * preds_old

# 역변환
seq_y_mean = meta_test["seq_y_mean"].values[:, None]
seq_y_std  = meta_test["seq_y_std"].values[:, None]
def to_orig(z): return z * seq_y_std + seq_y_mean
y_orig = to_orig(y_test)
new_orig = to_orig(preds_new)
old_orig = to_orig(preds_old)
ens_orig = to_orig(preds_ens)


# ===== 평가 =====
def metrics(preds_o, preds_z):
    mae = np.abs(preds_o - y_orig).mean()
    rmse = np.sqrt(((preds_o - y_orig) ** 2).mean())
    corrs = np.zeros(len(preds_o))
    for i in range(len(preds_o)):
        if np.std(preds_o[i]) > 1e-8 and np.std(y_orig[i]) > 1e-8:
            c = np.corrcoef(preds_o[i], y_orig[i])[0, 1]
            corrs[i] = 0.0 if np.isnan(c) else c
    cos = np.zeros(len(preds_z))
    for i in range(len(preds_z)):
        np_ = np.linalg.norm(preds_z[i]); nt = np.linalg.norm(y_test[i])
        if np_ > 1e-8 and nt > 1e-8:
            cos[i] = float(np.dot(preds_z[i], y_test[i]) / (np_ * nt))
    return {
        "mae": float(mae), "rmse": float(rmse),
        "mean_corr": float(corrs.mean()), "cos_sim": float(cos.mean()),
        "ratio_good": float((corrs > 0.7).mean()),
        "ratio_decent": float(((corrs > 0.3) & (corrs <= 0.7)).mean()),
        "ratio_neutral": float((np.abs(corrs) <= 0.3).mean()),
        "ratio_opposite": float((corrs < -0.3).mean()),
    }, corrs

m_new, c_new = metrics(new_orig, preds_new)
m_old, c_old = metrics(old_orig, preds_old)
m_ens, c_ens = metrics(ens_orig, preds_ens)

# ===== final_summary.json =====
with open(OUTPUT_DIR / "final_summary.json", "w") as f:
    json.dump({"weight_new": 0.5, "weight_old": 0.5,
               "ensemble": m_ens, "new_only": m_new, "old_only": m_old}, f, indent=2)

# ===== day별 MAE =====
mae_day_ens  = np.abs(ens_orig - y_orig).mean(axis=0)
rmse_day_ens = np.sqrt(((ens_orig - y_orig) ** 2).mean(axis=0))
mae_day_new  = np.abs(new_orig - y_orig).mean(axis=0)
mae_day_old  = np.abs(old_orig - y_orig).mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 16), mae_day_new, marker="o", label="new only",   alpha=0.6)
ax.plot(range(1, 16), mae_day_old, marker="s", label="old only",   alpha=0.6)
ax.plot(range(1, 16), mae_day_ens, marker="D", label="ensemble 0.5", linewidth=2)
ax.set_xlabel("Forecast Day"); ax.set_ylabel("MAE")
ax.set_title("Day-wise MAE")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "day_mae.png", dpi=150); plt.close()

pd.DataFrame({"day": range(1, 16),
              "mae_ensemble": mae_day_ens, "rmse_ensemble": rmse_day_ens,
              "mae_new": mae_day_new, "mae_old": mae_day_old}).to_csv(
    OUTPUT_DIR / "day_metrics.csv", index=False)

# ===== trajectory samples — 앙상블 best/worst 6개 =====
sorted_idx = np.argsort(c_ens)
sample_idx = np.concatenate([sorted_idx[-3:], sorted_idx[:3]])

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, idx in enumerate(sample_idx):
    ax = axes[i]
    ax.plot(y_orig[idx], label="actual", color="steelblue", linewidth=2)
    ax.plot(ens_orig[idx], label="ensemble", color="tomato", linestyle="--", linewidth=1.5)
    kw = meta_test.iloc[idx]["keyword"]
    tag = "BEST" if i < 3 else "WORST"
    ax.set_title(f"[{tag}] {kw}  r={c_ens[idx]:.3f}", fontsize=9)
    ax.legend(fontsize=7); ax.set_xlabel("day")
plt.suptitle(f"Ensemble Trajectory (mean_corr={m_ens['mean_corr']:.3f}, cos_sim={m_ens['cos_sim']:.3f})")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "trajectory_samples.png", dpi=150); plt.close()

# ===== ablation — 같은 키워드에서 new/old/ensemble 비교 4개 =====
# 앙상블이 단일들보다 잘 맞춘 케이스 위주로 선택
diff = c_ens - np.maximum(c_new, c_old)
top_improved = np.argsort(diff)[-4:]

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.flatten()
for i, idx in enumerate(top_improved):
    ax = axes[i]
    ax.plot(y_orig[idx], label="actual", color="black", linewidth=2)
    ax.plot(new_orig[idx], label=f"new (r={c_new[idx]:.2f})", color="tab:blue", linestyle="--", alpha=0.7)
    ax.plot(old_orig[idx], label=f"old (r={c_old[idx]:.2f})", color="tab:green", linestyle="--", alpha=0.7)
    ax.plot(ens_orig[idx], label=f"ensemble (r={c_ens[idx]:.2f})", color="tomato", linewidth=2)
    kw = meta_test.iloc[idx]["keyword"]
    ax.set_title(f"{kw}", fontsize=10)
    ax.legend(fontsize=7); ax.set_xlabel("day")
plt.suptitle("Ablation — Ensemble vs Individual Models (improved cases)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "ablation_samples.png", dpi=150); plt.close()

# ===== weight sweep 그래프 =====
weights = np.linspace(0, 1, 11)
sweep_corr = []; sweep_mae = []
for w in weights:
    p = w * preds_new + (1 - w) * preds_old
    p_o = to_orig(p)
    cs = []
    for i in range(len(p_o)):
        if np.std(p_o[i]) > 1e-8 and np.std(y_orig[i]) > 1e-8:
            c = np.corrcoef(p_o[i], y_orig[i])[0, 1]
            cs.append(0.0 if np.isnan(c) else c)
        else:
            cs.append(0.0)
    sweep_corr.append(np.mean(cs))
    sweep_mae.append(np.abs(p_o - y_orig).mean())

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(weights, sweep_corr, "o-", color="tab:blue", label="mean_corr")
ax1.set_xlabel("weight on new model (1-w on old)")
ax1.set_ylabel("mean_corr", color="tab:blue")
ax1.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(weights, sweep_mae, "s-", color="tab:red", label="MAE")
ax2.set_ylabel("MAE", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax1.set_title("Ensemble Weight Sweep (selected: 0.5/0.5)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "weight_sweep.png", dpi=150); plt.close()

# ===== 콘솔 출력 =====
print(f"\n=== 최종 앙상블 0.5/0.5 ===")
for k, v in m_ens.items():
    print(f"  {k:15s}: {v:.4f}")
print(f"\nfigures saved to {OUTPUT_DIR}")
for f in ["final_summary.json", "trajectory_samples.png", "day_mae.png",
          "ablation_samples.png", "weight_sweep.png", "day_metrics.csv"]:
    print(f"  - {f}")
