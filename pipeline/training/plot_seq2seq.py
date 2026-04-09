# seq2seq 결과 시각화 — trajectory_samples.png, day_mae.png 생성

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT
import torch.nn as nn

# Seq2SeqLSTM 직접 정의 — train_lstm_seq2seq 임포트 시 학습 코드 실행 방지
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, out_steps=15, dropout=0.3):
        super().__init__()
        self.out_steps   = out_steps
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 64),
                                nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        last_h    = h[-1]
        dec_in    = last_h.unsqueeze(1).repeat(1, self.out_steps, 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.fc(dec_out).squeeze(-1)

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = ROOT / 'data' / 'processed'

# 가장 최근 seq2seq 출력 폴더
seq_dirs = sorted(glob.glob(str(ROOT / 'outputs' / 'lstm_seq2seq_*')))
OUT_DIR  = Path(seq_dirs[-1])
print(f'output dir: {OUT_DIR}')

# 데이터 로드
X     = np.load(DATA_DIR / 'lstm_X.npy')
seq_y = np.load(DATA_DIR / 'lstm_seq_y.npy')
meta  = pd.read_csv(DATA_DIR / 'lstm_meta.csv')

seq_y_log = np.log1p(seq_y).astype(np.float32)

meta['date'] = pd.to_datetime(meta['date'])
dates = np.sort(meta['date'].unique())
n     = len(dates)
GAP   = pd.Timedelta(days=75)
train_end  = dates[int(n * 0.70)]
valid_start = train_end + GAP
remaining   = dates[dates >= valid_start]
valid_end   = remaining[int(len(remaining) * 0.5)]
test_start  = valid_end + GAP
test_mask   = meta['date'] >= test_start

X_test          = X[test_mask]
seq_y_test_orig = seq_y[test_mask]
seq_y_test_log  = seq_y_log[test_mask]

# 모델 로드
model = Seq2SeqLSTM().to(DEVICE)
model.load_state_dict(torch.load(OUT_DIR / 'seq2seq_best.pt', map_location=DEVICE))
model.eval()

# 예측
from torch.utils.data import DataLoader, TensorDataset
loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
    batch_size=512, shuffle=False
)
all_preds = []
with torch.no_grad():
    for (xb,) in loader:
        pred = model(xb.to(DEVICE)).cpu().numpy()
        all_preds.append(pred)

preds_log  = np.concatenate(all_preds, axis=0)
preds_orig = np.expm1(preds_log).clip(0)

# day별 MAE
mae_per_day  = np.abs(preds_orig - seq_y_test_orig).mean(axis=0)
rmse_per_day = np.sqrt(((preds_orig - seq_y_test_orig) ** 2).mean(axis=0))

print('day별 MAE:')
for d, mae in enumerate(mae_per_day, 1):
    print(f'  day{d:2d}: {mae:.4f}')

# 그래프 1 — day별 MAE/RMSE
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, 16), mae_per_day,  marker='o', label='MAE')
ax.plot(range(1, 16), rmse_per_day, marker='s', label='RMSE')
ax.set_xlabel('Forecast Day')
ax.set_ylabel('Error')
ax.set_title('Error by Forecast Day')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'day_mae.png', dpi=150)
plt.close()
print('saved: day_mae.png')

# 그래프 2 — 랜덤 샘플 6개 궤적 비교
correlations = [
    np.corrcoef(preds_orig[i], seq_y_test_orig[i])[0, 1]
    for i in range(len(preds_orig))
]
mean_corr = float(np.nanmean(correlations))

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
np.random.seed(42)
sample_idx = np.random.choice(len(preds_orig), 6, replace=False)
for i, idx in enumerate(sample_idx):
    axes[i].plot(seq_y_test_orig[idx], label='actual',    color='steelblue')
    axes[i].plot(preds_orig[idx],      label='predicted', color='tomato', linestyle='--')
    corr = np.corrcoef(preds_orig[idx], seq_y_test_orig[idx])[0, 1]
    kw   = meta[test_mask].iloc[idx]['keyword']
    axes[i].set_title(f'{kw}  corr={corr:.2f}')
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel('day')
plt.suptitle(f'Predicted vs Actual Trajectory  (mean corr={mean_corr:.3f})')
plt.tight_layout()
plt.savefig(OUT_DIR / 'trajectory_samples.png', dpi=150)
plt.close()
print('saved: trajectory_samples.png')

print(f'\nresults -> {OUT_DIR}')
