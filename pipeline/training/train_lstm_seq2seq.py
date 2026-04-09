# seq2seq LSTM 학습 스크립트 — 60일 시계열 입력 → 미래 15일 궤적 예측
# 출력값은 log1p 변환으로 극단값 처리, 평가는 원래 스케일로 복원

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR   = ROOT / 'data' / 'processed'
OUTPUT_DIR = ROOT / 'outputs' / f'lstm_seq2seq_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'device: {DEVICE}')

# 데이터 로드
X     = np.load(DATA_DIR / 'lstm_X.npy')        # (N, 60, 4)
seq_y = np.load(DATA_DIR / 'lstm_seq_y.npy')    # (N, 15)  정규화된 4채널 합산
meta  = pd.read_csv(DATA_DIR / 'lstm_meta.csv')

print(f'X: {X.shape}  seq_y: {seq_y.shape}')

# 극단값 처리 — log1p 변환 (max 5777 → ~8.7 로 압축)
seq_y_log = np.log1p(seq_y).astype(np.float32)
print(f'seq_y_log  mean={seq_y_log.mean():.4f}  std={seq_y_log.std():.4f}  max={seq_y_log.max():.4f}')

# 시계열 기반 train/valid/test 분할 — LightGBM, train_lstm.py와 동일 기준
meta['date'] = pd.to_datetime(meta['date'])
dates = np.sort(meta['date'].unique())
n     = len(dates)

GAP = pd.Timedelta(days=75)
train_end   = dates[int(n * 0.70)]
valid_start = train_end + GAP
remaining   = dates[dates >= valid_start]
valid_end   = remaining[int(len(remaining) * 0.5)]
test_start  = valid_end + GAP

train_mask = meta['date'] <= train_end
valid_mask = (meta['date'] >= valid_start) & (meta['date'] <= valid_end)
test_mask  = meta['date'] >= test_start

X_train, y_train = X[train_mask], seq_y_log[train_mask]
X_valid, y_valid = X[valid_mask], seq_y_log[valid_mask]
X_test,  y_test  = X[test_mask],  seq_y_log[test_mask]
seq_y_test_orig  = seq_y[test_mask]   # 원래 스케일 — 최종 평가용

print(f'train: {len(X_train):,}  valid: {len(X_valid):,}  test: {len(X_test):,}')
print(f'train ~{pd.Timestamp(train_end).date()} | gap | valid {pd.Timestamp(valid_start).date()}~{pd.Timestamp(valid_end).date()} | gap | test {pd.Timestamp(test_start).date()}~')


# PyTorch Dataset
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


BATCH_SIZE = 512

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(SeqDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(SeqDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# seq2seq LSTM 모델
# 인코더: 60일 4채널 시퀀스 → hidden state 압축
# 디코더: hidden state를 15번 펼쳐서 15일 궤적 생성
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, out_steps=15, dropout=0.3):
        super().__init__()
        self.out_steps   = out_steps
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # 인코더 — 과거 60일 패턴을 hidden state로 압축
        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout,
        )
        # 디코더 — hidden state를 받아 15일 시퀀스 생성
        self.decoder = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout,
        )
        # 출력 레이어 — 각 타임스텝을 값 하나로 변환
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, 60, 4)
        _, (h, c) = self.encoder(x)                         # h: (layers, batch, hidden)

        # 인코더 마지막 hidden을 15번 반복해 디코더 입력으로 사용
        last_h   = h[-1]                                    # (batch, hidden)
        dec_in   = last_h.unsqueeze(1).repeat(1, self.out_steps, 1)  # (batch, 15, hidden)

        dec_out, _ = self.decoder(dec_in, (h, c))           # (batch, 15, hidden)
        out        = self.fc(dec_out).squeeze(-1)           # (batch, 15)
        return out


model = Seq2SeqLSTM().to(DEVICE)
print(f'params: {sum(p.numel() for p in model.parameters()):,}')

# Huber Loss — MSE보다 극단값에 강건
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# 학습 루프
EPOCHS     = 30
EARLY_STOP = 7
MODEL_PATH = OUTPUT_DIR / 'seq2seq_best.pt'

best_val_loss = float('inf')
no_improve    = 0
history       = {'train_loss': [], 'val_loss': []}


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, total = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred   = model(xb)
            loss   = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            total      += len(yb)
    return total_loss / total


print('\n학습 시작')
print('=' * 60)

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(valid_loader, train=False)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f'[{epoch:02d}/{EPOCHS}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}', end='')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve    = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print('  *saved*')
    else:
        no_improve += 1
        print(f'  (no improve {no_improve}/{EARLY_STOP})')
        if no_improve >= EARLY_STOP:
            print('early stopping')
            break

print(f'\nbest val loss: {best_val_loss:.4f}')

# 학습 곡선 저장
plt.figure(figsize=(8, 4))
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'],   label='valid')
plt.title('Huber Loss')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'train_curve.png', dpi=150)
plt.close()

# 테스트셋 최종 평가
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        pred = model(xb.to(DEVICE)).cpu().numpy()
        all_preds.append(pred)

preds_log  = np.concatenate(all_preds, axis=0)        # (N, 15) log 스케일
preds_orig = np.expm1(preds_log).clip(0)              # 원래 스케일 복원

# 평가 지표 — 원래 스케일 기준
mae_per_day  = np.abs(preds_orig - seq_y_test_orig).mean(axis=0)
rmse_per_day = np.sqrt(((preds_orig - seq_y_test_orig) ** 2).mean(axis=0))
mae_total    = float(mae_per_day.mean())
rmse_total   = float(rmse_per_day.mean())

# 궤적 형태 유사도 — 예측 vs 실제 곡선의 Pearson 상관
correlations = [
    np.corrcoef(preds_orig[i], seq_y_test_orig[i])[0, 1]
    for i in range(len(preds_orig))
]
mean_corr = float(np.nanmean(correlations))

print(f'\n--- 테스트셋 평가 (원래 스케일) ---')
print(f'MAE (평균)        : {mae_total:.4f}')
print(f'RMSE (평균)       : {rmse_total:.4f}')
print(f'궤적 상관계수 평균 : {mean_corr:.4f}  (1=완벽, 0=무관계)')

# day별 MAE 출력
print('\nday별 MAE:')
for d, mae in enumerate(mae_per_day, 1):
    bar = '█' * int(mae * 5 / mae_per_day.max() * 20)
    print(f'  day{d:2d}: {mae:.4f}  {bar}')

# 예측 궤적 시각화 — 랜덤 샘플 6개
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
sample_idx = np.random.choice(len(preds_orig), 6, replace=False)
for i, idx in enumerate(sample_idx):
    axes[i].plot(seq_y_test_orig[idx], label='actual',    color='steelblue')
    axes[i].plot(preds_orig[idx],      label='predicted', color='tomato', linestyle='--')
    corr = np.corrcoef(preds_orig[idx], seq_y_test_orig[idx])[0, 1]
    axes[i].set_title(f'sample {idx}  corr={corr:.2f}')
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel('day')
plt.suptitle(f'Predicted vs Actual Trajectory  (mean_corr={mean_corr:.3f})')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trajectory_samples.png', dpi=150)
plt.close()
print(f'\nresults -> {OUTPUT_DIR}')
