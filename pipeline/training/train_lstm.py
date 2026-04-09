# LSTM 학습 스크립트 — peak_time 3-class 분류 (로컬 CPU 실행)

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT

from datetime import datetime

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR   = ROOT / 'data' / 'processed'
OUTPUT_DIR = ROOT / 'outputs' / f'lstm_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'device: {DEVICE}')

# 데이터 로드
X      = np.load(DATA_DIR / 'lstm_X.npy')
y      = np.load(DATA_DIR / 'lstm_y.npy')
labels = np.load(DATA_DIR / 'lstm_labels.npy')
meta   = pd.read_csv(DATA_DIR / 'lstm_meta.csv')

print(f'X: {X.shape}  y: {y.shape}  labels: {labels.shape}')
print(f'class dist  0={(y==0).sum()}  1={(y==1).sum()}  2={(y==2).sum()}')

# 시계열 기반 train/valid/test 분할 — LightGBM train.py와 동일 기준
meta['date'] = pd.to_datetime(meta['date'])
dates = np.sort(meta['date'].unique())
n = len(dates)

GAP = pd.Timedelta(days=75)
train_end   = dates[int(n * 0.70)]
valid_start = train_end + GAP
remaining   = dates[dates >= valid_start]
valid_end   = remaining[int(len(remaining) * 0.5)]
test_start  = valid_end + GAP

train_mask = meta['date'] <= train_end
valid_mask = (meta['date'] >= valid_start) & (meta['date'] <= valid_end)
test_mask  = meta['date'] >= test_start

X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[valid_mask], y[valid_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f'train: {len(X_train):,}  valid: {len(X_valid):,}  test: {len(X_test):,}')
print(f'train ~{pd.Timestamp(train_end).date()} | gap | valid {pd.Timestamp(valid_start).date()}~{pd.Timestamp(valid_end).date()} | gap | test {pd.Timestamp(test_start).date()}~')


# PyTorch Dataset
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


BATCH_SIZE = 512

# Windows에서 num_workers > 0 사용 시 멀티프로세싱 충돌 방지
train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valid_loader = DataLoader(SeqDataset(X_valid, y_valid), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(SeqDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# LSTM 모델
class ViralLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        # 시계열 인코더 — 60일 시퀀스에서 패턴 추출
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout,
        )
        # 분류기 — 마지막 hidden state로 3-class 예측
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        return self.classifier(last)


model = ViralLSTM().to(DEVICE)
print(f'params: {sum(p.numel() for p in model.parameters()):,}')

# 클래스 불균형 보정 — train 분포 기반 가중치
class_counts  = np.bincount(y_train)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
class_weights = class_weights / class_weights.sum() * len(class_counts)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# 학습 루프
EPOCHS     = 30
EARLY_STOP = 7
MODEL_PATH = OUTPUT_DIR / 'lstm_best.pt'

best_val_acc = 0.0
no_improve   = 0
history      = {'train_loss': [], 'val_loss': [], 'val_acc': []}


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
    return total_loss / total, correct / total


print('\n학습 시작')
print('=' * 60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(valid_loader, train=False)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f'[{epoch:02d}/{EPOCHS}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}', end='')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve   = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print('  *saved*')
    else:
        no_improve += 1
        print(f'  (no improve {no_improve}/{EARLY_STOP})')
        if no_improve >= EARLY_STOP:
            print('early stopping')
            break

print(f'\nbest val acc: {best_val_acc:.4f}')

# 학습 곡선 저장
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='train')
axes[0].plot(history['val_loss'],   label='valid')
axes[0].set_title('Loss')
axes[0].legend()
axes[1].plot(history['val_acc'], label='valid acc')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'train_curve.png', dpi=150)
plt.close()
print(f'train curve saved')

# 테스트셋 최종 평가
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)

test_acc      = (all_preds == all_true).mean()
class_mae     = np.abs(all_preds - all_true).mean()   # 클래스 거리 기반 MAE — 0=완벽, 1=한칸, 2=완전반대
class_mae_std = np.abs(all_preds - all_true).std()

print(f'\ntest accuracy : {test_acc:.4f}')
print(f'class MAE     : {class_mae:.4f}  (std={class_mae_std:.4f})')
print(f'  - 0.0: 모든 예측이 정확')
print(f'  - 1.0: 평균 한 구간 틀림 (5d 예측을 10d로)')
print(f'  - 2.0: 평균 두 구간 틀림 (5d 예측을 15d로)')
print('\n' + classification_report(all_true, all_preds, target_names=['<=5d', '6~10d', '11~15d']))

# Confusion Matrix 저장
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=5d', '6~10d', '11~15d'],
            yticklabels=['<=5d', '6~10d', '11~15d'])
plt.title(f'Confusion Matrix (test acc={test_acc:.4f})')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
plt.close()
print(f'confusion matrix saved')
print(f'\nresults -> {OUTPUT_DIR}')
