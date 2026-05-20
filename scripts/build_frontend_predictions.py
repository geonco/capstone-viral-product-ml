# 프론트엔드용 실모델 추론 — LightGBM 15 + LSTM 11 + seq2seq 미래궤적
# dataset_final.csv 마지막 시점 row와 lstm_X/feat/meta의 키워드별 최신 샘플을 사용해
# outputs/* 학습 결과 모델들을 모두 로드하고 추론, 프론트가 읽는 JSON으로 저장

import argparse
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    FEAT_COLS, LSTM_FEAT_COLS, LSTM_INPUT_CHANNELS,
    LSTM_LABEL_COLS, LSTM_LABEL_GROUPS,
)
from pipeline.training.train_all import (
    BUCKET_CONFIG, BUCKET_REG_OVERRIDE, SUST_EXTRA_FEATS, _add_sust_features,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 — 추론 입력과 학습 결과 모델 위치
DATA_DIR = ROOT / "data" / "processed"
RAW      = ROOT / "data" / "raw"
OUTPUTS  = ROOT / "outputs"
OUT_FRONT = ROOT / "frontend" / "public" / "mock"

DATASET_CSV = ROOT / "data_processed" / "dataset.csv"  # build_dataset.py 출력 — 316 피처
LSTM_CKPT   = OUTPUTS / "lstm_20260430_123657" / "lstm_best.pt"
S2S_CKPT    = OUTPUTS / "lstm_seq2seq_20260429_221146" / "seq2seq_best.pt"

SKIP_TARGETS: set[str] = set()

# 타겟별 LightGBM 모델 디렉토리
LGBM_DIRS = {
    "sustainability_5d":  OUTPUTS / "sustainability_5d"  / "models",
    "sustainability_10d": OUTPUTS / "sustainability_10d" / "models",
    "sustainability_15d": OUTPUTS / "sustainability_15d" / "models",
    "crash_5d":  OUTPUTS / "crash_5d"  / "models",
    "crash_10d": OUTPUTS / "crash_10d" / "models",
    "crash_15d": OUTPUTS / "crash_15d" / "models",
    "spike_5d":  OUTPUTS / "spike_5d"  / "models",
    "spike_10d": OUTPUTS / "spike_10d" / "models",
    "spike_15d": OUTPUTS / "spike_15d" / "models",
    "growth_5d":  OUTPUTS / "growth" / "models",
    "growth_10d": OUTPUTS / "growth" / "models",
    "growth_15d": OUTPUTS / "growth" / "models",
    "buzz_composite_5d":  OUTPUTS / "buzz" / "models",
    "buzz_composite_10d": OUTPUTS / "buzz" / "models",
    "buzz_composite_15d": OUTPUTS / "buzz" / "models",
}

# 타겟 패밀리별 전략
SINGLE_TARGETS = ["sustainability_5d", "sustainability_10d", "sustainability_15d",
                  "crash_5d", "crash_10d", "crash_15d",
                  "spike_5d", "spike_10d", "spike_15d"]
SINGLE_LOG = {  # 단일 회귀의 log1p 적용 여부
    "sustainability_5d": True, "sustainability_10d": True, "sustainability_15d": True,
    "crash_5d": False, "crash_10d": False, "crash_15d": False,
    "spike_5d": False, "spike_10d": False, "spike_15d": False,
}
STAGED_GROWTH   = ["growth_5d", "growth_10d", "growth_15d"]
STAGED_BUZZ     = ["buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d"]


def _check_model_compat(target, X_cols):
    # 학습된 모델의 feature_name이 현재 입력 컬럼 안에 다 들어있는지 검증
    d = LGBM_DIRS[target]
    # 대표 모델 하나만 검사 — staged면 clf, single이면 단일
    candidates = [d / f"lgbm_{target}.txt",
                  d / f"lgbm_{target}_clf.txt",
                  d / f"lgbm_{target}_clf1.txt"]
    for p in candidates:
        if p.exists():
            needed = lgb.Booster(model_file=str(p)).feature_name()
            missing = [c for c in needed if c not in X_cols]
            return len(missing) == 0, missing
    return False, ["(모델 파일 없음)"]

DEFAULT_DAYS = 365
EPS = 1e-6


# LightGBM 로딩 — Booster 객체로 직접 사용 (predict_proba 대신 predict)
def load_booster(path):
    return lgb.Booster(model_file=str(path))


def _aligned_X(booster, X):
    # 모델이 학습 시 본 feature_name 순서대로 컬럼 추출
    names = booster.feature_name()
    return X[names].values


def _family(target):
    return target.rsplit("_", 1)[0]


def detect_staged_buckets(target):
    # 디렉토리에서 lgbm_{target}_reg_*.txt 파일 스캔, name → path 매핑
    d = LGBM_DIRS[target]
    out = {}
    for p in d.glob(f"lgbm_{target}_reg_*.txt"):
        # _reg_<name>.txt 또는 _reg_<name>_q<NN>.txt
        stem = p.stem.replace(f"lgbm_{target}_reg_", "")
        # 분위 앙상블이면 name 부분에서 _qNN 떼어내기
        if "_q" in stem and stem.split("_q")[-1].isdigit():
            name = stem.rsplit("_q", 1)[0]
        else:
            name = stem
        out.setdefault(name, []).append(p)
    return out  # {name: [path...]}


def predict_single(target, X):
    # 단일 회귀 — _aligned_X로 학습 시 피처 순서 맞춤
    booster = load_booster(LGBM_DIRS[target] / f"lgbm_{target}.txt")
    raw = booster.predict(_aligned_X(booster, X))
    return np.expm1(raw) if SINGLE_LOG[target] else raw


def predict_staged_growth(target, X):
    # 1-stage 분류기 (3-class shrink/stable/surge) + 구간별 회귀
    d = LGBM_DIRS[target]
    clf = load_booster(d / f"lgbm_{target}_clf.txt")
    proba = clf.predict(_aligned_X(clf, X))
    if proba.ndim == 1:
        proba = np.stack([1 - proba, proba], axis=1)

    reg_files = detect_staged_buckets(target)
    ordered_names = [n for n in BUCKET_CONFIG["growth"]["names"] if n in reg_files]
    use_log = BUCKET_CONFIG["growth"]["log"]

    final = np.zeros(len(X))
    for b_idx, name in enumerate(ordered_names):
        boosters = [load_booster(p) for p in reg_files[name]]
        raw = np.mean([b.predict(_aligned_X(b, X)) for b in boosters], axis=0)
        bucket_log = BUCKET_REG_OVERRIDE.get(("growth", name), {}).get("log", use_log)
        pred = np.expm1(raw) if bucket_log else raw
        if b_idx < proba.shape[1]:
            final += proba[:, b_idx] * pred
    return final, proba, ordered_names


def predict_staged_buzz(target, X):
    # 계층적 (clf1: neg vs rest, clf2: neutral vs pos) + 3 구간 회귀
    d = LGBM_DIRS[target]
    clf1 = load_booster(d / f"lgbm_{target}_clf1.txt")
    clf2 = load_booster(d / f"lgbm_{target}_clf2.txt")
    p_nn  = clf1.predict(_aligned_X(clf1, X))
    p_pos = clf2.predict(_aligned_X(clf2, X))
    proba = np.stack([1 - p_nn, p_nn * (1 - p_pos), p_nn * p_pos], axis=1)

    names = ["negative", "neutral", "positive"]
    use_log_default = BUCKET_CONFIG["buzz_composite"]["log"]
    reg_files = detect_staged_buckets(target)

    final = np.zeros(len(X))
    for b_idx, name in enumerate(names):
        paths = reg_files.get(name, [])
        if not paths:
            continue
        boosters = [load_booster(p) for p in paths]
        raw = np.mean([b.predict(_aligned_X(b, X)) for b in boosters], axis=0)
        bucket_log = BUCKET_REG_OVERRIDE.get(("buzz_composite", name), {}).get("log", use_log_default)
        pred = np.expm1(raw) if bucket_log else raw
        final += proba[:, b_idx] * pred
    return final, proba, names


# LSTM 모델 클래스 — train_lstm.py에서 그대로 복제
class AttnPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, seq):
        scores  = self.proj(seq).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (seq * weights).sum(dim=1)


LABEL_TO_IDX  = {col: i for i, col in enumerate(LSTM_LABEL_COLS)}
GROUP_INDICES = {g: [LABEL_TO_IDX[c] for c in cols] for g, cols in LSTM_LABEL_GROUPS.items()}


class HybridBiLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2,
                 feat_size=24, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_pool = AttnPool(hidden_size * 2)
        self.seq_norm  = nn.LayerNorm(hidden_size * 8)
        self.feat_branch = nn.Sequential(
            nn.Linear(feat_size, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
        )
        backbone_in = hidden_size * 8 + 96
        self.backbone = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict()
        for g, cols in LSTM_LABEL_GROUPS.items():
            self.heads[g] = nn.Sequential(
                nn.Linear(256, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, len(cols)),
            )

    def forward(self, seq, feat):
        out, _ = self.lstm(seq)
        last     = out[:, -1, :]
        max_pool = out.max(dim=1).values
        avg_pool = out.mean(dim=1)
        attn     = self.attn_pool(out)
        seq_repr = torch.cat([last, max_pool, avg_pool, attn], dim=1)
        seq_repr = self.seq_norm(seq_repr)
        feat_repr = self.feat_branch(feat)
        h = torch.cat([seq_repr, feat_repr], dim=1)
        h = self.backbone(h)
        outs = torch.zeros(seq.size(0), len(LSTM_LABEL_COLS), device=seq.device)
        for g, idxs in GROUP_INDICES.items():
            head_out = self.heads[g](h)
            for i, lbl_idx in enumerate(idxs):
                outs[:, lbl_idx] = head_out[:, i]
        return outs


# seq2seq 모델 클래스 — train_lstm_seq2seq.py에서 복제 (random 미사용, tf_ratio=0)
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
        B = dec_hidden.size(0)
        T = enc_outputs.size(1)
        H = enc_outputs.size(2)
        q = self.q_proj(dec_hidden).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(enc_outputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(enc_outputs).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        ctx = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, H)
        return self.out_proj(ctx)


class Seq2SeqBiAttn(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, enc_layers=2,
                 feat_size=24, out_steps=15, num_heads=2, dropout=0.3):
        super().__init__()
        self.out_steps   = out_steps
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=enc_layers, batch_first=True, bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.enc_out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_h_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_c_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
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
        last_fwd = hc[-2]
        last_bwd = hc[-1]
        return torch.cat([last_fwd, last_bwd], dim=-1)

    def forward(self, seq, feat):
        # 추론 — teacher forcing 미사용
        B = seq.size(0)
        enc_outputs, (h, c) = self.encoder(seq)
        enc_outputs = self.enc_out_proj(enc_outputs)
        h_merged = self._merge_directions(h)
        c_merged = self._merge_directions(c)
        dec_h    = self.enc_h_proj(h_merged)
        dec_c    = self.enc_c_proj(c_merged)
        feat_vec = self.feat_proj(feat)
        dec_h    = dec_h + feat_vec
        prev_value = torch.zeros(B, 1, device=seq.device)
        outputs = []
        for t in range(self.out_steps):
            ctx = self.attention(dec_h, enc_outputs)
            prev_proj = self.input_proj(prev_value)
            dec_in    = torch.cat([prev_proj, ctx], dim=1)
            dec_h, dec_c = self.decoder_cell(dec_in, (dec_h, dec_c))
            delta = self.delta_head(self.dropout(dec_h))
            pred = prev_value + delta
            outputs.append(pred)
            prev_value = pred
        return torch.cat(outputs, dim=1)


# 학습 시점 스케일러 복원 — train.py의 시계열 분할 + RobustScaler/StandardScaler 통계 재계산
def restore_lstm_scalers():
    X      = np.load(DATA_DIR / "lstm_X.npy")
    feat   = np.load(DATA_DIR / "lstm_feat.npy")
    labels = np.load(DATA_DIR / "lstm_labels.npy")
    meta   = pd.read_csv(DATA_DIR / "lstm_meta.csv")
    meta["date"] = pd.to_datetime(meta["date"])
    dates = np.sort(meta["date"].unique())
    n = len(dates)
    train_end = dates[int(n * 0.70)]
    train_mask = meta["date"] <= train_end

    y_tr    = labels[train_mask]
    feat_tr = feat[train_mask]

    y_median = np.median(y_tr, axis=0)
    y_q25    = np.quantile(y_tr, 0.25, axis=0)
    y_q75    = np.quantile(y_tr, 0.75, axis=0)
    y_iqr    = (y_q75 - y_q25) + 1e-8

    feat_mean = feat_tr.mean(axis=0)
    feat_std  = feat_tr.std(axis=0) + 1e-8

    return {
        "X": X, "feat": feat, "labels": labels, "meta": meta,
        "y_median": y_median, "y_iqr": y_iqr,
        "feat_mean": feat_mean, "feat_std": feat_std,
    }


def predict_lstm(model, scaler_info, kw_idx):
    # 키워드별 마지막 샘플로 11개 라벨 추론, 역정규화 후 반환
    X     = scaler_info["X"]
    feat  = scaler_info["feat"]
    fm    = scaler_info["feat_mean"]
    fs    = scaler_info["feat_std"]
    ymed  = scaler_info["y_median"]
    yiqr  = scaler_info["y_iqr"]

    x_b = torch.tensor(X[kw_idx:kw_idx+1], dtype=torch.float32).to(DEVICE)
    f_b = torch.tensor((feat[kw_idx:kw_idx+1] - fm) / fs, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = model(x_b, f_b).cpu().numpy()[0]
    return pred * yiqr + ymed


def predict_seq2seq(s2s, scaler_info, kw_idx):
    # 키워드 미래 15일 궤적 추론, z-score 역변환
    X     = scaler_info["X"]
    feat  = scaler_info["feat"]
    fm    = scaler_info["feat_mean"]
    fs    = scaler_info["feat_std"]
    meta  = scaler_info["meta"]

    x_b = torch.tensor(X[kw_idx:kw_idx+1], dtype=torch.float32).to(DEVICE)
    f_b = torch.tensor((feat[kw_idx:kw_idx+1] - fm) / fs, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = s2s(x_b, f_b).cpu().numpy()[0]  # (15,)

    row = meta.iloc[kw_idx]
    mean_c = float(row["seq_y_mean"])
    std_c  = float(row["seq_y_std"])
    return z * std_c + mean_c  # combined (search+click+blog+insta) 합산 신호 스케일


# 시계열·인구통계 — 기존 mock 스크립트에서 그대로 가져온 유틸
def load_wide(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "keyword"})
    return df


def tail_series(df, keyword, days):
    row = df.loc[df["keyword"] == keyword]
    if row.empty:
        return []
    s = row.iloc[0, 1:].astype(float).tail(days)
    return [{"date": str(d), "value": (None if pd.isna(v) else float(v))} for d, v in s.items()]


def safe_mean(arr):
    a = [x for x in arr if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(a)) if a else 0.0


def categorize(kw):
    if any(t in kw for t in ["빵", "식빵", "베이글", "크로와상", "카스테라"]):
        return "베이커리"
    if any(t in kw for t in ["쿠키", "비스킷", "마카롱"]):
        return "쿠키"
    if any(t in kw for t in ["초콜릿", "초코", "캬라멜", "캔디", "사탕"]):
        return "초콜릿/캔디"
    if any(t in kw for t in ["젤리", "약과", "한과", "오란다", "강정"]):
        return "한과/젤리"
    return "스낵"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="키워드 수 제한 (0=전체)")
    parser.add_argument("--days",  type=int, default=DEFAULT_DAYS)
    args = parser.parse_args()

    print(f"device: {DEVICE}")
    print("[1/6] dataset_final.csv 로드")
    df = pd.read_csv(DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = _add_sust_features(df)

    # 키워드별 최신 시점 row — LightGBM 추론 입력
    df_latest = df.sort_values("date").groupby("keyword", as_index=False).tail(1).reset_index(drop=True)
    data_ref_date = str(df_latest["date"].max().date())
    print(f"  데이터 기준일: {data_ref_date}  키워드 수: {len(df_latest)}")

    print("[2/6] LSTM 스케일러 복원 + 키워드별 최신 샘플 인덱스")
    info = restore_lstm_scalers()
    meta = info["meta"]
    # 키워드별 마지막 샘플 인덱스
    last_idx = meta.reset_index().groupby("keyword")["index"].max().to_dict()

    print("[3/6] LSTM/seq2seq 체크포인트 로드")
    n_feat = len(LSTM_FEAT_COLS)
    lstm = HybridBiLSTM(LSTM_INPUT_CHANNELS, 128, 2, n_feat, 0.3).to(DEVICE)
    ckpt = torch.load(LSTM_CKPT, map_location=DEVICE, weights_only=True)
    lstm.load_state_dict(ckpt["model"])
    lstm.eval()

    s2s = Seq2SeqBiAttn(LSTM_INPUT_CHANNELS, 128, 2, n_feat, 15, 2, 0.3).to(DEVICE)
    s2s.load_state_dict(torch.load(S2S_CKPT, map_location=DEVICE, weights_only=True))
    s2s.eval()

    print("[4/6] 원시 시계열·인구통계 로드")
    search_csv  = load_wide(RAW / "search" / "absolute.csv")
    click_csv   = load_wide(RAW / "shopping_click" / "absolute.csv")
    male_csv    = load_wide(RAW / "shopping_click" / "gender_m_pct.csv")
    female_csv  = load_wide(RAW / "shopping_click" / "gender_f_pct.csv")
    age_dfs = {f"age_{a}": load_wide(RAW / "shopping_click" / f"age_{a}_pct.csv")
               for a in [10, 20, 30, 40, 50, 60]}
    sometrend = pd.read_csv(RAW / "sometrends" / "sometrend_mention_long.csv", encoding="utf-8-sig")
    sometrend["date"] = pd.to_datetime(sometrend["date"]).astype(str)

    # 키워드 목록 — dataset_final 기준 (모델이 알고 있는 키워드만)
    keywords = df_latest["keyword"].tolist()
    if args.limit > 0:
        keywords = keywords[:args.limit]
    print(f"  처리 대상: {len(keywords)} 키워드")

    print("[5/6] LightGBM 단일 회귀 9개 일괄 추론")
    df_input = df_latest[df_latest["keyword"].isin(keywords)].copy().reset_index(drop=True)
    feat_cols_sust = FEAT_COLS + SUST_EXTRA_FEATS
    X_base = df_input[FEAT_COLS]
    X_sust = df_input[feat_cols_sust]

    pred_lgbm = {kw: {} for kw in df_input["keyword"]}
    for tgt in SINGLE_TARGETS:
        if tgt in SKIP_TARGETS:
            print(f"  [skip] {tgt}")
            continue
        X = X_sust if _family(tgt) == "sustainability" else X_base
        ok, miss = _check_model_compat(tgt, set(X.columns))
        if not ok:
            print(f"  [skip] {tgt} — 누락 피처 {len(miss)}: {miss[:3]}...")
            continue
        preds = predict_single(tgt, X)
        for kw, v in zip(df_input["keyword"], preds):
            pred_lgbm[kw][tgt] = float(v)
        print(f"  {tgt}: mean={float(np.mean(preds)):.4f}")

    print("[5/6] LightGBM staged growth 3개")
    growth_probs = {kw: {} for kw in df_input["keyword"]}
    for tgt in STAGED_GROWTH:
        if tgt in SKIP_TARGETS:
            print(f"  [skip] {tgt} — 팀원 재학습 대기")
            continue
        ok, miss = _check_model_compat(tgt, set(X_base.columns))
        if not ok:
            print(f"  [skip] {tgt} — 누락 피처 {len(miss)}: {miss[:3]}...")
            continue
        preds, proba_all, names_avail = predict_staged_growth(tgt, X_base)
        for i, kw in enumerate(df_input["keyword"]):
            pred_lgbm[kw][tgt] = float(preds[i])
            growth_probs[kw][tgt] = {names_avail[j]: float(proba_all[i, j])
                                     for j in range(min(len(names_avail), proba_all.shape[1]))}
        print(f"  {tgt}: mean={float(np.mean(preds)):.4f}")

    print("[5/6] LightGBM 계층 buzz_composite 3개")
    buzz_probs = {kw: {} for kw in df_input["keyword"]}
    for tgt in STAGED_BUZZ:
        if tgt in SKIP_TARGETS:
            print(f"  [skip] {tgt}")
            continue
        ok, miss = _check_model_compat(tgt, set(X_base.columns))
        if not ok:
            print(f"  [skip] {tgt} — 누락 피처 {len(miss)}: {miss[:3]}...")
            continue
        final, proba, names = predict_staged_buzz(tgt, X_base)
        for i, kw in enumerate(df_input["keyword"]):
            pred_lgbm[kw][tgt] = float(final[i])
            buzz_probs[kw][tgt] = {names[j]: float(proba[i, j]) for j in range(3)}
        print(f"  {tgt}: mean={float(np.mean(final)):.4f}")

    print("[5/6] LSTM 11개 라벨 + seq2seq 미래 15일 (배치 추론)")
    # 키워드별 인덱스 모아 한 번에 배치 처리
    kw_with_idx = [(kw, last_idx[kw]) for kw in keywords if kw in last_idx]
    kws_lstm   = [kw for kw, _ in kw_with_idx]
    idx_arr    = np.array([i for _, i in kw_with_idx])

    X_seq = info["X"][idx_arr]
    F_seq = (info["feat"][idx_arr] - info["feat_mean"]) / info["feat_std"]
    xb = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    fb = torch.tensor(F_seq, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        lstm_pred = lstm(xb, fb).cpu().numpy()
        s2s_pred  = s2s(xb, fb).cpu().numpy()

    lstm_pred = lstm_pred * info["y_iqr"] + info["y_median"]

    pred_lstm = {}
    forecast_series = {}
    for j, kw in enumerate(kws_lstm):
        pred_lstm[kw] = {LSTM_LABEL_COLS[k]: float(lstm_pred[j, k]) for k in range(len(LSTM_LABEL_COLS))}
        row = info["meta"].iloc[idx_arr[j]]
        mean_c = float(row["seq_y_mean"])
        std_c  = float(row["seq_y_std"])
        traj = s2s_pred[j] * std_c + mean_c  # 합산 신호 스케일
        # meta date = forward window 시작일 (build_dataset_lstm 기준)
        start_date = pd.to_datetime(row["date"])
        forecast_series[kw] = [
            {"date": str((start_date + pd.Timedelta(days=k)).date()), "value": float(traj[k])}
            for k in range(15)
        ]

    print("[6/6] JSON 작성")
    OUT_FRONT.mkdir(parents=True, exist_ok=True)
    (OUT_FRONT / "keywords").mkdir(exist_ok=True)

    summary = []
    for kw in keywords:
        s_series = tail_series(search_csv, kw, args.days)
        c_series = tail_series(click_csv,  kw, args.days)
        m_row = male_csv.loc[male_csv["keyword"] == kw]
        f_row = female_csv.loc[female_csv["keyword"] == kw]
        gender = {
            "male":   safe_mean(m_row.iloc[0, 1:].astype(float).tail(30).tolist()) if not m_row.empty else 50.0,
            "female": safe_mean(f_row.iloc[0, 1:].astype(float).tail(30).tolist()) if not f_row.empty else 50.0,
        }
        ages = {}
        for label, dfa in age_dfs.items():
            r = dfa.loc[dfa["keyword"] == kw]
            ages[label] = safe_mean(r.iloc[0, 1:].astype(float).tail(30).tolist()) if not r.empty else 0.0

        st_kw = sometrend[sometrend["keyword"] == kw].sort_values("date").tail(args.days)
        blog_series = [{"date": d, "value": float(v) if not pd.isna(v) else None}
                       for d, v in zip(st_kw["date"], st_kw.get("블로그", pd.Series(dtype=float)))]
        insta_series = [{"date": d, "value": float(v) if not pd.isna(v) else None}
                        for d, v in zip(st_kw["date"], st_kw.get("인스타그램", pd.Series(dtype=float)))]

        lgbm_p = pred_lgbm.get(kw, {})
        lstm_p = pred_lstm.get(kw, {})

        # 윈도우/타깃 매트릭스 — 프론트 토글용
        windows = [5, 10, 15]
        targets_base = ["growth", "sustainability", "buzz_composite", "crash", "spike"]
        pred_matrix = {}
        for t in targets_base:
            pred_matrix[t] = {w: lgbm_p.get(f"{t}_{w}d") for w in windows}

        detail = {
            "keyword": kw,
            "category": categorize(kw),
            "data_reference_date": data_ref_date,
            "timeseries": {
                "search": s_series, "click": c_series,
                "blog": blog_series, "instagram": insta_series,
            },
            "demographics": {"gender": gender, "ages": ages},
            "predictions_lgbm": {
                "matrix": pred_matrix,
                "staged_probs": {
                    "growth": growth_probs.get(kw, {}),
                    "buzz_composite": buzz_probs.get(kw, {}),
                },
            },
            "predictions_lstm": lstm_p,
            "forecast_series": forecast_series.get(kw, []),
            # 호환성 — 구 schema 키 유지
            "predictions": {
                **lgbm_p,
                **{k: lstm_p.get(k) for k in [
                    "fw_peak_softpos_10d", "fw_peak_softpos_15d",
                    "fw_cv_10d", "fw_cv_15d",
                ] if k in lstm_p},
            },
            "shap": [],
            "related": [],
        }

        body = json.dumps(detail, ensure_ascii=False)
        (OUT_FRONT / "keywords" / f"{kw}.json").write_text(body, encoding="utf-8")

        s_vals = [p["value"] for p in s_series if p["value"] is not None]
        last30 = s_vals[-30:] if len(s_vals) >= 30 else s_vals
        prev30 = s_vals[-60:-30] if len(s_vals) >= 60 else s_vals
        recent_avg = float(np.mean(last30)) if last30 else 0.0
        prev_avg   = float(np.mean(prev30)) if prev30 else 0.0

        summary.append({
            "keyword": kw,
            "category": detail["category"],
            "growth_10d":         lgbm_p.get("growth_10d"),
            "sustainability_10d": lgbm_p.get("sustainability_10d"),
            "buzz_composite_10d": lgbm_p.get("buzz_composite_10d"),
            "crash_10d":          lgbm_p.get("crash_10d"),
            "spike_10d":          lgbm_p.get("spike_10d"),
            "growth_5d":          lgbm_p.get("growth_5d"),
            "growth_15d":         lgbm_p.get("growth_15d"),
            "sustainability_5d":  lgbm_p.get("sustainability_5d"),
            "sustainability_15d": lgbm_p.get("sustainability_15d"),
            "crash_15d":          lgbm_p.get("crash_15d"),
            "fw_peak_softpos_10d": lstm_p.get("fw_peak_softpos_10d"),
            "fw_peak_softpos_15d": lstm_p.get("fw_peak_softpos_15d"),
            "fw_magnitude_10d":    lstm_p.get("fw_magnitude_10d"),
            "fw_cv_10d":           lstm_p.get("fw_cv_10d"),
            "fw_log_growth_10d":   lstm_p.get("fw_log_growth_10d"),
            "fw_decline_10d":      lstm_p.get("fw_decline_10d"),
            "recent_avg": round(recent_avg, 1),
            "prev_avg":   round(prev_avg, 1),
            "delta_pct":  round((recent_avg / max(1.0, prev_avg) - 1) * 100, 1),
        })

    body = json.dumps({
        "data_reference_date": data_ref_date,
        "keywords": summary,
    }, ensure_ascii=False)
    (OUT_FRONT / "summary.json").write_text(body, encoding="utf-8")
    print(f"\n완료 — 키워드 {len(summary)}개, 기준일 {data_ref_date}")


if __name__ == "__main__":
    main()
