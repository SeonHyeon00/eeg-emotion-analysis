# ==============================================================
# eeg_model_test.py — 결과+성능 저장 버전
#  - 출력: "Positive xx.x% | Active yy.y%"
#  - 추가 저장: predictions.csv, metrics.json, classification_report.txt, confusion_matrix.png
# ==============================================================

import sys, io, json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.integrate import trapezoid
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # 터미널/서버 환경에서도 저장 가능
import matplotlib.pyplot as plt

# -----------------[ 상수/설정 ]-----------------
CLASS_NAMES  = ["Enjoyed","Funny","Relaxed","Sad","Scary"]
POSITIVE_SET = {"Enjoyed","Funny","Relaxed"}
SEQ_LEN, N_CLASSES = 113, 5
FS = 256  # 기본 샘플링 레이트(밴드 파워용)

# -----------------[ 경로 함수 ]-----------------
def repo_root_from_here():
    # 이 파일은 src/ 밑에 있으므로 repo 루트는 parents[1]
    return Path(__file__).resolve().parents[1]

# -----------------[ 유틸 ]-----------------
def resample_to_len(x_row, target_len):
    src = np.linspace(0, 1, len(x_row), dtype=np.float32)
    dst = np.linspace(0, 1, target_len, dtype=np.float32)
    return np.interp(dst, src, x_row).astype(np.float32)

def load_sheet_numeric(path, sheet=None):
    data = pd.read_excel(str(path), sheet_name=sheet, header=None)
    if isinstance(data, dict):  # 여러 시트면 첫 시트만
        df = next(iter(data.values()))
    else:
        df = data
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df.values.astype(np.float32)
    if X.shape[1] != SEQ_LEN:
        X = np.vstack([resample_to_len(r, SEQ_LEN) for r in X])
    # 표준화
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    return X  # shape: [N, SEQ_LEN]

def band_power_welch(x, fs, fmin, fmax):
    nperseg = min(1024, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    m = (f >= fmin) & (f <= fmax)
    return float(trapezoid(Pxx[m], f[m]))

def rel_bands(x, fs):
    total = band_power_welch(x, fs, 0.5, 45.0) + 1e-12
    return {
        "rel_alpha": band_power_welch(x, fs, 8, 13) / total,
        "rel_theta": band_power_welch(x, fs, 4, 8) / total,
        "rel_beta":  band_power_welch(x, fs, 13, 30) / total,
        "rel_gamma": band_power_welch(x, fs, 30, 45) / total,
    }

def active_percent_from_rel(rel):
    a, t, b, g = rel["rel_alpha"], rel["rel_theta"], rel["rel_beta"], rel["rel_gamma"]
    return float((b + g) / (a + t + b + g + 1e-9) * 100.0)

def probs_from_model(model, X2d, device):
    xb = torch.tensor(X2d, dtype=torch.float32, device=device).unsqueeze(-1)  # [B,T,1]
    pos = torch.arange(SEQ_LEN, device=device).unsqueeze(0).repeat(xb.size(0), 1)  # [B,T]
    with torch.no_grad():
        out = model(xb, pos)
        logits = out[1] if isinstance(out, tuple) else out  # PBT는 (features, logits, mask)
        prob = torch.softmax(logits, dim=1).cpu().numpy()   # [N, 5]
    return prob

def to_positive_percent_from_probs(prob):
    name2idx = {c:i for i,c in enumerate(CLASS_NAMES)}
    idxs = [name2idx[c] for c in CLASS_NAMES if c in POSITIVE_SET]
    return float(prob.mean(0)[idxs].mean() * 100.0)

def save_confusion_matrix_png(cm, labels, out_png):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # 값 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.colorbar(im, ax=ax)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# -----------------[ 메인 ]-----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/samples/dog_sample01.xlsx")
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="suppress console print; write files only")
    args = ap.parse_args()

    ROOT = repo_root_from_here()
    SRC_DIR = ROOT / "src"
    WEIGHTS_DIR = ROOT / "weights"
    STATE_PATH  = WEIGHTS_DIR / "pbt_dogready_state.pth"
    EXCEL_PATH  = Path(args.input) if args.input else (ROOT / "data" / "samples" / "dog_sample01.xlsx")
    OUT_DIR     = Path(args.outdir) if args.outdir else (ROOT / "results")
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    if args.verbose and not args.quiet:
        print(f"[INFO] ROOT={ROOT}")
        print(f"[INFO] EXCEL_PATH={EXCEL_PATH} exists={EXCEL_PATH.exists()}")
        print(f"[INFO] STATE_PATH={STATE_PATH} exists={STATE_PATH.exists()}")
        print(f"[INFO] OUT_DIR={OUT_DIR}")

    # 모델 import
    try:
        from pbt.model import PBT
    except ModuleNotFoundError:
        sys.path.insert(0, str(SRC_DIR))
        from pbt.model import PBT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose and not args.quiet:
        print(f"[INFO] device={device}, torch={torch.__version__}")

    # PBT 생성 (원본 구현은 레이어 생성 시 콘솔 문구가 있을 수 있어 숨김)
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    model = PBT(
        d_input=1,
        n_classes=N_CLASSES,
        num_embeddings=SEQ_LEN,
        num_tokens_per_channel=1,
        d_model=128,
        n_blocks=4,
        num_heads=4,
        dropout=0.2,
        device=device
    ).to(device)
    sys.stdout = _stdout

    # 가중치 로드
    if STATE_PATH.exists():
        state = torch.load(str(STATE_PATH), map_location=device)
        model.load_state_dict(state, strict=False)
        if args.verbose and not args.quiet:
            print("[INFO] weights loaded.")
    else:
        if args.verbose and not args.quiet:
            print(f"[WARN] weights not found → {STATE_PATH.name} skipped (random init)")
    model.eval()

    # 데이터 로드
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")
    X = load_sheet_numeric(EXCEL_PATH)
    if args.verbose and not args.quiet:
        print(f"[INFO] X shape={X.shape}")

    # 추론
    prob = probs_from_model(model, X, device)   # [N,5]
    pred_idx = prob.argmax(axis=1)
    pred_cls = [CLASS_NAMES[i] for i in pred_idx]

    # Positive/Active 백분율(전체 평균)
    pos_pct = to_positive_percent_from_probs(prob)
    act_vals = [active_percent_from_rel(rel_bands(x, FS)) for x in X]
    act_pct = float(np.mean(act_vals))

    # 콘솔 한 줄 + 파일 저장
    out_line = f"Positive {pos_pct:.1f}% | Active {act_pct:.1f}%"
    if not args.quiet:
        print(out_line)
    (OUT_DIR / "prediction_output.txt").write_text(out_line, encoding="utf-8")

    # 샘플별 예측 저장
    df_pred = pd.DataFrame({
        "id": np.arange(len(pred_cls)),
        "pred_class": pred_cls,
        "pred_prob": prob.max(axis=1),
        "positive_pct": pos_pct,  # 전체 평균 동일값(요약치)
        "active_pct": act_pct,    # 전체 평균 동일값
    })
    df_pred.to_csv(OUT_DIR / "predictions.csv", index=False, encoding="utf-8")

    # (선택) 성능 지표 계산 — 라벨이 있을 때만
    if args.labels:
        labels_path = Path(args.labels)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        if labels_path.suffix.lower() in [".xlsx", ".xls"]:
            df_lab = pd.read_excel(labels_path)
        else:
            df_lab = pd.read_csv(labels_path)
        # 라벨 컬럼 찾기
        col = "label" if "label" in df_lab.columns else df_lab.columns[0]
        y_true_cls = df_lab[col].astype(str).tolist()
        if len(y_true_cls) != len(pred_cls):
            raise ValueError(f"#labels({len(y_true_cls)}) != #predictions({len(pred_cls)})")

        # 멀티클래스 성능
        acc = accuracy_score(y_true_cls, pred_cls)
        f1_macro = f1_score(y_true_cls, pred_cls, labels=CLASS_NAMES, average="macro")
        f1_per_class = f1_score(y_true_cls, pred_cls, labels=CLASS_NAMES, average=None)
        cls_report = classification_report(y_true_cls, pred_cls, labels=CLASS_NAMES, digits=4, zero_division=0)
        (OUT_DIR / "classification_report.txt").write_text(cls_report, encoding="utf-8")

        # 혼동행렬
        cm = confusion_matrix(y_true_cls, pred_cls, labels=CLASS_NAMES)
        save_confusion_matrix_png(cm, CLASS_NAMES, OUT_DIR / "confusion_matrix.png")

        # Positive-vs-Others (이진) 성능도 함께
        y_true_pos = [1 if c in POSITIVE_SET else 0 for c in y_true_cls]
        y_pred_pos = [1 if c in POSITIVE_SET else 0 for c in pred_cls]
        acc_pos = accuracy_score(y_true_pos, y_pred_pos)
        f1_pos  = f1_score(y_true_pos, y_pred_pos)

        metrics = {
            "n_samples": len(pred_cls),
            "multiclass": {
                "accuracy": round(float(acc), 6),
                "macro_f1": round(float(f1_macro), 6),
                "per_class_f1": {cls: round(float(v), 6) for cls, v in zip(CLASS_NAMES, f1_per_class)}
            },
            "positive_vs_others": {
                "accuracy": round(float(acc_pos), 6),
                "f1": round(float(f1_pos), 6)
            },
            "positive_active_summary": {
                "positive_pct_mean": round(float(pos_pct), 3),
                "active_pct_mean": round(float(act_pct), 3)
            }
        }
        (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.verbose and not args.quiet:
            print("[INFO] metrics saved:",
                  (OUT_DIR / "metrics.json").as_posix(),
                  (OUT_DIR / "classification_report.txt").as_posix(),
                  (OUT_DIR / "confusion_matrix.png").as_posix())

if __name__ == "__main__":
    main()
