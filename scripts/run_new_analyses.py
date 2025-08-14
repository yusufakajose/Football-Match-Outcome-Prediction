import os
import sys
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.special import softmax
from scipy.optimize import minimize


DATA_URL = "https://raw.githubusercontent.com/datasets/football-datasets/master/datasets/la-liga/season-1112.csv"


def load_ordered_dataset() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as exc:
        print(f"ERROR: failed to load dataset: {exc}", file=sys.stderr)
        raise

    df = df.loc[df["FTR"].isin(["H", "D", "A"])].copy()
    # Choose a chronological column if available
    for col in ["Date", "MatchDate", "matchday", "MatchDay"]:
        if col in df.columns:
            df = df.sort_values(col)
            break
    df = df.reset_index(drop=True)
    return df[["HomeTeam", "AwayTeam", "FTR"]]


def build_estimator():
    pre = ColumnTransformer([
        ("cats", OneHotEncoder(handle_unknown="ignore"), ["HomeTeam", "AwayTeam"]),
    ])
    model = LogisticRegression(max_iter=400, multi_class="multinomial", solver="lbfgs")
    return make_pipeline(pre, model)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    onehot = (y_true[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def run_backtesting(df: pd.DataFrame, n_splits: int = 6) -> pd.DataFrame:
    X = df[["HomeTeam", "AwayTeam"]]
    y = df["FTR"]
    est = build_estimator()

    cv = TimeSeriesSplit(n_splits=n_splits)
    K_roll = max(100, int(len(X) * 0.25))

    results = []
    for split_idx, (tr, te) in enumerate(cv.split(X)):
        tr_expanding = tr
        tr_rolling = tr[-K_roll:] if len(tr) > K_roll else tr

        for regime_name, tr_idx in [("expanding", tr_expanding), ("rolling", tr_rolling)]:
            est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = est.predict_proba(X.iloc[te])
            y_true = y.iloc[te].to_numpy()
            classes = est.classes_
            results.append({
                "split": int(split_idx),
                "regime": regime_name,
                "log_loss": float(log_loss(y_true, proba, labels=classes)),
                "brier": multiclass_brier(y_true, proba, classes),
            })

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("regime")[ ["log_loss", "brier"] ].mean().sort_values("log_loss")
    print("===BACKTEST SUMMARY START===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("===BACKTEST SUMMARY END===")
    return summary


def run_learning_curve(df: pd.DataFrame, out_png: str) -> pd.DataFrame:
    X = df[["HomeTeam", "AwayTeam"]]
    y = df["FTR"]
    est = build_estimator()

    cv = TimeSeriesSplit(n_splits=5)
    train_sizes = np.linspace(0.2, 1.0, 6)
    sizes, train_scores, val_scores = learning_curve(
        estimator=est,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_log_loss",
        shuffle=False,
        n_jobs=None,
    )

    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    summary = pd.DataFrame({
        "train_size_obs": sizes.astype(int),
        "train_log_loss_mean": train_mean,
        "val_log_loss_mean": val_mean,
    })

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, train_mean, "o-", label="Train (log-loss)")
    plt.plot(sizes, val_mean, "o-", label="Validation (log-loss)")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    plt.xlabel("Number of training matches")
    plt.ylabel("Log-loss (lower is better)")
    plt.title("Learning curve (chronological CV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

    print("===LEARNING CURVE SUMMARY START===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("===LEARNING CURVE SUMMARY END===")
    print(f"Saved plot: {out_png}")
    return summary


def run_temperature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["HomeTeam", "AwayTeam"]]
    y = df["FTR"]
    est = build_estimator()

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, val_idx = None, None
    for tr, va in tscv.split(X):
        train_idx, val_idx = tr, va

    est.fit(X.iloc[train_idx], y.iloc[train_idx])
    classes = est.classes_
    proba_val = est.predict_proba(X.iloc[val_idx])

    # Try to get logits; if unavailable, fallback via log-probabilities
    logits_val = None
    if hasattr(est, "decision_function"):
        try:
            logits_val = est.decision_function(X.iloc[val_idx])
        except Exception:
            logits_val = None
    if logits_val is None:
        eps = 1e-12
        logits_val = np.log(np.clip(proba_val, eps, 1.0))
    if logits_val.ndim == 1:
        logits2 = np.column_stack([np.zeros_like(logits_val), logits_val])
    else:
        logits2 = logits_val

    y_val = y.iloc[val_idx].to_numpy()

    def nll_temperature(T_arr: np.ndarray) -> float:
        T = float(T_arr[0])
        probs = softmax(logits2 / T, axis=1)
        return float(log_loss(y_val, probs, labels=classes))

    opt = minimize(nll_temperature, x0=[1.0], bounds=[(0.05, 5.0)], method="L-BFGS-B")
    T_opt = float(opt.x[0])
    proba_val_cal = softmax(logits2 / T_opt, axis=1)

    baseline_ll = float(log_loss(y_val, proba_val, labels=classes))
    calibrated_ll = float(log_loss(y_val, proba_val_cal, labels=classes))
    baseline_brier = multiclass_brier(y_val, proba_val, classes)
    calibrated_brier = multiclass_brier(y_val, proba_val_cal, classes)

    summary = pd.DataFrame([
        {"model": "baseline", "log_loss": baseline_ll, "brier": baseline_brier, "T": 1.0},
        {"model": "temp_scaled", "log_loss": calibrated_ll, "brier": calibrated_brier, "T": T_opt},
    ]).set_index("model")

    print("===CALIBRATION SUMMARY START===")
    print(f"Optimal temperature T = {T_opt:.3f}")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("===CALIBRATION SUMMARY END===")
    return summary


def main() -> int:
    np.random.seed(42)
    df = load_ordered_dataset()

    # 1) Backtesting
    run_backtesting(df)

    # 2) Learning curve
    out_png = os.path.join("docs", "learning_curve.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    run_learning_curve(df, out_png=out_png)

    # 3) Temperature scaling
    run_temperature_scaling(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


