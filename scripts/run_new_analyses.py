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


def run_backtesting(df: pd.DataFrame, n_splits: int = 6, collect_details: bool = False):
    X = df[["HomeTeam", "AwayTeam"]]
    y = df["FTR"]
    est = build_estimator()

    cv = TimeSeriesSplit(n_splits=n_splits)
    K_roll = max(100, int(len(X) * 0.25))

    results = []
    details = []
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

            if collect_details and regime_name == "expanding":
                y_pred = classes[proba.argmax(axis=1)]
                p_max = proba.max(axis=1)
                sub = pd.DataFrame({
                    "split": int(split_idx),
                    "home": X.iloc[te]["HomeTeam"].to_numpy(),
                    "away": X.iloc[te]["AwayTeam"].to_numpy(),
                    "true": y_true,
                    "pred": y_pred,
                    "p_max": p_max,
                    "correct": (y_pred == y_true),
                })
                details.append(sub)

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("regime")[ ["log_loss", "brier"] ].mean().sort_values("log_loss")
    print("===BACKTEST SUMMARY START===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("===BACKTEST SUMMARY END===")
    # Save summary for notebook consumption
    out_csv = os.path.join("docs", "backtest_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary.to_csv(out_csv)
    if collect_details:
        details_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
        return summary, details_df
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


def reliability_and_ece(details_df: pd.DataFrame, out_png: str) -> pd.DataFrame:
    if details_df.empty:
        raise ValueError("Empty details_df for reliability computation")
    # Bin by top-class probability
    bins = np.linspace(0.0, 1.0, 11)
    labels = pd.IntervalIndex.from_breaks(bins)
    details_df = details_df.copy()
    # Clip to avoid 1.0 falling outside last interval due to right-open default
    eps = 1e-12
    details_df["p_max_clipped"] = np.clip(details_df["p_max"], bins[0] + eps, bins[-1] - eps)
    details_df["bin"] = pd.cut(details_df["p_max_clipped"], bins=bins)
    by = details_df.groupby("bin")
    calib = by.agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
        conf=("p_max", "mean"),
    ).reset_index()
    total = calib["n"].sum()
    calib["weight"] = calib["n"] / total
    calib["gap"] = (calib["acc"] - calib["conf"]).abs()
    ece = float((calib["weight"] * calib["gap"]).sum())
    mce = float(calib["gap"].max())

    # Plot reliability
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(calib["conf"], calib["acc"], "o-", label="Model")
    plt.xlabel("Predicted probability (confidence)")
    plt.ylabel("Observed accuracy")
    plt.title(f"Reliability curve\nECE={ece:.3f}  MCE={mce:.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()

    print("===RELIABILITY SUMMARY START===")
    print(f"ECE (Expected Calibration Error) = {ece:.4f}")
    print(f"MCE (Maximum Calibration Error) = {mce:.4f}")
    print("===RELIABILITY SUMMARY END===")
    out_csv = os.path.join("docs", "reliability_summary.csv")
    calib_out = calib.assign(ECE=ece, MCE=mce)
    calib_out.to_csv(out_csv, index=False)
    return calib_out


def error_slices(details_df: pd.DataFrame, out_png: str) -> pd.DataFrame:
    if details_df.empty:
        raise ValueError("Empty details_df for error slicing")
    # Home vs Away correctness
    # For team-level view, compute correctness when team appears at home and at away separately
    home_acc = details_df.groupby("home")["correct"].mean().sort_values(ascending=False)
    away_acc = details_df.groupby("away")["correct"].mean().sort_values(ascending=False)

    # Keep top 10 by sample count for each
    home_counts = details_df.groupby("home").size().sort_values(ascending=False)
    away_counts = details_df.groupby("away").size().sort_values(ascending=False)
    top_home = home_counts.head(10).index
    top_away = away_counts.head(10).index

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    home_acc.loc[top_home].sort_values().plot(kind="barh", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Accuracy by team (when at home)")
    axes[0].set_xlabel("Accuracy")
    away_acc.loc[top_away].sort_values().plot(kind="barh", ax=axes[1], color="#F58518")
    axes[1].set_title("Accuracy by team (when away)")
    axes[1].set_xlabel("Accuracy")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close(fig)

    print("===ERROR SLICES SUMMARY START===")
    print("Top-10 teams by sample size (home) with accuracy:")
    print(home_acc.loc[top_home].to_string(float_format=lambda v: f"{v:.3f}"))
    print("Top-10 teams by sample size (away) with accuracy:")
    print(away_acc.loc[top_away].to_string(float_format=lambda v: f"{v:.3f}"))
    print("===ERROR SLICES SUMMARY END===")
    out_df = pd.DataFrame({
        "home_team": home_acc.index,
        "home_acc": home_acc.values,
    }).merge(
        pd.DataFrame({"away_team": away_acc.index, "away_acc": away_acc.values}),
        left_on="home_team", right_on="away_team", how="outer"
    )
    out_csv = os.path.join("docs", "error_slices.csv")
    out_df.to_csv(out_csv, index=False)
    return out_df


def compute_elo_and_form_features(df: pd.DataFrame) -> pd.DataFrame:
    from collections import deque

    df = df.copy().reset_index(drop=True)
    # Ensure chronological order if date available
    for col in ["Date", "MatchDate", "matchday", "MatchDay"]:
        if col in df.columns:
            df = df.sort_values(col).reset_index(drop=True)
            break

    # Initialize structures
    elo_ratings = {}
    last_points = {}
    last_goal_diff = {}
    window = 5
    K = 20.0
    home_adv = 60.0

    elo_home, elo_away = [], []
    pts_home_l5, pts_away_l5 = [], []
    gd_home_l5, gd_away_l5 = [], []

    def points_from_outcome(outcome: str, is_home: bool) -> float:
        if outcome == "D":
            return 1.0
        if (outcome == "H" and is_home) or (outcome == "A" and not is_home):
            return 3.0
        return 0.0

    def update_elo(r_home: float, r_away: float, outcome: str) -> tuple[float, float]:
        p_home = 1.0 / (1.0 + 10.0 ** (-((r_home + home_adv) - r_away) / 400.0))
        s_home = {"H": 1.0, "D": 0.5, "A": 0.0}[outcome]
        r_home_new = r_home + K * (s_home - p_home)
        r_away_new = r_away + K * ((1.0 - s_home) - (1.0 - p_home))
        return r_home_new, r_away_new

    for i, row in df.iterrows():
        h, a, outcome = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        rh = elo_ratings.get(h, 1500.0)
        ra = elo_ratings.get(a, 1500.0)
        elo_home.append(rh)
        elo_away.append(ra)

        # Rolling windows
        hp = last_points.get(h, deque(maxlen=window))
        ap = last_points.get(a, deque(maxlen=window))
        hg = last_goal_diff.get(h, deque(maxlen=window))
        ag = last_goal_diff.get(a, deque(maxlen=window))

        pts_home_l5.append(sum(hp))
        pts_away_l5.append(sum(ap))
        gd_home_l5.append(sum(hg))
        gd_away_l5.append(sum(ag))

        # After match: update stats
        # If goal columns exist, use them; otherwise approximate via outcome
        gf_h = 1 if outcome == "H" else 0
        gf_a = 1 if outcome == "A" else 0
        ga_h = gf_a
        ga_a = gf_h

        hp.append(points_from_outcome(outcome, is_home=True))
        ap.append(points_from_outcome(outcome, is_home=False))
        hg.append(gf_h - ga_h)
        ag.append(gf_a - ga_a)
        last_points[h] = hp
        last_points[a] = ap
        last_goal_diff[h] = hg
        last_goal_diff[a] = ag

        # Elo update
        rh2, ra2 = update_elo(rh, ra, outcome)
        elo_ratings[h] = rh2
        elo_ratings[a] = ra2

    out = df.copy()
    out["elo_home"] = np.array(elo_home)
    out["elo_away"] = np.array(elo_away)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["points_home_l5"] = np.array(pts_home_l5)
    out["points_away_l5"] = np.array(pts_away_l5)
    out["gd_home_l5"] = np.array(gd_home_l5)
    out["gd_away_l5"] = np.array(gd_away_l5)
    return out


def run_backtest_with_elo_form(df_base: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler

    df_feat = compute_elo_and_form_features(df_base)
    X = df_feat[[
        "HomeTeam", "AwayTeam",
        "elo_diff", "points_home_l5", "points_away_l5", "gd_home_l5", "gd_away_l5"
    ]]
    y = df_feat["FTR"]

    pre = ColumnTransformer([
        ("cats", OneHotEncoder(handle_unknown="ignore"), ["HomeTeam", "AwayTeam"]),
        ("nums", StandardScaler(), [
            "elo_diff", "points_home_l5", "points_away_l5", "gd_home_l5", "gd_away_l5"
        ]),
    ])
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    est = make_pipeline(pre, model)

    cv = TimeSeriesSplit(n_splits=6)
    results = []
    for split_idx, (tr, te) in enumerate(cv.split(X)):
        est.fit(X.iloc[tr], y.iloc[tr])
        proba = est.predict_proba(X.iloc[te])
        y_true = y.iloc[te].to_numpy()
        classes = est.classes_
        results.append({
            "split": int(split_idx),
            "regime": "expanding+elo+form",
            "log_loss": float(log_loss(y_true, proba, labels=classes)),
            "brier": multiclass_brier(y_true, proba, classes),
        })

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("regime")[ ["log_loss", "brier"] ].mean()
    print("===ELO+FORM BACKTEST SUMMARY START===")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("===ELO+FORM BACKTEST SUMMARY END===")
    out_csv = os.path.join("docs", "elo_form_backtest.csv")
    summary.to_csv(out_csv)
    return summary


def load_la_liga_multiseason() -> pd.DataFrame:
    seasons = [
        "season-1011.csv",
        "season-1112.csv",
        "season-1213.csv",
    ]
    frames = []
    for fname in seasons:
        url = f"https://raw.githubusercontent.com/datasets/football-datasets/master/datasets/la-liga/{fname}"
        try:
            df = pd.read_csv(url)
            df = df.loc[df["FTR"].isin(["H","D","A"])].copy()
            df["SeasonFile"] = fname
            frames.append(df)
        except Exception as exc:
            print(f"WARN: could not load {fname}: {exc}")
    if not frames:
        raise RuntimeError("No seasons loaded for multi-season check")
    all_df = pd.concat(frames, ignore_index=True)
    for col in ["Date", "MatchDate", "matchday", "MatchDay"]:
        if col in all_df.columns:
            all_df = all_df.sort_values(col)
            break
    return all_df.reset_index(drop=True)[["HomeTeam","AwayTeam","FTR"]]


def run_learning_curve_multiseason(out_png: str) -> pd.DataFrame:
    df_multi = load_la_liga_multiseason()
    summary = run_learning_curve(df_multi, out_png=out_png)
    out_csv = os.path.join("docs", "learning_curve_multiseason.csv")
    summary.to_csv(out_csv, index=False)
    return summary


def main() -> int:
    np.random.seed(42)
    df = load_ordered_dataset()

    # 1) Backtesting (with details for reliability/error slicing)
    backtest_summary, details = run_backtesting(df, collect_details=True)

    # 2) Learning curve
    out_png = os.path.join("docs", "learning_curve.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    run_learning_curve(df, out_png=out_png)

    # 3) Temperature scaling
    run_temperature_scaling(df)

    # 4) Reliability and ECE/MCE
    rel_png = os.path.join("docs", "reliability_curve.png")
    reliability_and_ece(details, out_png=rel_png)

    # 5) Error slicing plots
    slices_png = os.path.join("docs", "error_slices.png")
    error_slices(details, out_png=slices_png)

    # 6) Elo + rolling form features backtest
    run_backtest_with_elo_form(df)

    # 7) Multi-season learning curve
    lc_multi_png = os.path.join("docs", "learning_curve_multiseason.png")
    run_learning_curve_multiseason(out_png=lc_multi_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


