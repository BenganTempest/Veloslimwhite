"""
The "does this look like stocks that broke out before" model.

Important framing, repeated in the README and on the dashboard itself:
this produces a PATTERN-SIMILARITY score, not a probability that a stock
will go up. It is trained on a mechanical historical label (did the stock
gain >= BREAKOUT_RETURN over the next BREAKOUT_WINDOW trading days) using
only price/volume features available at the time. Past breakouts having
looked a certain way is not evidence that stocks which look similar today
will repeat that outcome -- markets are not stationary, this dataset is
small relative to the pattern space, and this kind of setup is exactly
where overfitting is easiest. The honest out-of-sample metrics computed
here (and shown on the dashboard) are the check on that, not a guarantee.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from features import FEATURE_COLUMNS, compute_features, forward_return  # noqa: E402

log = logging.getLogger("pattern_model")


@dataclass
class TrainedModel:
    scaler: StandardScaler
    clf: LogisticRegression
    validation_auc: float
    validation_precision_at_top_decile: float
    n_train_rows: int
    n_positive_train: int
    n_validation_rows: int


def build_training_panel(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack (ticker, date) feature rows + breakout label across every ticker's
    full history. This is what the model is trained/validated on.
    """
    frames = []
    for ticker, df in prices.items():
        feats = compute_features(df)
        label = forward_return(df["Close"], config.BREAKOUT_WINDOW) >= config.BREAKOUT_RETURN
        panel = feats.copy()
        panel["label"] = label.astype("float")
        panel["ticker"] = ticker
        panel["date"] = panel.index
        frames.append(panel)

    if not frames:
        return pd.DataFrame(columns=[*FEATURE_COLUMNS, "label", "ticker", "date"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=FEATURE_COLUMNS + ["label"])
    return combined


def train(panel: pd.DataFrame) -> TrainedModel | None:
    if panel.empty:
        log.warning("Empty training panel -- skipping model training")
        return None

    panel = panel.sort_values("date")
    cutoff = panel["date"].max() - pd.Timedelta(days=int(config.VALIDATION_HOLDOUT_DAYS * 1.45))
    # drop the last BREAKOUT_WINDOW days entirely: their label is not yet
    # knowable (forward return would look into data we don't have).
    knowable = panel[panel["date"] <= panel["date"].max() - pd.Timedelta(days=config.BREAKOUT_WINDOW * 1.5)]

    train_rows = knowable[knowable["date"] <= cutoff]
    val_rows = knowable[knowable["date"] > cutoff]

    if train_rows["label"].sum() < 20 or val_rows.empty:
        log.warning("Not enough positive/labeled examples to train reliably (train pos=%s, val rows=%s)",
                    train_rows["label"].sum(), len(val_rows))
        if len(train_rows) < 50:
            return None

    X_train = train_rows[FEATURE_COLUMNS].values
    y_train = train_rows["label"].values
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    auc = float("nan")
    precision_top = float("nan")
    if not val_rows.empty and val_rows["label"].nunique() > 1:
        X_val_s = scaler.transform(val_rows[FEATURE_COLUMNS].values)
        y_val = val_rows["label"].values
        proba = clf.predict_proba(X_val_s)[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        top_decile_cut = np.quantile(proba, 0.9)
        top_mask = proba >= top_decile_cut
        if top_mask.sum() > 0:
            precision_top = float(precision_score(y_val, top_mask))

    return TrainedModel(
        scaler=scaler,
        clf=clf,
        validation_auc=auc,
        validation_precision_at_top_decile=precision_top,
        n_train_rows=len(train_rows),
        n_positive_train=int(train_rows["label"].sum()),
        n_validation_rows=len(val_rows),
    )


def score_today(model: TrainedModel, today_features: pd.DataFrame) -> pd.Series:
    """
    today_features: index=ticker, columns=FEATURE_COLUMNS (already computed,
    NaN rows should be dropped by the caller). Returns a 0-100 pattern score.
    """
    clean = today_features.dropna(subset=FEATURE_COLUMNS)
    if clean.empty:
        return pd.Series(dtype=float)
    X = model.scaler.transform(clean[FEATURE_COLUMNS].values)
    proba = model.clf.predict_proba(X)[:, 1]
    return pd.Series(proba * 100, index=clean.index, name="pattern_score")
