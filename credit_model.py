import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PATH = Path(__file__).parent / "model" / "pd_model.joblib"
# never trained on: ids, free text, geography (proxy risk), dates, and Lending Club's own pricing/score
DROP = re.compile(r"(^id$|member_id|url|desc|title|zip|addr_state|grade|int_rate|installment|_d$|earliest_cr|loan_status|policy_code)", re.I)


def numify(s):
    x = pd.to_numeric(s.astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    return x if x.notna().mean() > 0.9 and s.nunique() > 1 else None


def prep(df):
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            x = numify(df[c])
            if x is not None:
                df[c] = x
    return df


def find_target(df):
    for c in df.columns:
        if c.lower() in ("target", "default", "loan_status", "bad", "charged_off", "label", "y"):
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and s.dropna().nunique() <= 2:
                return c, s.astype(float)
            low = s.astype(str).str.lower()
            y = pd.Series(np.nan, index=s.index)
            y[low.str.contains("charged|default")] = 1.0
            y[low.str.contains("paid") & ~low.str.contains("charged|default|late|grace")] = 0.0
            if y.notna().mean() > 0.5:
                return c, y
    raise SystemExit("Could not find the target column. Columns: " + ", ".join(df.columns))


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return {"auc": round(float(auc), 4), "gini": round(float(2 * auc - 1), 4), "ks": round(float((tpr - fpr).max()), 4)}


def pipelines(num, cat):
    lr = Pipeline([
        ("t", ColumnTransformer([
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
            ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                            ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=50))]), cat)])),
        ("m", LogisticRegression(max_iter=300))])
    hgb = Pipeline([
        ("t", ColumnTransformer([
            ("n", "passthrough", num),
            ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan), cat)])),
        ("m", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0,
                                             categorical_features=[False] * len(num) + [True] * len(cat)))])
    return {"logistic": lr, "boosting": hgb}


def train(csv, max_rows=500000):
    raw = pd.read_csv(csv, low_memory=False)
    tcol, y = find_target(raw)
    ok = y.notna()
    raw, y = raw[ok], y[ok]
    print(f"target column: {tcol}, usable rows: {len(raw):,}")
    if len(raw) > max_rows:
        keep = raw.sample(max_rows, random_state=0).index
        raw, y = raw.loc[keep], y.loc[keep]
    if y.mean() > 0.5:
        y = 1 - y
        print("target looked inverted, flipped so 1 = default")
    dcol = next((c for c in raw.columns if c.lower() == "issue_d"), None)
    when = pd.to_datetime(raw[dcol], format="%b-%Y", errors="coerce") if dcol else None
    if when is not None and when.notna().mean() < 0.9:
        when = pd.to_datetime(raw[dcol], errors="coerce")
    if when is not None and when.notna().mean() > 0.9:
        order = np.argsort(when.values, kind="stable")
        print("splitting by issue date: train = oldest 70%, test = newest 20%")
    else:
        order = np.random.RandomState(0).permutation(len(raw))
        print("no issue date found, random split")
    X = prep(raw.drop(columns=[c for c in raw.columns if c == tcol or DROP.search(c)])).iloc[order]
    y = y.values[order]
    X = X.loc[:, (X.isna().mean() < 0.6) & (X.nunique() > 1)]
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num and X[c].nunique() <= 50]
    X = X[num + cat]
    print("numeric features:", num)
    print("categorical features:", cat)
    n = len(X)
    a, b = int(n * 0.7), int(n * 0.8)
    Xtr, Xca, Xte = X.iloc[:a], X.iloc[a:b], X.iloc[b:]
    ytr, yca, yte = y[:a], y[a:b], y[b:]
    fitted = {}
    for name, pipe in pipelines(num, cat).items():
        pipe.fit(Xtr, ytr)
        fitted[name] = (pipe, roc_auc_score(yca, pipe.predict_proba(Xca)[:, 1]))
        print(f"{name}: validation AUC {fitted[name][1]:.4f}")
    name = max(fitted, key=lambda k: fitted[k][1])
    model = fitted[name][0]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(model.predict_proba(Xca)[:, 1], yca)
    p_te = iso.predict(model.predict_proba(Xte)[:, 1])
    m = metrics(yte, p_te)
    print(f"\nchosen model: {name}")
    print("TEST (newest loans):", m)
    d = pd.DataFrame({"predicted": p_te, "actual": yte})
    d["bin"] = pd.qcut(d["predicted"].rank(method="first"), 10, labels=False)
    print("\nrisk deciles (low to high) - predicted vs actual default rate:")
    print(d.groupby("bin")[["predicted", "actual"]].mean().round(3).to_string())
    from sklearn.inspection import permutation_importance
    idx = np.random.RandomState(0).choice(len(Xte), min(4000, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte.iloc[idx], yte[idx], scoring="roc_auc", n_repeats=2, random_state=0)
    importance = dict(sorted(zip(num + cat, pi.importances_mean.round(5).tolist()), key=lambda kv: -kv[1]))
    print("\nmost important inputs:", list(importance)[:10])
    bundle = {
        "model": model, "iso": iso, "num": num, "cat": cat, "features": num + cat, "target": tcol,
        "baseline": {**{c: float(X[c].median()) for c in num}, **{c: X[c].mode().iat[0] for c in cat}},
        "cat_values": {c: X[c].value_counts().head(30).index.tolist() for c in cat},
        "importance": importance, "metrics": {**m, "model": name, "rows": int(n), "train_default_rate": round(float(ytr.mean()), 4),
                    "test_default_rate": round(float(yte.mean()), 4)},
    }
    PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, PATH)
    print("\nsaved:", PATH)


def _match(options, v):
    s = str(v).strip().lower()
    for o in options:
        if str(o).strip().lower() == s:
            return o
    d = re.findall(r"\d+", s)
    for o in options:
        if d and re.findall(r"\d+", str(o)) == d:
            return o
    return v


def frame(b, cases):
    rows = []
    for c in cases:
        r = {}
        for f in b["features"]:
            v = c.get(f)
            if v in (None, ""):
                r[f] = np.nan
            elif f in b["num"]:
                try:
                    r[f] = float(v)
                except (TypeError, ValueError):
                    r[f] = np.nan
            else:
                r[f] = _match(b["cat_values"][f], v)
        rows.append(r)
    df = pd.DataFrame(rows, columns=b["features"])
    df[b["num"]] = df[b["num"]].astype(float)
    df[b["cat"]] = df[b["cat"]].astype(object)
    return df


def score(b, cases, calibrated=True):
    p = b["model"].predict_proba(frame(b, cases))[:, 1]
    return b["iso"].predict(p) if calibrated else p


def reasons(b, case, k=4):
    used = [f for f in b["features"] if case.get(f) not in (None, "")]
    if not used:
        return {"up": [], "down": []}
    p = score(b, [case] + [{**case, f: b["baseline"][f]} for f in used], calibrated=False)
    out = sorted(({"feature": f, "value": case[f], "typical": b["baseline"][f], "effect_pts": round(float((p[0] - q) * 100), 2)}
                  for f, q in zip(used, p[1:])), key=lambda r: -r["effect_pts"])
    return {"up": [r for r in out if r["effect_pts"] > 0.1][:k],
            "down": [r for r in out[::-1] if r["effect_pts"] < -0.1][:2]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python credit_model.py path\\to\\loans.csv [max_rows]")
    train(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500000)
