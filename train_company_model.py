"""Trains the company (business lending) risk model on SYNTHETIC data - there is no
real company-default dataset analogous to Lending Club available for this prototype, so
a synthetic generator stands in for one, with a fixed seed for reproducibility. Reuses
credit_model.py's generic pipeline/metric helpers (they're written against a passed-in
bundle, not hardcoded to personal-loan fields) rather than duplicating them.

Run: python train_company_model.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import credit_model as cm

PATH = Path(__file__).parent / "model" / "company_risk_model.joblib"
N = 8000
INDUSTRIES = ["retail", "manufacturing", "technology", "construction", "healthcare", "hospitality", "professional_services"]
# fictional baseline risk adjustment per industry, in logit points - construction/hospitality
# riskier (cyclical, thin margins), healthcare/professional_services steadier
INDUSTRY_RISK = {"retail": 0.15, "manufacturing": 0.05, "technology": 0.0, "construction": 0.45,
                  "healthcare": -0.35, "hospitality": 0.55, "professional_services": -0.2}
DEBT_SERVICE_RATE = 0.20  # assumed ~5-year average amortization, for a synthetic DSCR


def generate(n=N, seed=0):
    rng = np.random.RandomState(seed)
    revenue = np.exp(rng.normal(np.log(5_000_000), 1.3, n)).clip(150_000, 800_000_000)
    years = rng.gamma(2.0, 4.0, n).clip(0, 60)
    industry = rng.choice(INDUSTRIES, n)
    # debt and cash flow both scale with revenue but with independent noise, so leverage
    # and margin vary realistically rather than being deterministic functions of revenue
    total_debt = revenue * np.exp(rng.normal(0.3, 0.6, n)) * 0.6
    margin = rng.normal(0.08, 0.10, n)  # operating margin, can go negative (distressed)
    operating_cash_flow = revenue * margin
    leverage = total_debt / revenue
    debt_service = total_debt * DEBT_SERVICE_RATE
    dscr = operating_cash_flow / debt_service.clip(1)

    risk = (
        1.4 * leverage.clip(0, 8)
        - 1.0 * dscr.clip(-3, 5)
        - 0.02 * years
        - 0.18 * np.log(revenue / 1_000_000).clip(-2, 8)
        + np.array([INDUSTRY_RISK[i] for i in industry])
        + rng.normal(0, 1.5, n)  # unobserved factors - keeps this genuinely uncertain, not trivially separable
    )
    p_default = 1 / (1 + np.exp(-(risk - 2.6)))  # offset tuned so overall default rate lands near a plausible 15-20%
    y = rng.binomial(1, p_default)

    return pd.DataFrame({
        "revenue": revenue, "total_debt": total_debt, "operating_cash_flow": operating_cash_flow,
        "years_in_business": years, "leverage": leverage, "dscr": dscr, "industry": industry,
    }), y


def train():
    X, y = generate()
    num, cat = ["revenue", "total_debt", "operating_cash_flow", "years_in_business", "leverage", "dscr"], ["industry"]
    n = len(X)
    order = np.random.RandomState(0).permutation(n)
    X, y = X.iloc[order].reset_index(drop=True), y[order]
    a, b = int(n * 0.7), int(n * 0.8)
    Xtr, Xca, Xte = X.iloc[:a], X.iloc[a:b], X.iloc[b:]
    ytr, yca, yte = y[:a], y[a:b], y[b:]

    fitted = {}
    for name, pipe in cm.pipelines(num, cat).items():
        pipe.fit(Xtr, ytr)
        fitted[name] = (pipe, roc_auc_score(yca, pipe.predict_proba(Xca)[:, 1]))
        print(f"{name}: validation AUC {fitted[name][1]:.4f}")
    name = max(fitted, key=lambda k: fitted[k][1])
    model = fitted[name][0]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(model.predict_proba(Xca)[:, 1], yca)
    p_te = iso.predict(model.predict_proba(Xte)[:, 1])
    m = cm.metrics(yte, p_te)
    print(f"\nchosen model: {name}")
    print("TEST:", m)

    idx = np.random.RandomState(0).choice(len(Xte), min(2000, len(Xte)), replace=False)
    pi = permutation_importance(model, Xte.iloc[idx], yte[idx], scoring="roc_auc", n_repeats=2, random_state=0)
    importance = dict(sorted(zip(num + cat, pi.importances_mean.round(5).tolist()), key=lambda kv: -kv[1]))
    print("most important inputs:", list(importance)[:10])

    bundle = {
        "model": model, "iso": iso, "num": num, "cat": cat, "features": num + cat, "target": "synthetic_default",
        "baseline": {**{c: float(X[c].median()) for c in num}, **{c: X[c].mode().iat[0] for c in cat}},
        "cat_values": {c: X[c].value_counts().head(30).index.tolist() for c in cat},
        "importance": importance,
        "metrics": {**m, "model": name, "rows": int(n), "train_default_rate": round(float(ytr.mean()), 4),
                    "test_default_rate": round(float(yte.mean()), 4)},
        "note": "trained on synthetic data - no real company-default dataset was used",
    }
    PATH.parent.mkdir(exist_ok=True)
    import joblib
    joblib.dump(bundle, PATH)
    print("\nsaved:", PATH)


if __name__ == "__main__":
    train()
