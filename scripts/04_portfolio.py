from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
WINDOW = 252   
HOLD = 63    

prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
returns = np.log(prices).diff().dropna()
T, N = returns.shape
q = N / WINDOW
print(f"N = {N} stocks, window = {WINDOW} days, q = {q:.2f}")


def clip_correlation(C, q):
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    lam_max = (1 + np.sqrt(q)) ** 2
    noise = eigenvalues < lam_max
    cleaned = eigenvalues.copy()
    cleaned[noise] = eigenvalues[noise].mean()    
    C_clean = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    d = np.sqrt(np.diag(C_clean))
    return C_clean / np.outer(d, d)              


def min_variance_weights(cov):
    w = np.linalg.solve(cov, np.ones(len(cov)))
    return w / w.sum()


results = []
for start in range(0, T - WINDOW - HOLD + 1, HOLD):
    past = returns.iloc[start:start + WINDOW]
    future = returns.iloc[start + WINDOW:start + WINDOW + HOLD]
    vol = past.std().values
    C_raw = np.corrcoef(past.values.T)

    C_clean = clip_correlation(C_raw, q)
    cov_raw = C_raw * np.outer(vol, vol)      
    cov_clean = C_clean * np.outer(vol, vol)

    portfolios = {
        "Raw correlations": (cov_raw, min_variance_weights(cov_raw)),
        "RMT-cleaned": (cov_clean, min_variance_weights(cov_clean)),
        "Equal weight": (cov_raw, np.ones(N) / N),
    }
    for method, (cov, w) in portfolios.items():
        results.append({
            "date": future.index[0],
            "method": method,
            "predicted": np.sqrt(w @ cov @ w * 252),
            "realised": (future.values @ w).std() * np.sqrt(252),
            "gross": np.abs(w).sum(),
        })

results = pd.DataFrame(results)
summary = results.groupby("method")[["predicted", "realised", "gross"]].mean()
summary["realised / predicted"] = summary["realised"] / summary["predicted"]
print(f"\nAverage over {results['date'].nunique()} rebalancing periods "
      "(annualised volatility):")
print(summary.round(3))

order = ["Equal weight", "Raw correlations", "RMT-cleaned"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                               gridspec_kw={"width_ratios": [1, 2]})
x = np.arange(len(order))
ax1.bar(x - 0.2, summary.loc[order, "predicted"], width=0.4, label="Predicted")
ax1.bar(x + 0.2, summary.loc[order, "realised"], width=0.4, label="Realised")
ax1.set_xticks(x)
ax1.set_xticklabels(order, rotation=15)
ax1.set(ylabel="Annualised volatility", title="Predicted vs realised risk")
ax1.legend()

for method in order:
    r = results[results["method"] == method]
    ax2.plot(r["date"], r["realised"], marker="o", ms=3, label=method)
ax2.set(ylabel="Realised volatility (next 3 months)",
        title="Out-of-sample risk of each portfolio")
ax2.legend()

fig.tight_layout()
fig.savefig(FIG / "portfolio_risk.png", dpi=150)
print("Figure saved to", FIG / "portfolio_risk.png")