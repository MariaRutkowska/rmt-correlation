from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
N_SHOW = 6 

prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
returns = np.log(prices).diff().dropna()
T, N = returns.shape
X = (returns - returns.mean()) / returns.std()
C = X.T @ X / (T - 1)
eigenvalues, eigenvectors = np.linalg.eigh(C)
eigenvalues = eigenvalues[::-1]        
eigenvectors = eigenvectors[:, ::-1]    

sectors = pd.read_csv(ROOT / "tickers.csv", index_col="ticker")["sector"]
sectors = sectors.loc[returns.columns]

print(f"{'k':>2} {'eigenvalue':>10} {'stocks involved':>16}  main sector (share of weight)")
for k in range(N_SHOW):
    v = eigenvectors[:, k]
    ipr = np.sum(v ** 4)                
    weight_by_sector = pd.Series(v ** 2, index=sectors.index).groupby(sectors).sum()
    top_sector = weight_by_sector.idxmax()
    print(f"{k + 1:>2} {eigenvalues[k]:>10.2f} {1 / ipr:>16.0f}  "
          f"{top_sector} ({weight_by_sector.max():.0%})")

market_mode = X.values @ eigenvectors[:, 0]
average_stock = returns.mean(axis=1)
corr = np.corrcoef(market_mode, average_stock)[0, 1]
print(f"\nCorrelation of eigenvector 1 with the average stock return: {abs(corr):.3f}")

order = np.argsort(sectors.values, kind="stable")
sector_names = sorted(sectors.unique())
colors = plt.cm.tab20(np.linspace(0, 1, len(sector_names)))
color_of = dict(zip(sector_names, colors))
bar_colors = [color_of[s] for s in sectors.values[order]]

fig, axes = plt.subplots(N_SHOW, 1, figsize=(10, 2 * N_SHOW), sharex=True)
for k, ax in enumerate(axes):
    v = eigenvectors[order, k]
    if v[np.argmax(np.abs(v))] < 0:  
        v = -v
    ax.bar(range(N), v, color=bar_colors, width=1.0)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(f"#{k + 1}\nλ={eigenvalues[k]:.1f}")
axes[-1].set_xlabel("Stocks, grouped by sector")
handles = [plt.Rectangle((0, 0), 1, 1, color=color_of[s]) for s in sector_names]
fig.legend(handles, sector_names, loc="upper center", ncol=4, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "eigenvectors.png", dpi=150)
print("Figure saved to", FIG / "eigenvectors.png")