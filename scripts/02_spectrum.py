from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
returns = np.log(prices).diff().dropna()
T, N = returns.shape
q = N / T
print(f"N = {N} stocks, T = {T} days, q = N/T = {q:.3f}")

X = (returns - returns.mean()) / returns.std()    
C = X.T @ X / (T - 1)                          
eigenvalues = np.linalg.eigvalsh(C)[::-1]

sigma2 = 1 - eigenvalues[0] / N
lam_min = sigma2 * (1 - np.sqrt(q)) ** 2
lam_max = sigma2 * (1 + np.sqrt(q)) ** 2
n_signal = (eigenvalues > lam_max).sum()

print(f"Largest eigenvalue: {eigenvalues[0]:.1f} "
      f"({eigenvalues[0] / N:.0%} of total variance)")
print("Next 10 eigenvalues:", np.round(eigenvalues[1:11], 2))
print(f"Marchenko-Pastur noise band: [{lam_min:.2f}, {lam_max:.2f}]")
print(f"Eigenvalues above the noise band: {n_signal} out of {N}")


def marchenko_pastur(lam, q, sigma2):
    lo = sigma2 * (1 - np.sqrt(q)) ** 2
    hi = sigma2 * (1 + np.sqrt(q)) ** 2
    density = np.zeros_like(lam)
    inside = (lam > lo) & (lam < hi)
    density[inside] = (np.sqrt((hi - lam[inside]) * (lam[inside] - lo))
                       / (2 * np.pi * q * sigma2 * lam[inside]))
    return density


fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(eigenvalues, bins=np.linspace(0, 4, 80), density=True, alpha=0.6,
        label="Eigenvalues of the real correlation matrix")
x = np.linspace(0.01, 4, 1000)
ax.plot(x, marchenko_pastur(x, q, sigma2), color="black",
        label="Marchenko-Pastur (pure noise)")
ax.axvline(lam_max, color="red", ls="--", label=f"Noise edge = {lam_max:.2f}")
ax.text(3.95, ax.get_ylim()[1] * 0.5,
        f"Largest eigenvalue = {eigenvalues[0]:.0f}\n(market mode, off the chart)",
        ha="right")
ax.set(xlabel="Eigenvalue", ylabel="Density",
       title=f"Spectrum of the correlation matrix of {N} US stocks, 2015-2025")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "spectrum.png", dpi=150)
print("Figure saved to", FIG / "spectrum.png")