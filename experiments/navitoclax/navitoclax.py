"""Navitoclax dose-response (Figure 1, Figure 12, IC50 estimates and Table 5).

We model the viability of two cancer cell lines, P12-ICHIKAWA (y1) and ARH-77 (y2), as a
function of the log10 dose of the drug Navitoclax with an ICM multi-output GP (T = 2
outputs, N = 21 doses). P12-ICHIKAWA contains a natural outlier at dose 10^0 uM. We compare
the standard MOGP (marginal likelihood), R-MOGP (weighted LOO-CV), and the non-conjugate
t-MOGP and L-MOGP, all with constant prior mean m_t(x) = 0.5.

Data: ``navitoclax.csv`` holds the Navitoclax rows of the two cell lines, taken from the
GDSC2 raw data (Genomics of Drug Sensitivity in Cancer, bulk download at
https://www.cancerrxgene.org/downloads/bulk_download). Viability is the GDSC-normalised
cell viability with replicates averaged, as described in Appendix C.11; ``log10_dose`` is
the dose in log10 uM.

Usage:
    python navitoclax.py [--epsilon 0.11] [--baseline-maxiter 1000]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rmogp import ICM, MOGP, RMOGP, robust_covariance
from rmogp.baselines import LMOGP, TMOGP

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "navitoclax.csv"

CELL_LINES = ["P12-ICHIKAWA", "ARH-77"]
OUTLIER_ROW = 15  # Y[15, 0]: P12-ICHIKAWA at log10 dose 0
PRIOR_MEAN = 0.5
INIT_LENGTHSCALE = 1.0
INIT_NOISE_VAR = 0.01
MCD_SUPPORT_FRACTION = 0.9
RMOGP_TOL = 1e-2  # L-BFGS-B tolerance of the R-MOGP LOO-CV objective
BASELINE_KERNEL_VAR = 0.1
BASELINE_W_INIT = 0.1
T_DF = 3.0
T_SCALE = 1.0
L_SCALE = 0.1

COLORS = {"MOGP": "green", "R-MOGP": "royalblue", "t-MOGP": "orange", "L-MOGP": "#800020"}
LABELS = {"t-MOGP": r"$t$-MOGP"}
ALPHAS = {"MOGP": 0.2, "R-MOGP": 0.4, "t-MOGP": 0.4, "L-MOGP": 0.4}
TITLES = [r"P12-ICHIKAWA ($y_1$)", r"ARH-77 ($y_2$)"]
XLABEL = r"Navitoclax dose ($\log_{10}\,\mu$M)"
RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
}


def load_data():
    """Doses ``x`` (N, 1) and the (N, T) viability matrix ``Y``, one column per cell line."""
    table = pd.read_csv(DATA).pivot(index="log10_dose", columns="cell_line", values="viability")[CELL_LINES]
    return table.index.to_numpy()[:, None], table.to_numpy()


def fit_models(x, Y, epsilon, baseline_maxiter):
    """Fit all models; return the fitted models and their runtimes (s).

    * MOGP / R-MOGP: ICM kernel with A initialised as the Cholesky factor of a FastMCD estimate of
      the output covariance (fit on the observed data). MOGP maximises the marginal likelihood; R-MOGP
      minimises the weighted LOO-CV objective with weights fixed at the FastMCD covariance (power 2).
    * t-MOGP / L-MOGP: GPflow VGP with a Student-t (3 degrees of freedom) or Laplace likelihood, a
      fixed RBF variance of 0.1 and a coregionalisation matrix initialised at 0.1.
    * MOGP (no outlier): the MOGP fitted with the outlier removed from the data (Table 5 only).

    The runtime covers the FastMCD initialisation and the hyperparameter optimisation.
    """
    T = Y.shape[1]

    def init_kernel(Y_train):
        cov = robust_covariance(Y_train, support_fraction=MCD_SUPPORT_FRACTION)
        return ICM(np.linalg.cholesky(cov), INIT_LENGTHSCALE), cov

    def fit_mogp(Y_train):
        kernel, _ = init_kernel(Y_train)
        model = MOGP(kernel, noise_var=INIT_NOISE_VAR, mean=PRIOR_MEAN).fit(x, Y_train)
        model.optimize()
        return model

    def fit_rmogp(Y_train):
        kernel, cov = init_kernel(Y_train)
        model = RMOGP(kernel, noise_var=INIT_NOISE_VAR, mean=PRIOR_MEAN, epsilons=epsilon).fit(x, Y_train)
        model.optimize(init_cov=cov, weight_power=2, tol=RMOGP_TOL)
        return model

    def fit_baseline(model_class, Y_train, **likelihood):
        model = model_class(T, A=np.full((T, T), BASELINE_W_INIT), lengthscale=INIT_LENGTHSCALE,
                            kernel_variance=BASELINE_KERNEL_VAR, mean=PRIOR_MEAN, **likelihood).fit(x, Y_train)
        model.optimize(maxiter=baseline_maxiter)
        return model

    Y_no_outlier = Y.copy()
    Y_no_outlier[OUTLIER_ROW, 0] = np.nan
    fitters = {
        "MOGP": lambda: fit_mogp(Y),
        "R-MOGP": lambda: fit_rmogp(Y),
        "t-MOGP": lambda: fit_baseline(TMOGP, Y, df=T_DF, scale=T_SCALE),
        "L-MOGP": lambda: fit_baseline(LMOGP, Y, scale=L_SCALE),
        "MOGP (no outlier)": lambda: fit_mogp(Y_no_outlier),
    }
    models, runtimes = {}, {}
    for name, fit in fitters.items():
        start = time.perf_counter()
        models[name] = fit()
        runtimes[name] = time.perf_counter() - start
    return models, runtimes


def plot_fit(ax, x_test, prediction, name, t):
    mean, std = prediction
    ax.plot(x_test[:, 0], mean[:, t], color=COLORS[name], linewidth=5, label=LABELS.get(name, name))
    ax.fill_between(x_test[:, 0], mean[:, t] - 2 * std[:, t], mean[:, t] + 2 * std[:, t],
                    color=COLORS[name], alpha=ALPHAS[name])


def plot_data(ax, x, Y, t):
    ax.plot(x[:, 0], Y[:, t], "o", color="black", alpha=0.6, markersize=7)
    if t == 0:
        ax.plot(x[OUTLIER_ROW, 0], Y[OUTLIER_ROW, 0], "o", color="red", markersize=7)
    ax.set_xlim(x[0, 0], x[-1, 0])
    ax.set_ylim(-0.5, 1.8)
    for spine in ax.spines.values():
        spine.set_linewidth(2)


def figure_1(x, Y, x_test, predictions):
    """MOGP vs R-MOGP; bands show +-2 predictive standard deviations."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for t, ax in enumerate(axs):
        plot_data(ax, x, Y, t)
        for name in ["MOGP", "R-MOGP"]:
            plot_fit(ax, x_test, predictions[name], name, t)
        ax.set_title(TITLES[t], pad=10)
    axs[0].set_ylabel("Viability", labelpad=15)
    axs[1].set_yticklabels([])
    axs[1].legend(loc="lower left", frameon=False, borderpad=0.1, labelspacing=0.4)
    fig.text(0.55, -0.01, XLABEL, ha="center")
    fig.tight_layout()
    return fig


def figure_12(x, Y, x_test, predictions):
    """All four methods; bands show +-2 predictive standard deviations."""
    fig, axs = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for col, name in enumerate(COLORS):
        for t in range(2):
            ax = axs[t, col]
            plot_data(ax, x, Y, t)
            plot_fit(ax, x_test, predictions[name], name, t)
            if col == 0:
                ax.set_ylabel(TITLES[t], fontsize=20, labelpad=15)
            else:
                ax.set_yticklabels([])
        axs[0, col].set_title(LABELS.get(name, name), pad=10)
    fig.text(0.5, -0.02, XLABEL, ha="center")
    fig.tight_layout(pad=0.8, w_pad=0.6, h_pad=0.6)
    return fig


def ic50(x_test, mean, std):
    """IC50 (log10 dose where the posterior mean of f1 equals 0.5) and its 95% interval.

    The interval is the range of doses around the IC50 for which 0.5 lies inside the pointwise 95%
    credible band of the latent function f1 (posterior mean +- 1.96 posterior std, without noise).
    """
    crossing = np.where(np.diff(np.sign(mean - 0.5)) != 0)[0][0]
    x0, x1 = x_test[crossing, 0], x_test[crossing + 1, 0]
    m0, m1 = mean[crossing], mean[crossing + 1]
    inside = np.abs(mean - 0.5) <= 1.96 * std
    lo = hi = crossing
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(inside) - 1 and inside[hi + 1]:
        hi += 1
    return x0 + (0.5 - m0) * (x1 - x0) / (m1 - m0), x_test[lo, 0], x_test[hi, 0]


def table_5(x, Y, models, runtimes):
    """In-sample RMSE and NLPD of the predictive distribution over the 20 non-outlying observations of
    P12-ICHIKAWA (the outlier itself is excluded), with standard errors, and the runtime of the fit."""
    keep = np.arange(len(Y)) != OUTLIER_ROW
    y1 = Y[keep, 0]
    n = len(y1)
    rows = {}
    for name in ["MOGP (no outlier)", *COLORS]:
        mean, var = models[name].predict_y(x)
        mean, var = mean[keep, 0], var[keep, 0]
        sq_err = (y1 - mean) ** 2
        nlpd_i = 0.5 * np.log(2 * np.pi * var) + sq_err / (2 * var)
        rmse = np.sqrt(sq_err.mean())
        rows[name] = {
            "RMSE": f"{rmse:.3f} ± {sq_err.std() / (2 * np.sqrt(n) * rmse):.3f}",
            "NLPD": f"{nlpd_i.mean():.3f} ± {nlpd_i.std() / np.sqrt(n):.3f}",
            "Runtime (s)": f"{runtimes[name]:.2f}",
        }
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epsilon", type=float, default=0.11, help="R-MOGP expected proportion of outliers")
    parser.add_argument("--baseline-maxiter", type=int, default=1000, help="L-BFGS-B iterations of the baselines")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC_PARAMS)

    x, Y = load_data()
    x_test = np.linspace(-2.7, 1.5, 200)[:, None]
    models, runtimes = fit_models(x, Y, args.epsilon, args.baseline_maxiter)

    predictions = {}
    for name in COLORS:
        mean, var = models[name].predict_y(x_test)
        predictions[name] = (mean, np.sqrt(var))
    figure_1(x, Y, x_test, predictions).savefig(args.fig_dir / "navitoclax_rmogp.pdf", bbox_inches="tight")
    figure_12(x, Y, x_test, predictions).savefig(args.fig_dir / "navitoclax_all_methods.pdf", bbox_inches="tight")

    ic50_rows = {}
    for name in COLORS:
        mean, var = models[name].predict_f(x_test)
        ic50_rows[name] = ic50(x_test, mean[:, 0], np.sqrt(var[:, 0]))
    ic50_table = pd.DataFrame.from_dict(ic50_rows, orient="index", columns=["IC50", "95% lower", "95% upper"])
    table5 = table_5(x, Y, models, runtimes)
    ic50_table.to_csv(args.results_dir / "navitoclax_ic50.csv")
    table5.to_csv(args.results_dir / "navitoclax_table5.csv")
    print(f"IC50 of P12-ICHIKAWA (log10 uM)\n{ic50_table.round(3).to_string()}\n")
    print(f"Table 5\n{table5.to_string()}")


if __name__ == "__main__":
    main()
