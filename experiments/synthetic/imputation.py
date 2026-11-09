"""Synthetic imputation under outliers (Figure 2, Figure 7 and Figure 8).

Two outputs are drawn from an ICM with a squared exponential kernel at N = 120 inputs
uniform on [0, 1] and observed with Gaussian noise. Output y1 is removed on 0.3 < x < 0.7
and 2.5% of output y2 near x = 0.5 is replaced by outliers. The MOGP imputes y1 through
its correlation with y2 and is pulled towards the outliers; R-MOGP down-weights them.

* Figure 2: MOGP vs R-MOGP.
* Figure 7: adds the non-conjugate t-MOGP and L-MOGP baselines.
* Figure 8: coverage of the predictive intervals with N = 1000 inputs, with and without outliers.

All models use the true hyperparameters (no optimisation) and the empirical mean of each
output as prior mean (the data are centred).

Usage:
    python imputation.py [--seed 77] [--n 120] [--n-coverage 1000] [--baseline-maxiter 1000]
"""

import argparse
from pathlib import Path

import gpflow
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from rmogp import ICM, MOGP, RMOGP
from rmogp.baselines import LMOGP, TMOGP

ROOT = Path(__file__).resolve().parents[2]

# Data-generating ICM: B = A A^T, so B_11 = 2, B_12 ~ 1.2, B_22 ~ 1.
A = np.array([[1.414, 0.0], [0.849, 0.529]])
LENGTHSCALE = 0.05
NOISE_VAR = 0.05
OUTLIER_FRACTION = 0.025
EPSILONS = [0.0, 0.025]  # expected proportion of outliers per output given to R-MOGP

COLORS = {"MOGP": "green", "R-MOGP": "royalblue", "$t$-MOGP": "orange", "L-MOGP": "#800020"}
DATA_COLOR = "black"  # data points and true function
RC_PARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 26,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
    "lines.linewidth": 6,
    "lines.markersize": 7,
}


def se_kernel(x, lengthscale):
    # Written exactly as in the original experiments: the prior covariance is nearly singular, so the
    # sample depends on floating-point details and this form makes the seed reproduce the paper's data.
    return np.exp(-0.5 / lengthscale**2 * (x**2 + (x**2).T - 2 * x @ x.T))


def simulate(n, seed, contaminated=True):
    """Return inputs, true functions, clean observations, observations and the outlier mask.

    f ~ GP(0, B (x) k) with k a unit-variance squared exponential kernel and y = f + N(0, 0.05) noise.
    Outliers replace a fraction of the y2 observations with 0.45 <= x <= 0.55 by draws from N(6, 0.5^2).
    The data (and the true functions) are centred by the empirical mean of the contaminated observations,
    and y1 is removed on 0.3 <= x <= 0.7.
    """
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(0, 1, n))[:, None]
    K = np.kron(A @ A.T, se_kernel(x, LENGTHSCALE))
    f = rng.multivariate_normal(np.zeros(2 * n), K).reshape(2, n).T
    y_clean = f + rng.normal(0, np.sqrt(NOISE_VAR), size=(n, 2))
    y = y_clean.copy()
    outliers = np.zeros(n, dtype=bool)
    if contaminated:
        candidates = np.where((x[:, 0] >= 0.45) & (x[:, 0] <= 0.55))[0]
        idx = rng.choice(candidates, int(OUTLIER_FRACTION * n), replace=False)
        y[idx, 1] = rng.normal(6, 0.5, size=len(idx))
        outliers[idx] = True
    shift = y.mean(axis=0)
    f, y_clean, y = f - shift, y_clean - shift, y - shift
    y[(x[:, 0] >= 0.3) & (x[:, 0] <= 0.7), 0] = np.nan
    return x, f, y_clean, y, outliers


def fit_models(x, y, baseline_maxiter):
    """MOGP and R-MOGP with the true ICM and noise; t-MOGP and L-MOGP with the same kernel.

    The t-MOGP has a Student-t likelihood with 3 degrees of freedom and scale sqrt(0.05 / 3) (noise
    variance 0.05); the L-MOGP a Laplace likelihood with scale 0.1. Their kernel and likelihood
    parameters are fixed and only the variational posterior is optimised.
    """
    def fit_baseline(model):
        model.fit(x, y)
        gpflow.utilities.set_trainable(model.model.kernel, False)
        gpflow.utilities.set_trainable(model.model.likelihood, False)
        model.optimize(maxiter=baseline_maxiter)
        return model

    return {
        "MOGP": MOGP(ICM(A, LENGTHSCALE), noise_var=NOISE_VAR).fit(x, y),
        "R-MOGP": RMOGP(ICM(A, LENGTHSCALE), noise_var=NOISE_VAR, epsilons=EPSILONS).fit(x, y),
        "$t$-MOGP": fit_baseline(TMOGP(2, df=3.0, scale=np.sqrt(NOISE_VAR / 3), A=A, lengthscale=LENGTHSCALE)),
        "L-MOGP": fit_baseline(LMOGP(2, scale=0.1, A=A, lengthscale=LENGTHSCALE)),
    }


def plot_panel(ax, data, t, mean, std, color):
    x, f, y, outliers, x_test = data
    if t == 0:
        ax.axvspan(0.3, 0.7, color="grey", alpha=0.2, zorder=0)
    red = outliers & (t == 1)
    ax.plot(x[~red, 0], y[~red, t], "o", color=DATA_COLOR, alpha=0.6)
    ax.plot(x[red, 0], y[red, t], "o", color="red", alpha=0.6)
    ax.plot(x[:, 0], f[:, t], "--", color=DATA_COLOR, alpha=0.6, linewidth=2)
    ax.plot(x_test[:, 0], mean[:, t], color=color)
    ax.fill_between(x_test[:, 0], mean[:, t] - 2 * std[:, t], mean[:, t] + 2 * std[:, t], color=color, alpha=0.2)
    ax.set(xlim=(0, 1), ylim=(-5, 7), xticks=[0, 0.5], yticks=[-4, 0, 4])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=1.4, length=3.5)


def legend_extras():
    true_line = mlines.Line2D([], [], color=DATA_COLOR, linestyle="--", alpha=0.6, linewidth=4, label="True")
    missing_patch = mpatches.Patch(color="grey", alpha=0.2, label="Missing")
    return [true_line, missing_patch]


def figure_2(data, predictions):
    fig, axs = plt.subplots(1, 4, figsize=(20, 3.5))
    for k, name in enumerate(["MOGP", "R-MOGP"]):
        for t in range(2):
            ax = axs[2 * k + t]
            plot_panel(ax, data, t, *predictions[name], COLORS[name])
            ax.set_title(f"$y_{t + 1}$", pad=10, fontsize=26)
            ax.set_xlabel("$x$", fontsize=26)
            if 2 * k + t > 0:
                ax.set_yticklabels([])
    handles = [mlines.Line2D([], [], color=COLORS[name], label=name) for name in ["MOGP", "R-MOGP"]]
    fig.legend(handles=handles + legend_extras(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.3),
               frameon=False)
    fig.subplots_adjust(wspace=0.05)
    return fig


def figure_7(data, predictions):
    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    for j, name in enumerate(predictions):
        for t in range(2):
            ax = axs[t, j]
            plot_panel(ax, data, t, *predictions[name], COLORS[name])
            if t == 0:
                ax.set_title(name, pad=10)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("$x$")
            if j == 0:
                ax.set_ylabel(f"$y_{t + 1}$")
            else:
                ax.set_yticklabels([])
    handles = [mlines.Line2D([], [], color=COLORS[name], label=name) for name in predictions]
    fig.legend(handles=handles + legend_extras(), loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.15),
               frameon=False)
    fig.subplots_adjust(wspace=0.1, hspace=0.15)
    return fig


def coverage_curves(n, seed, levels):
    """Proportion of clean observations (both outputs, including the missing region of y1) inside the
    central Gaussian predictive interval of nominal level q, predicting at the training inputs."""
    def coverage(y_true, mean, var):
        z = norm.ppf(0.5 + levels / 2)
        inside = np.abs(y_true - mean)[None] <= z[:, None, None] * np.sqrt(var)[None]
        return inside.mean(axis=(1, 2))

    curves = {}
    for scenario, contaminated in [("No outliers", False), ("Outliers", True)]:
        x, _, y_clean, y, _ = simulate(n, seed, contaminated)
        curves[scenario] = {
            "MOGP": coverage(y_clean, *MOGP(ICM(A, LENGTHSCALE), noise_var=NOISE_VAR).fit(x, y).predict_y(x)),
            "R-MOGP": coverage(y_clean, *RMOGP(ICM(A, LENGTHSCALE), noise_var=NOISE_VAR, epsilons=EPSILONS)
                               .fit(x, y).predict_y(x)),
        }
    return curves


def figure_8(levels, curves):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (scenario, scenario_curves) in zip(axs, curves.items()):
        for name, cov in scenario_curves.items():
            ax.plot(levels, cov, color=COLORS[name], linewidth=5, label=name)
        ax.plot([0, 1], [0, 1], "--", color=DATA_COLOR, alpha=0.6, linewidth=5, label="Ideal")
        ax.set_title(scenario, pad=15)
        ax.set(xlabel="$q$", xlim=(0, 1), ylim=(0, 1), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, 1.01, 0.2))
        ax.set_ylabel("Coverage", labelpad=12)
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(width=2, length=5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               frameon=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.24, wspace=0.40)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=77, help="seed of the simulated data")
    parser.add_argument("--n", type=int, default=120, help="number of inputs (Figures 2 and 7)")
    parser.add_argument("--n-coverage", type=int, default=1000, help="number of inputs (Figure 8)")
    parser.add_argument("--baseline-maxiter", type=int, default=1000, help="L-BFGS-B iterations of the baselines")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC_PARAMS)

    x, f, _, y, outliers = simulate(args.n, args.seed)
    x_test = np.linspace(0, 1, 100)[:, None]
    predictions = {}
    for name, model in fit_models(x, y, args.baseline_maxiter).items():
        mean, var = model.predict_y(x_test)
        predictions[name] = (mean, np.sqrt(var))
    data = (x, f, y, outliers, x_test)
    figure_2(data, predictions).savefig(args.fig_dir / "imputation.pdf", bbox_inches="tight")
    figure_7(data, predictions).savefig(args.fig_dir / "imputation_baselines.pdf", bbox_inches="tight")

    levels = np.concatenate([[0.0], np.arange(0.04, 0.97, 0.04), [1.0]])
    curves = coverage_curves(args.n_coverage, args.seed, levels)
    figure_8(levels, curves).savefig(args.fig_dir / "coverage.pdf", bbox_inches="tight")
    print(f"Figures saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
