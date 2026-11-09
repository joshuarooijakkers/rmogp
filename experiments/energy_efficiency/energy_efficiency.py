"""UCI Energy Efficiency: robust MOGPs (Table 2, Figure 3 and Figure 4).

The UCI Energy Efficiency dataset has N = 768 buildings, d = 8 covariates and T = 2 outputs
(heating load and cooling load). For each of 10 random 75/25 train/test splits, we add
outliers to 10% of the heating-load training observations and compare a standard MOGP,
R-MOGP and two non-conjugate robust MOGPs with a Student-t (t-MOGP) and a Laplace (L-MOGP)
likelihood. All methods use an ICM kernel with a squared exponential base kernel. We report
test RMSE, test NLPD and runtimes.

Outlier scenarios (added to the standardised training outputs; the test set stays clean):

* ``none``: no outliers;
* ``uniform``: +-U(6, 9) with a random sign per outlier;
* ``asymmetric``: +U(6, 9);
* ``student_t``: additive t_3(0, 5) noise (Student-t with 3 degrees of freedom and scale 5).

The data are downloaded from the UCI repository on first use and cached in ``data/``.
The full experiment takes about a day; use ``--maxiter`` and ``--n-splits`` for a quick run.

Usage:
    python energy_efficiency.py [--n-splits 10] [--scenarios none uniform asymmetric student_t] [--maxiter N]
"""

import argparse
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rmogp import ICM, MOGP, RMOGP, robust_covariance
from rmogp.baselines import LMOGP, TMOGP
from rmogp.metrics import nlpd, rmse

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
FILENAME = "ENB2012_data.xlsx"

SCENARIOS = {"none": "No outliers", "uniform": "Uniform", "asymmetric": "Asymmetric", "student_t": "Student-t"}
OUTLIER_T_DF = 3.0
OUTLIER_T_SCALE = 5.0
TOL = {"MOGP": 1e-5, "R-MOGP": 1e-5, "t-MOGP": 1e-5, "L-MOGP": 1e-6}  # optimiser tolerances
METHODS = list(TOL)
STYLE = {
    "MOGP": dict(color="green", marker="D", label="MOGP"),
    "R-MOGP": dict(color="royalblue", marker="s", label="R-MOGP"),
    "t-MOGP": dict(color="orange", marker="o", label=r"$t$-MOGP"),
    "L-MOGP": dict(color="#800020", marker="^", label="L-MOGP"),
}


def load_energy():
    """Inputs ``X`` (768, 8) and outputs ``Y`` (768, 2) = (heating load, cooling load)."""
    path = DATA_DIR / FILENAME
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extract(FILENAME, DATA_DIR)
    df = pd.read_excel(path)
    return df.loc[:, "X1":"X8"].to_numpy(dtype=float), df.loc[:, ["Y1", "Y2"]].to_numpy(dtype=float)


def contaminate(Y, scenario, eps, rng):
    """Add outliers to a fraction ``eps`` of the rows of the first output (heating load)."""
    Y = Y.copy()
    if scenario == "none":
        return Y
    n_out = int(round(eps * len(Y)))
    rows = rng.choice(len(Y), size=n_out, replace=False)
    if scenario == "uniform":
        noise = rng.uniform(6.0, 9.0, size=n_out) * rng.choice([-1.0, 1.0], size=n_out)
    elif scenario == "asymmetric":
        noise = rng.uniform(6.0, 9.0, size=n_out)
    else:
        noise = OUTLIER_T_SCALE * rng.standard_t(OUTLIER_T_DF, size=n_out)
    Y[rows, 0] += noise
    return Y


def make_split(X, Y, split, scenario, eps):
    """Random 75/25 split ``split``, standardised with training statistics; only training outputs are contaminated."""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=split)
    x_scaler, y_scaler = StandardScaler().fit(X_train), StandardScaler().fit(Y_train)
    X_train, X_test = x_scaler.transform(X_train), x_scaler.transform(X_test)
    Y_train, Y_test = y_scaler.transform(Y_train), y_scaler.transform(Y_test)
    Y_train = contaminate(Y_train, scenario, eps, np.random.default_rng(split))
    return X_train, Y_train, X_test, Y_test


def build_model(method, X_train, Y_train, eps):
    """Return a model fitted to the data and the keyword arguments of its ``optimize``.

    * MOGP and R-MOGP: A initialised as the Cholesky factor of a FastMCD estimate of the output
      covariance of the observed (contaminated) training outputs; lengthscale 1 and noise variance 0.1,
      shared by both outputs (as in the GPflow baselines). MOGP uses the standard LOO-CV, R-MOGP the
      weighted LOO-CV of Section 3.3 with epsilons (eps, 0). The prior mean is zero (standardised data).
    * t-MOGP and L-MOGP (GPflow VGP): base kernel variance fixed to 0.1, lengthscale 1, a rank-2
      coregionalisation matrix initialised at 0.1, Student-t likelihood with 3 degrees of freedom
      (scale initialised at 1) or Laplace likelihood (scale initialised at 0.1), trained with L-BFGS-B.
    """
    if method in ("MOGP", "R-MOGP"):
        init_cov = robust_covariance(Y_train, support_fraction=1 - eps)
        kernel = ICM(np.linalg.cholesky(init_cov), lengthscale=1.0)
        if method == "MOGP":
            model = MOGP(kernel, noise_var=0.1, mean=0.0, shared_noise=True)
            return model.fit(X_train, Y_train), dict(objective="loo")
        model = RMOGP(kernel, noise_var=0.1, mean=0.0, epsilons=[eps, 0.0], shared_noise=True)
        return model.fit(X_train, Y_train), dict(init_cov=init_cov, weight_power=2)
    gpflow_kwargs = dict(A=0.1 * np.ones((2, 2)), lengthscale=1.0, kernel_variance=0.1, mean=0.0)
    if method == "t-MOGP":
        model = TMOGP(2, df=3.0, scale=1.0, **gpflow_kwargs)
    else:
        model = LMOGP(2, scale=0.1, **gpflow_kwargs)
    return model.fit(X_train, Y_train), dict(method="lbfgs")


def run(method, X_train, Y_train, X_test, Y_test, eps, maxiter):
    """Optimise one model and evaluate it on the test set.

    The runtime is the wall-clock time of the hyperparameter optimisation (FastMCD and prediction
    excluded). Predictive variances include the observation noise.
    """
    model, opt_kwargs = build_model(method, X_train, Y_train, eps)
    start = time.perf_counter()
    res = model.optimize(maxiter=maxiter, tol=TOL[method], **opt_kwargs)
    runtime = time.perf_counter() - start
    mean, var = model.predict_y(X_test)
    return dict(rmse=rmse(Y_test, mean), nlpd=nlpd(Y_test, mean, var), runtime=runtime,
                nfev=res.nfev, time_per_eval=runtime / res.nfev)


def table_2(results, scenarios):
    """Test RMSE and NLPD, mean +- standard deviation over the splits."""
    def mean_std(values):
        return f"{np.mean(values):.2f} ± {np.std(values):.2f}"

    table = results.pivot_table(index="method", columns="scenario", values=["rmse", "nlpd"], aggfunc=mean_std)
    table = table.reindex(index=METHODS, columns=pd.MultiIndex.from_product([["rmse", "nlpd"], scenarios]))
    return table.rename(columns=SCENARIOS, level=1).rename(columns={"rmse": "RMSE", "nlpd": "NLPD"}, level=0)


def figure_3(results):
    """Average total runtime (log scale) against average test NLPD, one marker per method and scenario."""
    averages = results.groupby(["method", "scenario"])[["runtime", "nlpd"]].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in METHODS:
        avg = averages.loc[method]
        ax.scatter(avg["runtime"], avg["nlpd"], s=200, edgecolors="black", linewidths=1.5, zorder=3, **STYLE[method])
    ax.set_xscale("log")
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis="both", which="major", labelsize=20, width=2.5, length=7)
    ax.tick_params(axis="both", which="minor", width=2, length=4)
    ax.set_xlabel("Runtime (s)", fontsize=22)
    ax.set_ylabel("NLPD", fontsize=22)
    ax.legend(loc="upper right", fontsize=20, frameon=False, handletextpad=0.3)
    fig.tight_layout()
    return fig


def figure_4(results):
    """Mean +- standard deviation of the optimisation runtime per objective evaluation.

    ``rmogp`` computes gradients of the (weighted) LOO objective with TensorFlow autodiff, whereas the
    original implementation used finite differences, so the number of function evaluations and the
    runtimes differ from the paper. A function evaluation includes its gradient, as for the baselines.
    """
    per_eval = results.groupby("method")["time_per_eval"].agg(mean="mean", std=lambda v: v.std(ddof=0))
    per_eval = per_eval.reindex(METHODS)
    fig, ax = plt.subplots(figsize=(4, 7))
    ax.bar([STYLE[m]["label"] for m in METHODS], per_eval["mean"], yerr=per_eval["std"], capsize=10,
           color=[STYLE[m]["color"] for m in METHODS], edgecolor="black", alpha=0.9, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.tick_params(axis="y", labelsize=22, length=6)
    ax.tick_params(axis="x", labelsize=22, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_ylabel("Runtime per function evaluation (s)", fontsize=26, labelpad=15)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--eps", type=float, default=0.1, help="fraction of contaminated heating loads")
    parser.add_argument("--maxiter", type=int, default=None, help="optimiser iterations (default: until convergence)")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    X, Y = load_energy()
    records = []
    for scenario in args.scenarios:
        for split in tqdm(range(args.n_splits), desc=SCENARIOS[scenario]):
            data = make_split(X, Y, split, scenario, args.eps)
            for method in METHODS:
                records.append(dict(scenario=scenario, split=split, method=method,
                                    **run(method, *data, eps=args.eps, maxiter=args.maxiter)))
    results = pd.DataFrame(records)
    results.to_csv(args.results_dir / "energy_efficiency.csv", index=False)

    print(f"Table 2\n{table_2(results, args.scenarios).to_string()}")
    figure_3(results).savefig(args.fig_dir / "scatterplot_runtimes.pdf", bbox_inches="tight")
    figure_4(results).savefig(args.fig_dir / "barchart_runtime_feval.pdf", bbox_inches="tight")
    print(f"Figures saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
