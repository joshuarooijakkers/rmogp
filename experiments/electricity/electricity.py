"""Electricity load: scalability of the sparse models (Figure 5).

We model the daily electricity load of T = 5 Nordic and Baltic countries (Denmark, Estonia,
Finland, Norway, Sweden) from Open Power System Data (Wiese et al., 2019), January 2010 to
September 2020. For growing training-set sizes N, 5% of the Danish load (output 1) in a
narrow time window is replaced by large outliers. We compare the sparse variational SVMOGP
and R-SVMOGP with sparse t- and Laplace-likelihood MOGPs (t-SVMOGP, L-SVMOGP) in total
runtime (hyperparameter optimisation + prediction) and test RMSE on output 1, against NT.

Data: two hourly OPSD releases are combined, downloaded to ``data/`` on first use (about
170 MB in total; https://data.open-power-system-data.org/time_series/):

* ``timeseries60min.csv`` (2016-07-14 release), covering 2010 to mid-2016;
* ``time_series_60min_singleindex.csv`` (2020-10-06 release), covering 2015 to September 2020.

Usage:
    python electricity.py [--n-seeds 10] [--train-sizes 600 1000 ... 3415] [--steps 2000]
"""

import argparse
import shutil
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rmogp import ICM, RSVMOGP, SVMOGP
from rmogp.baselines import LMOGP, TMOGP
from rmogp.metrics import rmse

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://data.open-power-system-data.org/time_series"
COUNTRIES = ["DK", "EE", "FI", "NO", "SE"]
SOURCES = {  # file name: (url, time column, load columns in the order of COUNTRIES)
    "timeseries60min.csv": (
        f"{BASE_URL}/2016-07-14/timeseries60min.csv",
        "Unnamed: 0",
        [f"load_{c}_load" for c in COUNTRIES],
    ),
    "time_series_60min_singleindex.csv": (
        f"{BASE_URL}/2020-10-06/time_series_60min_singleindex.csv",
        "utc_timestamp",
        [f"{c}_load_actual_entsoe_transparency" for c in COUNTRIES],
    ),
}

OUTLIER_MEAN = 9.0
OUTLIER_VAR = 0.2
OUTLIER_PERCENTILES = (45, 55)
INIT_A = [[1.0], [1.1], [0.95], [0.9], [1.05]]  # R = 1 latent process; order DK, EE, FI, NO, SE
INIT_LENGTHSCALE = 0.1
INIT_NOISE_VAR = 0.1
T_DF = 3.0
LIKELIHOOD_SCALE = 0.1
TEST_SIZE = 0.1
SPLIT_SEED = 42

NAMES = ["SVMOGP", "R-SVMOGP", "t-SVMOGP", "L-SVMOGP"]
STYLE = {
    "SVMOGP": dict(color="green", lw=5),
    "R-SVMOGP": dict(color="royalblue", lw=4),
    "t-SVMOGP": dict(color="orange", lw=6),
    "L-SVMOGP": dict(color="#800020", lw=4),
}
LABELS = {"t-SVMOGP": r"$t$-SVMOGP"}


def download(url, path):
    print(f"Downloading {url} ...")
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(path)


def load_electricity():
    """Standardised day index ``X`` (n, 1) and standardised daily loads ``Y`` (n, 5).

    Each series is averaged per day; both files are stacked, keeping one row per day where the releases
    overlap (2015 to mid-2016), and days with a missing value are dropped. The input is the day index
    counted before dropping days.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (url, time_col, columns) in SOURCES.items():
        path = DATA_DIR / name
        if not path.exists():
            download(url, path)
        df = pd.read_csv(path, usecols=[time_col, *columns], index_col=time_col, parse_dates=True)
        frames.append(df[columns].set_axis(COUNTRIES, axis=1).resample("D").mean())
    loads = pd.concat(frames).sort_index()
    loads = loads[~loads.index.duplicated(keep="first")]
    complete = loads.notna().all(axis=1).to_numpy()
    x = np.arange(len(loads))[complete].reshape(-1, 1).astype(float)
    y = loads[complete].to_numpy()
    return (x - x.mean()) / x.std(), (y - y.mean(axis=0)) / y.std(axis=0)


def contaminate(X, Y, epsilon, rng):
    """Replace a fraction ``epsilon`` of output 1 at inputs between the 45th and 55th percentile by draws
    from N(9, 0.2), about nine standard deviations above the mean."""
    Y = Y.copy()
    lo, hi = np.percentile(X, OUTLIER_PERCENTILES)
    eligible = np.flatnonzero((X[:, 0] >= lo) & (X[:, 0] <= hi))
    n = min(int(epsilon * len(Y)), len(eligible))
    idx = rng.choice(eligible, size=n, replace=False)
    Y[idx, 0] = rng.normal(OUTLIER_MEAN, np.sqrt(OUTLIER_VAR), size=n)
    return Y


def make_models(Z, T, epsilon):
    """All models start from the same hyperparameters; the Gaussian models share one noise variance
    across outputs, and the t and Laplace likelihoods use one scale for all outputs."""
    gaussian = dict(Z=Z, noise_var=INIT_NOISE_VAR, shared_noise=True)
    gpflow = dict(A=INIT_A, lengthscale=INIT_LENGTHSCALE, Z=Z)
    return dict(zip(NAMES, [
        SVMOGP(ICM(INIT_A, INIT_LENGTHSCALE), **gaussian),
        RSVMOGP(ICM(INIT_A, INIT_LENGTHSCALE), epsilons=epsilon, **gaussian),
        TMOGP(T, df=T_DF, scale=LIKELIHOOD_SCALE, **gpflow),
        LMOGP(T, scale=LIKELIHOOD_SCALE, **gpflow),
    ]))


def evaluate(model, X, Y, X_test, Y_test, steps, learning_rate):
    """Runtime of full-batch Adam and prediction (for the closed-form sparse models this includes
    refitting the variational posterior), and test RMSE on output 1."""
    model.fit(X, Y)
    start = time.perf_counter()
    model.optimize(method="adam", steps=steps, learning_rate=learning_rate)
    mean, _ = model.predict_f(X_test)
    runtime = time.perf_counter() - start
    return runtime, rmse(Y_test[:, 0], mean[:, 0])


def figure_5(results, NT):
    """Runtime (left) and test RMSE on output 1 (right) against NT; bands show +- one standard deviation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, ylabel in zip(axes, ["runtime", "rmse"], ["Runtime (s)", "RMSE"]):
        for name in NAMES:
            values = results[results.method == name].pivot(index="seed", columns="N", values=metric)
            mean, std = values.mean(0).to_numpy(), values.std(0, ddof=0).to_numpy()
            ax.plot(NT, mean, label=LABELS.get(name, name), color=STYLE[name]["color"], linewidth=STYLE[name]["lw"])
            ax.fill_between(NT, mean - std, mean + std, color=STYLE[name]["color"], alpha=0.2)
        ax.set_xlabel("$NT$", fontsize=28, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=28, labelpad=5)
        ax.tick_params(width=2.5, labelsize=26)
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
    axes[0].legend(fontsize=22, frameon=False)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[600, 1000, 1400, 1800, 2200, 2600, 3000, 3415])
    parser.add_argument("--steps", type=int, default=2000, help="Adam steps")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--epsilon", type=float, default=0.05, help="fraction of outliers in output 1")
    parser.add_argument("--fig-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    X, Y = load_electricity()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    T = Y.shape[1]
    print(f"{len(X)} days; train: {len(X_train)}, test: {len(X_test)}")

    # For every seed and training size: draw a random training subset, contaminate it, and fit all models
    # with Q = round(sqrt(N)) fixed inducing inputs on an even grid.
    records = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        for n in args.train_sizes:
            idx = rng.choice(len(X_train), size=n, replace=False)
            X_sub, Y_sub = X_train[idx], contaminate(X_train[idx], Y_train[idx], args.epsilon, rng)
            Z = np.linspace(X.min(), X.max(), round(np.sqrt(n))).reshape(-1, 1)
            for name, model in make_models(Z, T, args.epsilon).items():
                runtime, error = evaluate(model, X_sub, Y_sub, X_test, Y_test, args.steps, args.learning_rate)
                records.append(dict(seed=seed, N=n, method=name, runtime=runtime, rmse=error))
                print(f"seed {seed}, N = {n}, {name}: RMSE {error:.3f} ({runtime:.1f} s)")
    results = pd.DataFrame(records)
    results.to_csv(args.results_dir / "electricity.csv", index=False)

    def mean_std(values):
        return f"{np.mean(values):.3f} ± {np.std(values):.3f}"

    summary = results.pivot_table(index="N", columns="method", values=["runtime", "rmse"], aggfunc=mean_std)
    summary = summary.reindex(columns=pd.MultiIndex.from_product([["runtime", "rmse"], NAMES]))
    print(summary.rename(columns={"runtime": "Runtime (s)", "rmse": "RMSE"}, level=0).to_string())
    NT = T * np.array(sorted(args.train_sizes))
    figure_5(results, NT).savefig(args.fig_dir / "electricity.pdf", bbox_inches="tight")
    print(f"Figure saved to {args.fig_dir}")


if __name__ == "__main__":
    main()
