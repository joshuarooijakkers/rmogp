"""Synthetic comparison of MOGP and R-MOGP (Table 1 and Table 3).

Data are drawn from a 3-output ICM on N = 100 inputs uniform on [-5, 5]. We compare the
standard MOGP with R-MOGP using the RCGP weight (centred on the prior mean, w_RCGP) and the
proposed R-MOGP weight (centred on the conditional mean given the other outputs, w_RMOGP),
in a well-specified setting (no outliers) and a contaminated setting (10% outliers in
output 1). All models are evaluated on the clean observations at the training inputs.

Table 3 compares hyperparameter objectives: leave-one-out cross-validation (LOO-CV) with
unweighted terms (power 0), terms weighted by the R-MOGP weights (power 1), and by the
squared weights (power 2). For MOGP, the LOO terms are weighted by the initial R-MOGP
weights and the "unweighted" entry is the marginal-likelihood MOGP. Table 1 is the subset
with the marginal-likelihood MOGP and both R-MOGPs with squared weights.

Usage:
    python comparison.py [--n-seeds 20] [--n 100]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rmogp import ICM, MOGP, RMOGP, compute_weights, robust_covariance
from rmogp.metrics import nlpd, rmse

ROOT = Path(__file__).resolve().parents[2]

B_TRUE = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.8], [0.7, 0.8, 1.0]])
T = B_TRUE.shape[0]
LENGTHSCALE = 1.0
NOISE_STD = 0.1
OUTLIER_FRACTION = 0.1
INIT_LENGTHSCALE = 1.0
INIT_NOISE_VAR = 0.1
LOO_TOL = 1e-2  # L-BFGS-B tolerance of the LOO-CV objectives

SETTINGS = {"well-specified": False, "contaminated": True}
METHODS = {"rcgp": "R-MOGP (w_RCGP)", "rmogp": "R-MOGP (w_RMOGP)"}
POWERS = {0: "unweighted", 1: "weighted", 2: "squared"}
METRICS = ["RMSE", "NLPD"]
ROW_ORDER = ["MOGP", *METHODS.values()]


def simulate(n, seed, contaminated):
    """Sample an ICM with a unit-variance squared exponential kernel, observed with Gaussian noise.

    In the contaminated setting, 10% of the observations of output 1 are shifted by U[2, 3] towards
    the opposite sign: upwards if y <= 0 (the prior mean) and downwards otherwise.
    """
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-5, 5, n))[:, None]
    k = np.exp(-0.5 * (X - X.T) ** 2 / LENGTHSCALE**2)
    L = np.linalg.cholesky(np.kron(B_TRUE, k) + 1e-6 * np.eye(T * n))
    F = (L @ rng.standard_normal(T * n)).reshape(T, n).T
    Y_clean = F + rng.normal(0, NOISE_STD, F.shape)
    Y = Y_clean.copy()
    if contaminated:
        rows = rng.choice(n, int(OUTLIER_FRACTION * n), replace=False)
        shift = rng.uniform(2, 3, len(rows))
        Y[rows, 0] += np.where(Y[rows, 0] > 0, -shift, shift)
    return X, Y, Y_clean


def evaluate(model, X, Y_clean):
    mean, var = model.predict_y(X)
    return {"RMSE": rmse(Y_clean, mean), "NLPD": nlpd(Y_clean, mean, var)}


def run(n, seed, contaminated):
    """Fit all models on one data set.

    The expected outlier proportions are (0.1, 0, 0) in the contaminated setting and 0 otherwise.
    A FastMCD covariance of the observed outputs (support fraction 1 - 2 max_t eps_t) gives both the
    initial mixing matrix of every model and the covariance for the initial R-MOGP weights. The prior
    mean is zero for all models.
    """
    X, Y, Y_clean = simulate(n, seed, contaminated)
    epsilons = np.array([OUTLIER_FRACTION, 0.0, 0.0]) if contaminated else np.zeros(T)
    init_cov = robust_covariance(Y, support_fraction=1 - 2 * epsilons.max())
    A0 = np.linalg.cholesky(init_cov)

    def mogp():
        return MOGP(ICM(A0, INIT_LENGTHSCALE), noise_var=INIT_NOISE_VAR, mean=0.0).fit(X, Y)

    results = []
    model = mogp()
    model.optimize()
    results.append(("MOGP", 0, evaluate(model, X, Y_clean)))

    loo_weights = compute_weights(Y, 0.0, epsilons, cov=init_cov)["w01"]
    for power in POWERS:
        if power > 0:
            model = mogp()
            model.optimize(objective="loo", loo_weights=loo_weights**power, tol=LOO_TOL)
            results.append(("MOGP", power, evaluate(model, X, Y_clean)))
        for weight, name in METHODS.items():
            model = RMOGP(ICM(A0, INIT_LENGTHSCALE), noise_var=INIT_NOISE_VAR, mean=0.0,
                          epsilons=epsilons, weight=weight).fit(X, Y)
            model.optimize(init_cov=init_cov, weight_power=power, tol=LOO_TOL)
            results.append((name, power, evaluate(model, X, Y_clean)))
    return results


def summarise(df, columns, column_order):
    """Mean +- standard deviation over seeds."""
    def mean_std(s):
        return f"{s.mean():.4f} ± {s.std(ddof=0):.4f}"

    table = df.groupby(["method", columns])[METRICS].agg(mean_std).unstack(columns)
    return table.reindex(index=ROW_ORDER, columns=pd.MultiIndex.from_product([METRICS, column_order]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=100, help="number of inputs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for setting, contaminated in SETTINGS.items():
        for seed in tqdm(range(args.n_seeds), desc=setting):
            for method, power, scores in run(args.n, seed, contaminated):
                records.append({"setting": setting, "seed": seed, "method": method,
                                "objective": POWERS[power], **scores})
    results = pd.DataFrame(records)
    results.to_csv(args.results_dir / "comparison.csv", index=False)

    is_table1 = np.where(results.method == "MOGP", results.objective == "unweighted", results.objective == "squared")
    print("Table 1\n" + summarise(results[is_table1], "setting", list(SETTINGS)).to_string())
    for setting in SETTINGS:
        table3 = summarise(results[results.setting == setting], "objective", list(POWERS.values()))
        print(f"\nTable 3 ({setting})\n{table3.to_string()}")


if __name__ == "__main__":
    main()
