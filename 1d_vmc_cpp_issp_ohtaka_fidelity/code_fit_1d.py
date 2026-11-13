#!/usr/bin/env python3
"""Finite-size extrapolation of Lesanovsky/PXP VMC summary CSV files.

The script recursively searches for ``dat01.csv``, ``dat02.csv``, ...
and, within each directory, reads only the file with the largest numeric suffix.
It then groups equal (jx, jn, jp) and fits each numeric observable versus L.
Candidate corrections are ranked with AICc rather than selecting 1/L,
1/L^2, or exp(-L/xi) in advance:

  constant              y_inf
  inverse_L             y_inf + a/L
  inverse_L2            y_inf + a/L^2
  inverse_L_L2          y_inf + a/L + b/L^2
  exponential           y_inf + a exp(-L/xi)
  power_free            y_inf + a/L^p
  zero_inverse_L        a/L
  zero_inverse_L2       a/L^2
  zero_exponential      a exp(-L/xi)
  zero_power_free       a/L^p

Physical guidance:
* For a 1D gapped phase with periodic boundaries, local bulk observables
  normally approach the thermodynamic limit exponentially in L/xi.
* At a 1+1D conformal critical point with periodic boundaries, the ground-state
  energy density has a leading 1/L^2 correction. Other observables have powers
  fixed by scaling dimensions, which need not be 1 or 2.
* A 1/L term is useful for boundary/surface corrections and as a diagnostic,
  but periodic boundaries remove the ordinary open-boundary surface term.
* Variance and V-score vanish for an exact eigenstate. In finite-VMC data their
  residual size dependence can also reflect optimization/sampling quality, so
  both free-intercept and zero-intercept models are compared.

Outputs: per-coupling CSVs of model rankings and best extrapolations, plots,
and a Markdown report. Statistical uncertainties are used as weights when a
matching ``<observable>_error`` column exists and is finite/positive.
"""
from __future__ import annotations
import argparse
import csv
import math
import re
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ERROR_SUFFIX = "_error"
IDENTITY_COLUMNS = {
    "L",
    "jx",
    "jn",
    "jp",
    "z_initializer",
    "parameters",
    "averaged_iterations",
    "blockade_dimension_transfer",
    "final_samples_used",
    "final_blocks",
    "elapsed_seconds",
    "frustration_free",
}


def safe_tag(x: float) -> str:
    return (
        format(float(x), ".12g").replace("-", "m").replace("+", "p").replace(".", "p")
    )


def read_rows(root: Path, pattern: str) -> list[dict]:
    """Read the highest-numbered ``datNN.csv`` in each directory.

    For example, if one directory contains ``dat01.csv``, ``dat02.csv``,
    and ``dat03.csv``, only ``dat03.csv`` is read.  Selection is performed
    independently in every directory below ``root``.  The numeric suffix may
    contain two or more digits, so names beyond ``dat99.csv`` also work.
    """
    numbered_name = re.compile(r"^dat(\d{2,})\.csv$")
    latest_by_directory: dict[Path, tuple[int, Path]] = {}

    for file in root.glob(pattern):
        match = numbered_name.fullmatch(file.name)
        if match is None:
            continue
        number = int(match.group(1))
        current = latest_by_directory.get(file.parent)
        if current is None or number > current[0]:
            latest_by_directory[file.parent] = (number, file)

    selected_files = sorted(
        (item[1] for item in latest_by_directory.values()),
        key=lambda file: str(file),
    )
    if not selected_files:
        raise FileNotFoundError(
            f"no numbered CSV (datNN.csv) matched {pattern!r} below {root}"
        )

    rows = []
    for file in selected_files:
        with file.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(file)
                rows.append(row)
    return rows


def as_float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def models(include_zero: bool):
    out = {
        # Return an array, not a scalar.  scipy.curve_fit accepts a scalar
        # through broadcasting, but matplotlib requires x and y to have the
        # same first dimension when the fitted curve is plotted.
        "constant": (
            lambda L, c: np.full_like(np.asarray(L, dtype=float), c),
            ["y_inf"],
            None,
        ),
        "inverse_L": (lambda L, c, a: c + a / L, ["y_inf", "a"], None),
        "inverse_L2": (lambda L, c, a: c + a / L**2, ["y_inf", "a"], None),
        "inverse_L_L2": (
            lambda L, c, a, b: c + a / L + b / L**2,
            ["y_inf", "a", "b"],
            None,
        ),
        "exponential": (
            lambda L, c, a, xi: c + a * np.exp(-L / xi),
            ["y_inf", "a", "xi"],
            ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
        ),
        "power_free": (
            lambda L, c, a, power: c + a / L**power,
            ["y_inf", "a", "power"],
            ([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 8.0]),
        ),
    }
    if include_zero:
        out.update(
            {
                "zero_inverse_L": (lambda L, a: a / L, ["a"], None),
                "zero_inverse_L2": (lambda L, a: a / L**2, ["a"], None),
                "zero_exponential": (
                    lambda L, a, xi: a * np.exp(-L / xi),
                    ["a", "xi"],
                    ([-np.inf, 1e-6], [np.inf, np.inf]),
                ),
                "zero_power_free": (
                    lambda L, a, power: a / L**power,
                    ["a", "power"],
                    ([-np.inf, 0.05], [np.inf, 8.0]),
                ),
            }
        )
    return out


def initial_guess(name, y):
    span = float(y[0] - y[-1]) if len(y) else 1.0
    c = float(y[-1])
    if name == "constant":
        return [float(np.mean(y))]
    if name in ("inverse_L", "inverse_L2"):
        return [c, span]
    if name == "inverse_L_L2":
        return [c, span, span]
    if name == "exponential":
        return [c, span, 3.0]
    if name == "power_free":
        return [c, span, 1.0]
    if name in ("zero_inverse_L", "zero_inverse_L2"):
        return [float(y[0])]
    if name == "zero_exponential":
        return [float(y[0]), 3.0]
    if name == "zero_power_free":
        return [float(y[0]), 1.0]
    raise KeyError(name)


def fit_one(name, spec, L, y, sigma):
    fun, names, bounds = spec
    k = len(names)
    n = len(y)
    if n <= k:
        return None
    kwargs = dict(p0=initial_guess(name, y), maxfev=100000)
    if bounds is not None:
        kwargs["bounds"] = bounds
    if sigma is not None:
        kwargs.update(sigma=sigma, absolute_sigma=True)
    try:
        popt, pcov = curve_fit(fun, L, y, **kwargs)
    except Exception:
        return None
    pred = fun(L, *popt)
    residual = y - pred
    if sigma is None:
        rss = float(np.sum(residual**2))
        scale = max(rss / n, np.finfo(float).tiny)
        minus2loglike = n * math.log(scale)
        chi2 = float("nan")
    else:
        z = residual / sigma
        chi2 = float(np.sum(z * z))
        minus2loglike = chi2 + float(np.sum(np.log(2 * np.pi * sigma * sigma)))
        rss = float(np.sum(residual**2))
    aic = minus2loglike + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else float("inf")
    perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    params = dict(zip(names, map(float, popt)))
    errors = dict(zip((x + "_error" for x in names), map(float, perr)))
    return dict(
        model=name,
        n=n,
        k=k,
        rss=rss,
        chi2=chi2,
        aicc=aicc,
        params=params,
        errors=errors,
        function=fun,
    )


def observable_kind(name):
    lower = name.lower()
    return (
        "zero_expected"
        if ("variance" in lower or "v_score" in lower)
        else "finite_limit"
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument(
        "--pattern",
        default="**/dat*.csv",
        help=(
            "glob used to find numbered CSVs; only the largest datNN.csv "
            "suffix in each directory is read"
        ),
    )
    ap.add_argument("--output-dir", default="finite_size_extrapolation")
    ap.add_argument(
        "--observables",
        nargs="*",
        default=None,
        help="explicit columns; default is every numeric non-metadata column",
    )
    ap.add_argument("--min-L", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)
    rows = read_rows(root, args.pattern)
    groups = {}
    for row in rows:
        key = tuple(as_float(row.get(x)) for x in ("jx", "jn", "jp"))
        if None in key or as_float(row.get("L")) is None:
            continue
        groups.setdefault(key, []).append(row)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for (jx, jn, jp), group in sorted(groups.items()):
        group = sorted(group, key=lambda r: float(r["L"]))
        columns = sorted(set().union(*(r.keys() for r in group)))
        if args.observables is None:
            observables = []
            for col in columns:
                if (
                    col.startswith("_")
                    or col in IDENTITY_COLUMNS
                    or col.endswith(ERROR_SUFFIX)
                ):
                    continue
                vals = [as_float(r.get(col)) for r in group]
                if sum(v is not None for v in vals) >= 3:
                    observables.append(col)
        else:
            observables = args.observables
        tag = f"jx{safe_tag(jx)}_jn{safe_tag(jn)}_jp{safe_tag(jp)}"
        out = out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        rankings = []
        best_rows = []
        report = [f"# Finite-size extrapolation: jx={jx:g}, jn={jn:g}, jp={jp:g}", ""]
        for obs in observables:
            data = []
            err_col = obs + ERROR_SUFFIX
            for r in group:
                L = as_float(r.get("L"))
                y = as_float(r.get(obs))
                e = as_float(r.get(err_col))
                if L is not None and y is not None and L >= args.min_L:
                    data.append((L, y, e))
            if len(data) < 3:
                continue
            L = np.asarray([x[0] for x in data], float)
            y = np.asarray([x[1] for x in data], float)
            err_values = [x[2] for x in data]
            sigma = (
                np.asarray(err_values, float)
                if all(x is not None and x > 0 for x in err_values)
                else None
            )
            include_zero = observable_kind(obs) == "zero_expected"
            fits = []
            for name, spec in models(include_zero).items():
                fit = fit_one(name, spec, L, y, sigma)
                if fit is not None and np.isfinite(fit["aicc"]):
                    fits.append(fit)
            if not fits:
                continue
            fits.sort(key=lambda z: z["aicc"])
            amin = fits[0]["aicc"]
            weights = np.exp(-0.5 * np.asarray([f["aicc"] - amin for f in fits]))
            weights /= weights.sum()
            for fit, weight in zip(fits, weights):
                record = dict(
                    observable=obs,
                    model=fit["model"],
                    aicc=fit["aicc"],
                    delta_aicc=fit["aicc"] - amin,
                    akaike_weight=float(weight),
                    n=fit["n"],
                    k=fit["k"],
                    rss=fit["rss"],
                    chi2=fit["chi2"],
                )
                record.update(fit["params"])
                record.update(fit["errors"])
                rankings.append(record)
            best = fits[0]
            best_rows.append(
                dict(
                    observable=obs,
                    best_model=best["model"],
                    y_inf=best["params"].get("y_inf", 0.0),
                    y_inf_error=best["errors"].get("y_inf_error", 0.0),
                    aicc=best["aicc"],
                    n=len(L),
                    weighted=bool(sigma is not None),
                )
            )
            report += [
                f"## {obs}",
                "",
                f"Best model: `{best['model']}`; AICc={best['aicc']:.6g}; "
                f"thermodynamic limit={best['params'].get('y_inf', 0.0):.12g} "
                f"+/- {best['errors'].get('y_inf_error', 0.0):.3g}.",
                "",
            ]
            # Plot against 1/L while evaluating every fit in its native
            # variable L.  Include 1/L=0 so that the thermodynamic-limit
            # intercept is visible.  For exponential and power-law models we
            # avoid evaluating exactly at L=infinity and explicitly append
            # their known limiting value instead.
            inverse_L_data = 1.0 / L
            positive_inverse_L = np.linspace(
                1.0 / (1.15 * L.max()), 1.0 / (0.8 * L.min()), 399
            )
            L_grid = 1.0 / positive_inverse_L
            inverse_L_grid = np.concatenate(([0.0], positive_inverse_L))

            fig, ax = plt.subplots(figsize=(6.4, 4.5))
            if sigma is None:
                ax.plot(inverse_L_data, y, "o", label="data")
            else:
                ax.errorbar(
                    inverse_L_data, y, yerr=sigma, fmt="o", capsize=3, label="data"
                )
            for fit, weight in zip(fits[:3], weights[:3]):
                parameter_names = models(include_zero)[fit["model"]][1]
                pars = [fit["params"][x] for x in parameter_names]
                finite_curve = np.asarray(fit["function"](L_grid, *pars), dtype=float)
                if finite_curve.ndim == 0:
                    finite_curve = np.full_like(L_grid, float(finite_curve))
                else:
                    finite_curve = np.broadcast_to(finite_curve, L_grid.shape)
                thermodynamic_limit = fit["params"].get("y_inf", 0.0)
                fitted_curve = np.concatenate(([thermodynamic_limit], finite_curve))
                ax.plot(
                    inverse_L_grid,
                    fitted_curve,
                    label=f"{fit['model']} (w={weight:.2f})",
                )
            ax.axvline(0.0, color="0.75", linewidth=0.8)
            ax.set_xlim(left=0.0)
            ax.set_xlabel("1/L")
            ax.set_ylabel(obs)

            # Robust y limits: a poorly constrained exponential fit can have
            # a huge amplitude/correlation length outside the data range.  Do
            # not let such a curve make the measured data visually collapse.
            # The displayed window is determined primarily by the data and
            # their error bars, then enlarged modestly to include reasonable
            # fit predictions and thermodynamic-limit intercepts.  Extreme
            # fit values are clipped by the axes, while the numerical fit and
            # all CSV outputs remain unchanged.
            if sigma is None:
                data_low = y.copy()
                data_high = y.copy()
            else:
                data_low = y - sigma
                data_high = y + sigma
            finite_data = np.concatenate(
                (data_low[np.isfinite(data_low)], data_high[np.isfinite(data_high)])
            )
            if finite_data.size:
                data_min = float(np.min(finite_data))
                data_max = float(np.max(finite_data))
                data_span = data_max - data_min
                if not np.isfinite(data_span) or data_span <= 0.0:
                    data_span = max(abs(data_min), abs(data_max), 1.0) * 0.10

                # Candidate fit values are first restricted to a generous
                # data-centred guard band.  This permits meaningful curvature
                # without allowing an unstable exponential tail to dominate.
                guard_low = data_min - 5.0 * data_span
                guard_high = data_max + 5.0 * data_span
                visible_values = [finite_data]
                for line in ax.lines:
                    line_y = np.asarray(line.get_ydata(), dtype=float)
                    line_y = line_y[np.isfinite(line_y)]
                    line_y = line_y[(line_y >= guard_low) & (line_y <= guard_high)]
                    if line_y.size:
                        visible_values.append(line_y)
                visible = np.concatenate(visible_values)
                y_low = float(np.min(visible))
                y_high = float(np.max(visible))
                visible_span = y_high - y_low
                if not np.isfinite(visible_span) or visible_span <= 0.0:
                    visible_span = data_span
                padding = 0.08 * visible_span
                ax.set_ylim(y_low - padding, y_high + padding)

            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"{re.sub('[^A-Za-z0-9_.-]+','_',obs)}.png", dpi=180)
            plt.close(fig)
        if rankings:
            fields = sorted(set().union(*(r.keys() for r in rankings)))
            with (out / "model_rankings.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=fields)
                w.writeheader()
                w.writerows(rankings)
        if best_rows:
            fields = list(best_rows[0])
            with (out / "best_extrapolations.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=fields)
                w.writeheader()
                w.writerows(best_rows)
        (out / "report.md").write_text("\n".join(report))
        reports += report + [""]
    (out_root / "all_groups_report.md").write_text("\n".join(reports))
    print(f"read {len(rows)} rows in {len(groups)} coupling group(s)")
    print(f"saved extrapolations under: {out_root}")


if __name__ == "__main__":
    main()
