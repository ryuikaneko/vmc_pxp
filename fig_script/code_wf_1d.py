#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "axes.linewidth": 1.8,
        "lines.linewidth": 3.0,
        "lines.markersize": 14,
    }
)

LABEL_RE = re.compile(r"^s([01])s([01])\[d=(\d+)\]$")


def read_parameters(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(x for x in f if not x.startswith("#")))
    rows.sort(key=lambda r: int(r["index"]))
    if [int(r["index"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: indices must be consecutive from zero")
    return [r["label"] for r in rows], np.array([float(r["theta"]) for r in rows])


def infer_L(labels):
    matches = [LABEL_RE.fullmatch(x) for x in labels]
    if any(m is None for m in matches):
        raise ValueError("unrecognized parameter label")
    return 2 * max(int(m.group(3)) for m in matches)


def build_F(L, labels, theta):
    F = np.zeros((2 * L, 2 * L))
    for c, label in zip(theta, labels):
        s0, s1, target = map(int, LABEL_RE.fullmatch(label).groups())
        for a in range(2 * L):
            i, si = a % L, a // L
            for b in range(a + 1, 2 * L):
                j, sj = b % L, b // L
                d = min(abs(j - i), L - abs(j - i))
                if (si, sj, d) == (s0, s1, target):
                    v = c * (-1.0 if j < i else 1.0)
                    F[a, b] += v
                    F[b, a] -= v
    return 0.5 * (F - F.T)


def pfaffian(A, tol=1e-13):
    A = np.array(A, float, copy=True)
    n = A.shape[0]
    ans = 1.0
    for k in range(0, n - 1, 2):
        q = k + 1 + np.argmax(np.abs(A[k, k + 1 :]))
        if not np.isfinite(abs(A[k, q])) or abs(A[k, q]) <= tol:
            return 0.0
        if q != k + 1:
            A[[k + 1, q], :] = A[[q, k + 1], :]
            A[:, [k + 1, q]] = A[:, [q, k + 1]]
            ans = -ans
        pivot = A[k, k + 1]
        ans *= pivot
        for i in range(k + 2, n):
            for j in range(i + 1, n):
                A[i, j] += (A[k + 1, i] * A[k, j] - A[k, i] * A[k + 1, j]) / pivot
                A[j, i] = -A[i, j]
    return ans


def blockade_states(L):
    mask = (1 << L) - 1
    out = []
    for state in range(1 << L):
        shifted = ((state << 1) | (state >> (L - 1))) & mask
        if state & shifted == 0:
            out.append(state)
    return np.asarray(out, dtype=np.uint64)


def variational_wavefunction(F, states, L):
    sites = np.arange(L)
    psi = np.empty(len(states))
    for k, raw in enumerate(states):
        state = int(raw)
        occ = np.array([(state >> i) & 1 for i in range(L)], dtype=bool)
        q = np.where(occ, L + sites, sites)
        psi[k] = pfaffian(F[np.ix_(q, q)])
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid wavefunction norm")
    return psi / norm


def popcount_integer(state):
    # Python 3.7 and earlier compatible replacement for int.bit_count().
    state = int(state)
    count = 0
    while state:
        state &= state - 1
        count += 1
    return count


def count_ppp(state, L):
    return sum(
        ((state >> ((i - 1) % L)) & 1) == 0
        and ((state >> i) & 1) == 0
        and ((state >> ((i + 1) % L)) & 1) == 0
        for i in range(L)
    )


def can_flip(state, i, L):
    return bool(
        (state >> i) & 1
        or (((state >> ((i - 1) % L)) & 1) == 0 and ((state >> ((i + 1) % L)) & 1) == 0)
    )


def exact_ground_state(states, L, jx, jn, jp):
    index = {int(s): i for i, s in enumerate(states)}
    rows = []
    cols = []
    vals = []
    for a, raw in enumerate(states):
        state = int(raw)
        rows.append(a)
        cols.append(a)
        vals.append(jn * popcount_integer(state) + jp * count_ppp(state, L))
        for site in range(L):
            if can_flip(state, site, L):
                rows.append(index[state ^ (1 << site)])
                cols.append(a)
                vals.append(jx)
    H = coo_matrix((vals, (rows, cols)), shape=(len(states), len(states))).tocsr()
    e, v = eigsh(H, k=1, which="SA", tol=1e-12)
    psi = v[:, 0] / np.linalg.norm(v[:, 0])
    residual = np.linalg.norm(H @ psi - e[0] * psi)
    return float(e[0]), psi, float(residual)


def align(reference, vector):
    ov = float(reference @ vector)
    return (-vector if ov < 0 else vector), abs(ov)


def bits(state, L):
    return "".join("1" if (int(state) >> i) & 1 else "0" for i in range(L))


def parameter_curves(labels, theta):
    """Return pairing parameters grouped by spin channel and distance."""
    curves = {(0, 0): {}, (0, 1): {}, (1, 1): {}}
    for label, value in zip(labels, theta):
        match = LABEL_RE.fullmatch(label)
        if match is None:
            raise ValueError(f"unrecognized parameter label: {label}")
        s0, s1, d = map(int, match.groups())

        # Use f^{up,down} as the representative mixed-spin channel.
        # If a file uses s1s0 rather than s0s1, collect it in the same panel.
        channel = tuple(sorted((s0, s1)))
        if channel not in curves:
            continue
        if d in curves[channel]:
            raise ValueError(
                f"duplicate parameter for spin channel {channel} and distance d={d}"
            )
        curves[channel][d] = float(value)
    return curves


def plot_pairing_parameters(labels, theta_initial, theta_optimized, output):
    """Plot the three pairing channels as vertically joined panels."""
    initial = parameter_curves(labels, theta_initial)
    optimized = parameter_curves(labels, theta_optimized)

    channels = [
        ((0, 0), r"$f^{\!\uparrow\!\!\!,\!\!\!\uparrow}$"),
        ((0, 1), "Pairing parameters\n" + r"$f^{\!\uparrow\!\!\!,\!\!\!\downarrow}$"),
        ((1, 1), r"$f^{\!\downarrow\!\!\!,\!\!\!\downarrow}$"),
    ]
    panel_labels = ["(a)", "(b)", "(c)"]

    # The three panels share exactly the same horizontal axis.  The zero-distance
    # point is displayed in every panel even though the equal-spin channels do
    # not contain a parameter at d=0.
    fig, axes = plt.subplots(
        3,
        1,
#        figsize=(10, 10),
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )

    for ax, (channel, ylabel), panel_label in zip(
        axes, channels, panel_labels
    ):
        d0 = set(initial[channel])
        d1 = set(optimized[channel])
        if d0 != d1:
            missing_initial = sorted(d1 - d0)
            missing_optimized = sorted(d0 - d1)
            raise ValueError(
                f"distance sets differ for channel {channel}: "
                f"missing in initial={missing_initial}, "
                f"missing in optimized={missing_optimized}"
            )

        distances = np.array(sorted(d0), dtype=int)
        if distances.size == 0:
            ax.text(
                0.5,
                0.5,
                "no parameters",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            y_initial = np.array([initial[channel][d] for d in distances])
            y_optimized = np.array([optimized[channel][d] for d in distances])
            ax.plot(
                distances,
                y_initial,
                color="gray",
                marker="s",
                markerfacecolor="none",
                markeredgecolor="gray",
                markeredgewidth=2.5,
                linewidth=2.5,
                label="initial",
            )
            ax.plot(
                distances,
                y_optimized,
                color="red",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=4.0,
                linewidth=4.0,
                label="optimized",
            )
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)

            # Add equal vertical padding above and below the plotted values so that
            # each curve sits near the center of its panel instead of being pressed
            # against the upper or lower border.
            y_all = np.concatenate((y_initial, y_optimized))
            y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))
            y_span = y_max - y_min
            if y_span == 0.0:
                # Give an exactly flat curve a finite, scale-aware vertical range.
                y_span = max(abs(y_min), 1.0) * 0.20
            y_padding = 0.25 * y_span
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

#        ax.set_ylabel(ylabel)
        if channel == (0, 1):
            ax.set_ylabel(ylabel, linespacing=1.5)
        else:
            ax.set_ylabel(ylabel)

        ax.set_xlim(-0.25, 8.25)
        ax.set_xticks(np.arange(0, 9, dtype=int))
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            length=8,
            width=1.8,
            pad=8,
        )
        ax.grid(axis="x", alpha=0.25)
        ax.text(
            0.035,
            0.25,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=28,
        )

    # With hspace=0, suppress duplicated x tick labels at the joined borders.
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].set_xlabel(r"Distance $d$")

    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.35,
        handletextpad=0.5,
        labelspacing=0.35,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--initial",
        type=Path,
        default=Path(
            "../1d_vmc_cpp_issp_ohtaka_fidelity/dat/L16_jx1_jn0_jp0/dat01_initial_parameters_ideal.csv"
        ),
    )
    ap.add_argument(
        "--optimized",
        type=Path,
        default=Path(
            "../1d_vmc_cpp_issp_ohtaka_fidelity/dat/L16_jx1_jn0_jp0/dat02_parameters.csv"
        ),
    )
    default_output_dir = Path("wavefunc_plots_1d")
    ap.add_argument(
        "--output",
        type=Path,
        #        default=default_output_dir / "wavefunction_initial_optimized_exact.png",
        default=default_output_dir / "wavefunction_initial_optimized_exact.pdf",
    )
    ap.add_argument(
        "--csv-output",
        type=Path,
        default=default_output_dir / "wavefunction_top_components.csv",
    )
    ap.add_argument(
        "--f-output",
        type=Path,
        default=default_output_dir / "pairing_parameters_initial_optimized.pdf",
        help=(
            "output figure comparing initial and optimized "
            "f^{up,up}, f^{up,down}, and f^{down,down}"
        ),
    )
    ap.add_argument("--top", type=int, default=65536)
    ap.add_argument("--jx", type=float, default=1.0)
    ap.add_argument("--jn", type=float, default=0.0)
    ap.add_argument("--jp", type=float, default=0.0)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.csv_output.parent.mkdir(parents=True, exist_ok=True)
    a.f_output.parent.mkdir(parents=True, exist_ok=True)
    labels0, t0 = read_parameters(a.initial)
    labels1, t1 = read_parameters(a.optimized)
    if labels0 != labels1:
        raise ValueError("parameter labels/order differ")

    # Plot the pairing parameters themselves, independently of spin-basis amplitudes.
    plot_pairing_parameters(labels0, t0, t1, a.f_output)

    L = infer_L(labels0)
    states = blockade_states(L)
    p0 = variational_wavefunction(build_F(L, labels0, t0), states, L)
    p1 = variational_wavefunction(build_F(L, labels1, t1), states, L)
    energy, pe, residual = exact_ground_state(states, L, a.jx, a.jn, a.jp)
    p0, o0 = align(pe, p0)
    p1, o1 = align(pe, p1)
    o01 = abs(float(p0 @ p1))
    probs = np.vstack((p0 * p0, p1 * p1, pe * pe))
    top = min(max(1, a.top), len(states))
    # Sort configurations by the exact-state probability.
    # order = np.argsort(probs.max(axis=0))[::-1][:top]
    order = np.argsort(probs[2])[::-1][:top]
    rank = np.arange(1, top + 1)
    with a.csv_output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "state_integer",
                "configuration_site0_to_Lminus1",
                "psi_initial",
                "psi_optimized",
                "psi_exact",
                "prob_initial",
                "prob_optimized",
                "prob_exact",
            ]
        )
        for r, i in enumerate(order, 1):
            w.writerow(
                [
                    r,
                    int(states[i]),
                    bits(states[i], L),
                    p0[i],
                    p1[i],
                    pe[i],
                    probs[0, i],
                    probs[1, i],
                    probs[2, i],
                ]
            )
    # Keep only the former upper-right panel: probability versus exact-state rank.
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
#    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
#    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
#    fig, ax = plt.subplots(figsize=(9, 6.75), constrained_layout=True)
#    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    #    colors = ["tab:blue", "tab:orange", "black"]
    colors = ["gray", "red", "black"]
    names = ["initial", "optimized", "exact"]
    linewidths = [2.5, 4.5, 1.5]  # initial, optimized, exact
    floor = np.finfo(float).tiny
    for y, c, n, lw in zip(probs, colors, names, linewidths):
        ax.semilogy(
            rank,
            np.maximum(y[order], floor),
            color=c,
            label=n,
            linewidth=lw,
        )

    ax.set(
        xlabel="Rank by exact-state probability",
#        ylabel=r"Probability $|\psi(\sigma)|^2$",
        ylabel=r"Probability $|\Psi(x)|^2$",
    )
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=8,
        width=1.8,
        pad=8,
    )
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.35,
        handletextpad=0.5,
        labelspacing=0.35,
    )

    fig.savefig(a.output, bbox_inches="tight")
    plt.close(fig)
    print(f"ed_energy={energy:.17g}")
    print(f"ed_residual={residual:.17g}")
    print(f"fidelity_initial_exact={o0**2:.17g}")
    print(f"fidelity_optimized_exact={o1**2:.17g}")
    print(f"fidelity_initial_optimized={o01**2:.17g}")
    print(f"saved_plot={a.output}")
    print(f"saved_pairing_plot={a.f_output}")
    print(f"saved_components={a.csv_output}")


if __name__ == "__main__":
    main()
