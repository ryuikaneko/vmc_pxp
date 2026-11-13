#!/usr/bin/env python3
"""Kasteleyn/projectively translation-symmetric pBCS VMC on a periodic square lattice.

Hamiltonian
-----------
    H = sum_i Q_i (jx X_i + jn n_i + jp P_i) Q_i,
    Q_i = product_{j in nn(i)} P_j.

Ansatz
------
The real antisymmetric 2N x 2N pairing matrix is expanded in a basis built
from projective translations.  Translation in x is antiperiodic at the
single-orbital level,

    Ux**Lx = -I,

while Uy is periodic.  The fixed nonsingular sign kernel

    S = Ux - Ux.T

is invariant under the projective translation operators and has no zero mode
for even Lx.  Physical translations differ from Ux only by a site gauge sign.
The variational amplitudes are tied by displacement, so the parameter count
is O(N), rather than choose(2N,2).

The projected-product initializer is

    F0 = outer((1,-z),(1,-z)) kron S,

which gives amplitudes proportional to (-z)**N1 because Pf(D S D) =
det(D) Pf(S).  The script checks both the projection residual and the
nonsingularity of the all-zero selected matrix before sampling.
"""
from __future__ import annotations
import argparse
import csv
import math
import time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def neighbors(Lx, Ly):
    out = []
    for y in range(Ly):
        for x in range(Lx):
            values = {
                y * Lx + (x - 1) % Lx,
                y * Lx + (x + 1) % Lx,
                ((y - 1) % Ly) * Lx + x,
                ((y + 1) % Ly) * Lx + x,
            }
            out.append(tuple(sorted(values)))
    return out


def neighbor_matrix(nn):
    width = max(len(v) for v in nn)
    out = np.full((len(nn), width), -1, np.int64)
    for i, values in enumerate(nn):
        out[i, : len(values)] = values
    return out


def is_blockaded(state, nn):
    state = int(state)
    return all(
        not ((state >> i) & 1) or all(not ((state >> j) & 1) for j in nn[i])
        for i in range(len(nn))
    )


def can_flip(state, i, nn):
    state = int(state)
    return bool((state >> i) & 1) or all(not ((state >> j) & 1) for j in nn[i])


def count_particles(state):
    return bin(int(state)).count("1")


def count_empty_center(state, i, nn):
    state = int(state)
    return int(not ((state >> i) & 1) and all(not ((state >> j) & 1) for j in nn[i]))


def diagonal_energy(state, nn, jn, jp):
    return jn * count_particles(state) + jp * sum(
        count_empty_center(state, i, nn) for i in range(len(nn))
    )


def kasteleyn_sign_kernel(Lx, Ly, twist_x=-1, twist_y=-1):
    """Full-rank 2x1 Kasteleyn-oriented kernel on the square torus.

    Horizontal arrows point +x.  Vertical arrows carry (-1)^x.  The default
    antiperiodic twists in both winding directions select a nonsingular torus
    sector for the even-by-even lattices targeted here.
    """
    N = Lx * Ly
    S = np.zeros((N, N), float)

    def add_edge(i, j, sign):
        S[i, j] += sign
        S[j, i] -= sign

    for y in range(Ly):
        for x in range(Lx):
            i = x + Lx * y
            jx = (x + 1) % Lx + Lx * y
            sx = twist_x if x == Lx - 1 else 1
            add_edge(i, jx, sx)

            jy = x + Lx * ((y + 1) % Ly)
            sy = -1 if x & 1 else 1
            if y == Ly - 1:
                sy *= twist_y
            add_edge(i, jy, sy)
    return 0.5 * (S - S.T)


def kasteleyn_projective_translations(Lx, Ly, twist_x=-1, twist_y=-1):
    """Projective one-site translations preserving the Kasteleyn kernel.

    Tx is the physical +x shift followed by gauge (-1)^y.  Boundary gauge
    factors implement the torus twists.  Ty is the physical +y shift with its
    winding twist.  For the default (-1,-1) sector these obey
    Tx S Tx.T = S and Ty S Ty.T = S.
    """
    N = Lx * Ly
    Tx = np.zeros((N, N), float)
    Ty = np.zeros((N, N), float)
    for y in range(Ly):
        for x in range(Lx):
            src = x + Lx * y
            dst_x = (x + 1) % Lx + Lx * y
            gx = -1.0 if y & 1 else 1.0
            if x == Lx - 1:
                gx *= twist_x
            Tx[dst_x, src] = gx

            dst_y = x + Lx * ((y + 1) % Ly)
            gy = float(twist_y) if y == Ly - 1 else 1.0
            Ty[dst_y, src] = gy
    return Tx, Ty


def matrix_powers(U, n):
    powers = [np.eye(U.shape[0])]
    for _ in range(1, n):
        powers.append(powers[-1] @ U)
    return powers


def build_kasteleyn_translation_basis(Lx, Ly, twist_x=-1, twist_y=-1, tol=1.0e-12):
    """Orbit-average all displacement seeds under projective translations.

    Each retained B_p is real antisymmetric and satisfies projective one-site
    translation symmetry.  This realizes

        F_(r,a),(r',b) = S_(r,r') g_ab(r'-r)

    in a gauge-covariant form.  Linear duplicates are discarded by incremental
    orthogonalization, leaving O(N) parameters.
    """
    N = Lx * Ly
    Tx, Ty = kasteleyn_projective_translations(Lx, Ly, twist_x=twist_x, twist_y=twist_y)
    px = matrix_powers(Tx, Lx)
    py = matrix_powers(Ty, Ly)
    translations = [px[x] @ py[y] for y in range(Ly) for x in range(Lx)]

    basis = []
    labels = []
    orth = []
    for a in range(2):
        for b in range(2):
            for dy in range(Ly):
                for dx in range(Lx):
                    origin = 0
                    target = (dx % Lx) + Lx * (dy % Ly)
                    seed = np.zeros((2 * N, 2 * N), float)
                    ia = a * N + origin
                    jb = b * N + target
                    if ia == jb:
                        continue
                    seed[ia, jb] = 1.0
                    seed[jb, ia] = -1.0

                    B = np.zeros_like(seed)
                    for Tsite in translations:
                        Torb = np.zeros((2 * N, 2 * N), float)
                        Torb[:N, :N] = Tsite
                        Torb[N:, N:] = Tsite
                        B += Torb @ seed @ Torb.T
                    B /= len(translations)
                    B = 0.5 * (B - B.T)
                    norm = np.linalg.norm(B)
                    if norm < tol:
                        continue
                    B /= norm
                    v = B.reshape(-1).copy()
                    for q in orth:
                        v -= np.dot(q, v) * q
                    residual = np.linalg.norm(v)
                    if residual < tol:
                        continue
                    orth.append(v / residual)
                    basis.append(B)
                    labels.append(f"a={a},b={b},dx={dx},dy={dy}")
    return (np.ascontiguousarray(np.asarray(basis)), labels, Tx, Ty)


def exact_projected_product_pairing(Lx, Ly, z, twist_x=-1, twist_y=-1):
    N = Lx * Ly
    spinor = np.array([1.0, -float(z)])
    spinor /= np.linalg.norm(spinor)
    S = kasteleyn_sign_kernel(Lx, Ly, twist_x, twist_y)
    F = np.kron(np.outer(spinor, spinor), S)
    return 0.5 * (F - F.T)


def project_to_basis(target, basis):
    X = basis.reshape(len(basis), -1).T
    theta = np.linalg.lstsq(X, target.reshape(-1), rcond=None)[0]
    reconstructed = np.tensordot(theta, basis, axes=(0, 0))
    residual = float(np.max(np.abs(reconstructed - target)))
    return theta, residual


def projective_symmetry_residual(F, Lx, Ly, twist_x=-1, twist_y=-1):
    N = Lx * Ly
    Txsite, Tysite = kasteleyn_projective_translations(Lx, Ly, twist_x, twist_y)
    Tx = np.zeros((2 * N, 2 * N))
    Ty = np.zeros((2 * N, 2 * N))
    Tx[:N, :N] = Txsite
    Tx[N:, N:] = Txsite
    Ty[:N, :N] = Tysite
    Ty[N:, N:] = Tysite
    rx = np.max(np.abs(Tx @ F @ Tx.T - F))
    ry = np.max(np.abs(Ty @ F @ Ty.T - F))
    return float(rx), float(ry)


@njit(cache=True)
def can_flip_jit(state, site, nn):
    if (state >> site) & 1:
        return True
    for k in range(nn.shape[1]):
        j = nn[site, k]
        if j >= 0 and ((state >> j) & 1):
            return False
    return True


@njit(cache=True)
def count_particles_jit(state, N):
    total = 0
    for i in range(N):
        total += (state >> i) & 1
    return total


@njit(cache=True)
def diagonal_energy_jit(state, nn, jn, jp):
    N = nn.shape[0]
    particles = count_particles_jit(state, N)
    centers = 0
    for i in range(N):
        if (state >> i) & 1:
            continue
        clear = True
        for k in range(nn.shape[1]):
            j = nn[i, k]
            if j >= 0 and ((state >> j) & 1):
                clear = False
                break
        if clear:
            centers += 1
    return jn * particles + jp * centers


@njit(cache=True)
def make_F_jit(theta, basis):
    n = basis.shape[1]
    F = np.zeros((n, n), np.float64)
    for p in range(theta.size):
        for i in range(n):
            for j in range(n):
                F[i, j] += theta[p] * basis[p, i, j]
    return F


@njit(cache=True)
def selected_jit(F, state, N):
    q = np.empty(N, np.int64)
    A = np.empty((N, N), np.float64)
    for i in range(N):
        q[i] = N + i if ((state >> i) & 1) else i
    for i in range(N):
        for j in range(N):
            A[i, j] = F[q[i], q[j]]
    scale = np.max(np.abs(A))
    if (not np.isfinite(scale)) or scale == 0.0:
        raise ValueError("singular selected Pfaffian matrix")
    Ainv = np.linalg.inv(A / scale) / scale
    return q, A, Ainv


@njit(cache=True)
def replacement_row_jit(F, q, site, N, row):
    orb = site if q[site] >= N else N + site
    for j in range(N):
        row[j] = F[orb, q[j]]
    row[site] = 0.0


@njit(cache=True)
def inverse_replace_jit(A, Ainv, row, site, tol=1.0e-12):
    N = A.shape[0]
    d = np.empty(N, np.float64)
    u = np.empty(N, np.float64)
    x = np.empty(N, np.float64)
    ratio = 1.0
    for j in range(N):
        d[j] = row[j] - A[site, j]
        u[j] = Ainv[j, site]
        ratio += d[j] * u[j]
    if (not np.isfinite(ratio)) or abs(ratio) < tol:
        return False
    for i in range(N):
        value = 0.0
        for j in range(N):
            value -= Ainv[i, j] * d[j]
        x[i] = value
    inv_ratio = 1.0 / ratio
    for i in range(N):
        for j in range(N):
            Ainv[i, j] -= (u[i] * x[j] - x[i] * u[j]) * inv_ratio
    for i in range(N):
        Ainv[i, i] = 0.0
        for j in range(i + 1, N):
            value = 0.5 * (Ainv[i, j] - Ainv[j, i])
            Ainv[i, j] = value
            Ainv[j, i] = -value
    for j in range(N):
        A[site, j] = row[j]
        A[j, site] = -row[j]
    A[site, site] = 0.0
    return True


@njit(cache=True)
def log_derivatives_jit(q, Ainv, basis, out):
    out[:] = 0.0
    N = q.size
    for p in range(basis.shape[0]):
        value = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                value += Ainv[i, j] * basis[p, q[j], q[i]]
        out[p] = value


@njit(cache=True)
def sample_chain_jit(
    theta, basis, nn, jx, jn, jp, ns, burn, thin, state, seed, need_O, rebuild_interval
):
    np.random.seed(seed)
    N = nn.shape[0]
    F = make_F_jit(theta, basis)
    q, A, Ainv = selected_jit(F, state, N)
    E = np.empty(ns, np.float64)
    Mz = np.empty(ns, np.float64)
    Mx = np.empty(ns, np.float64)
    O = np.empty((ns, theta.size), np.float64) if need_O else np.empty((0, 0))
    row = np.empty(N, np.float64)
    owork = np.empty(theta.size, np.float64)
    accepted = 0
    proposed = 0
    measured = 0
    since_rebuild = 0
    for sweep in range(burn + ns * thin):
        for _ in range(N):
            site = np.random.randint(0, N)
            proposed += 1
            if not can_flip_jit(state, site, nn):
                continue
            replacement_row_jit(F, q, site, N, row)
            ratio = 0.0
            for j in range(N):
                ratio += row[j] * Ainv[j, site]
            ratio2 = ratio * ratio
            if np.isfinite(ratio2) and (ratio2 >= 1.0 or np.random.random() < ratio2):
                ok = inverse_replace_jit(A, Ainv, row, site)
                state ^= 1 << site
                accepted += 1
                since_rebuild += 1
                if ok:
                    q[site] = site if q[site] >= N else N + site
                if (not ok) or (
                    rebuild_interval > 0 and since_rebuild >= rebuild_interval
                ):
                    q, A, Ainv = selected_jit(F, state, N)
                    since_rebuild = 0
        if sweep >= burn and ((sweep - burn) % thin == 0):
            energy = diagonal_energy_jit(state, nn, jn, jp)
            mx = 0.0
            for site in range(N):
                if can_flip_jit(state, site, nn):
                    replacement_row_jit(F, q, site, N, row)
                    ratio = 0.0
                    for j in range(N):
                        ratio += row[j] * Ainv[j, site]
                    energy += jx * ratio
                    mx += ratio
            E[measured] = energy
            Mz[measured] = (N - 2.0 * count_particles_jit(state, N)) / N
            Mx[measured] = mx / N
            if need_O:
                log_derivatives_jit(q, Ainv, basis, owork)
                O[measured, :] = owork
            measured += 1
    return E, O, Mz, Mx, state, accepted / max(proposed, 1)


def sample(theta, basis, nn_matrix, args, ns, burn, state, rng, need_O):
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    E, O, Mz, Mx, state, acceptance = sample_chain_jit(
        np.ascontiguousarray(theta, dtype=np.float64),
        np.ascontiguousarray(basis, dtype=np.float64),
        np.ascontiguousarray(nn_matrix, dtype=np.int64),
        float(args.jx),
        float(args.jn),
        float(args.jp),
        int(ns),
        int(burn),
        int(args.thin),
        int(state),
        seed,
        bool(need_O),
        int(args.pf_rebuild_interval),
    )
    return E, (O if need_O else None), Mz, Mx, int(state), float(acceptance)


def exact_ed(Lx, Ly, jx, jn, jp):
    nn = neighbors(Lx, Ly)
    N = Lx * Ly
    states = np.asarray([s for s in range(1 << N) if is_blockaded(s, nn)], np.int64)
    index = {int(s): i for i, s in enumerate(states)}
    rows, cols, data = [], [], []
    for a, state0 in enumerate(states):
        state = int(state0)
        rows.append(a)
        cols.append(a)
        data.append(diagonal_energy(state, nn, jn, jp))
        for i in range(N):
            if can_flip(state, i, nn):
                rows.append(a)
                cols.append(index[state ^ (1 << i)])
                data.append(jx)
    H = csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
    value = eigsh(H, k=1, which="SA", tol=1.0e-12)[0][0]
    return float(value), len(states)


def binned_mean_error(values, block_size):
    nblock = len(values) // block_size
    if nblock < 2:
        return float(np.mean(values)), float("nan")
    blocks = values[: nblock * block_size].reshape(nblock, block_size).mean(axis=1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / math.sqrt(nblock))


def conditional_v_score(E, N, block_size, enabled):
    if not enabled:
        return float("nan"), float("nan")
    energy = float(E.mean())
    variance = float(np.mean((E - energy) ** 2))
    if energy == 0.0:
        return float("nan"), float("nan")
    value = N * variance / energy**2
    nblock = len(E) // block_size
    if nblock < 2:
        return float(value), float("nan")
    blocks = E[: nblock * block_size].reshape(nblock, block_size)
    eb = blocks.mean(axis=1)
    vb = np.mean((blocks - eb[:, None]) ** 2, axis=1)
    valid = eb != 0.0
    scores = N * vb[valid] / eb[valid] ** 2
    error = (
        scores.std(ddof=1) / math.sqrt(len(scores))
        if len(scores) >= 2
        else float("nan")
    )
    return float(value), float(error)


def tag(value):
    return (
        format(float(value), ".12g")
        .replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--Lx", type=int, default=4)
    p.add_argument("--Ly", type=int, default=4)
    p.add_argument("--jx", type=float, default=1.0)
    p.add_argument("--jn", type=float, default=0.0)
    p.add_argument("--jp", type=float, default=0.0)
    p.add_argument("--z", type=float, default=0.65)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--average-iterations", type=int, default=100)
    p.add_argument("--opt-samples", type=int, default=10000)
    p.add_argument("--final-samples", type=int, default=100000)
    p.add_argument("--opt-burn", type=int, default=500)
    p.add_argument("--final-burn", type=int, default=2000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--sr-diag", type=float, default=1.0e-3)
    p.add_argument("--max-step", type=float, default=0.3)
    p.add_argument("--pf-rebuild-interval", type=int, default=256)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ed-max-sites", type=int, default=16)
    p.add_argument("--output-root", default=".")
    p.add_argument("--output-prefix", default="dat")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    N = args.Lx * args.Ly
    if N % 2:
        p.error("Lx*Ly must be even")
    if args.Lx % 2:
        p.error("the antiperiodic-x projective sign kernel currently requires even Lx")
    if args.average_iterations < 1 or args.average_iterations > args.iterations:
        p.error("average-iterations must satisfy 1 <= average-iterations <= iterations")

    start = time.time()
    nn = neighbors(args.Lx, args.Ly)
    nn_matrix = neighbor_matrix(nn)
    basis, labels, Ux, Uy = build_kasteleyn_translation_basis(args.Lx, args.Ly)
    target = exact_projected_product_pairing(args.Lx, args.Ly, args.z)
    theta, residual = project_to_basis(target, basis)
    theta /= np.linalg.norm(theta)
    F = np.tensordot(theta, basis, axes=(0, 0))
    rx, ry = projective_symmetry_residual(F, args.Lx, args.Ly)
    S = kasteleyn_sign_kernel(args.Lx, args.Ly)
    singular_values = np.linalg.svd(S, compute_uv=False)
    sigma_min = float(singular_values[-1])
    if residual > 1.0e-10:
        raise RuntimeError(f"initializer projection residual is too large: {residual}")
    if sigma_min < 1.0e-12:
        raise RuntimeError(f"projective sign kernel is singular: sigma_min={sigma_min}")

    print(
        f"N={N}, Kasteleyn/projective parameters={len(theta)}, initializer residual={residual:.3e}"
    )
    print(
        f"Kasteleyn-kernel sigma_min={sigma_min:.3e}; symmetry residuals Tx={rx:.3e}, Ty={ry:.3e}"
    )

    ed_energy = float("nan")
    ed_dimension = float("nan")
    if N <= args.ed_max_sites:
        ed_energy, ed_dimension = exact_ed(args.Lx, args.Ly, args.jx, args.jn, args.jp)
        print(
            f"ED ground-state total energy={ed_energy:+.12f}, blockade dimension={ed_dimension}"
        )

    rng = np.random.default_rng(args.seed)
    state = 0
    history = []
    tail = []
    for iteration in range(args.iterations):
        E, O, _, _, state, acceptance = sample(
            theta,
            basis,
            nn_matrix,
            args,
            args.opt_samples,
            args.opt_burn,
            state,
            rng,
            True,
        )
        energy = float(E.mean())
        centered = O - O.mean(axis=0)
        metric = centered.T @ centered / len(E)
        gradient = 2.0 * np.mean(centered * (E - energy)[:, None], axis=0)
        regularizer = args.sr_diag * max(0.1, 0.95**iteration)
        metric += regularizer * np.diag(np.maximum(np.diag(metric), 1.0e-8))
        metric += 1.0e-10 * np.eye(len(theta))
        delta = np.linalg.solve(metric, -gradient)
        dnorm = np.linalg.norm(delta)
        if dnorm > args.max_step:
            delta *= args.max_step / dnorm
        theta += args.learning_rate * delta
        theta /= np.linalg.norm(theta)
        if iteration >= args.iterations - args.average_iterations:
            tail.append(theta.copy())
        history.append(
            (
                iteration,
                energy,
                float(np.var(E)),
                float(np.linalg.norm(gradient)),
                acceptance,
            )
        )
        if args.verbose and (iteration % 10 == 0 or iteration + 1 == args.iterations):
            print(
                f"SR {iteration:4d} E={energy:+.10f} var={np.var(E):.3e} "
                f"|g|={np.linalg.norm(gradient):.3e} acc={acceptance:.3f}"
            )

    theta = np.mean(np.asarray(tail), axis=0)
    theta /= np.linalg.norm(theta)
    F = np.tensordot(theta, basis, axes=(0, 0))
    rx, ry = projective_symmetry_residual(F, args.Lx, args.Ly)
    E, _, Mz, Mx, state, acceptance = sample(
        theta,
        basis,
        nn_matrix,
        args,
        args.final_samples,
        args.final_burn,
        state,
        rng,
        False,
    )
    total_energy, total_energy_error = binned_mean_error(E, args.block_size)
    mz, mz_error = binned_mean_error(Mz, args.block_size)
    mx, mx_error = binned_mean_error(Mx, args.block_size)
    variance = float(np.var(E))
    v_enabled = abs(args.jn) < 1.0e-14 and abs(args.jp) < 1.0e-14
    v_score, v_score_error = conditional_v_score(E, N, args.block_size, v_enabled)

    out = (
        Path(args.output_root)
        / f"Lx{args.Lx}_Ly{args.Ly}_jx{tag(args.jx)}_jn{tag(args.jn)}_jp{tag(args.jp)}"
    )
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / Path(args.output_prefix).name
    summary = dict(
        Lx=args.Lx,
        Ly=args.Ly,
        N=N,
        jx=args.jx,
        jn=args.jn,
        jp=args.jp,
        z=args.z,
        parameters=len(theta),
        averaged_iterations=len(tail),
        parameterization="kasteleyn_2x1_projective_translation_twist_mm",
        initializer_residual=residual,
        sign_kernel_sigma_min=sigma_min,
        projective_Tx_residual=rx,
        projective_Ty_residual=ry,
        infinite_temperature_energy=(0.0 if v_enabled else float("nan")),
        v_score_enabled=v_enabled,
        v_score=v_score,
        v_score_error=v_score_error,
        total_energy=total_energy,
        total_energy_error=total_energy_error,
        energy_density=total_energy / N,
        energy_density_error=total_energy_error / N,
        total_energy_variance=variance,
        energy_variance_density=variance / N,
        longitudinal_magnetization=mz,
        longitudinal_magnetization_error=mz_error,
        transverse_magnetization=mx,
        transverse_magnetization_error=mx_error,
        acceptance=acceptance,
        ed_total_energy=ed_energy,
        ed_dimension=ed_dimension,
        elapsed_seconds=time.time() - start,
    )
    with open(str(prefix) + ".csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    np.savez(
        str(prefix) + "_parameters.npz",
        theta=theta,
        F=F,
        basis=basis,
        labels=np.asarray(labels),
        Ux=Ux,
        Uy=Uy,
        sign_kernel=S,
        history=np.asarray(history, float),
    )
    np.savetxt(
        str(prefix) + "_sr_history.csv",
        np.asarray(history),
        delimiter=",",
        header="iteration,energy,variance,grad_norm,acceptance",
        comments="",
    )
    print(
        f"VMC total energy={total_energy:+.12f} +/- {total_energy_error:.3e}; "
        f"energy density={total_energy/N:+.12f}; acc={acceptance:.3f}"
    )
    if v_enabled:
        print(f"V-score={v_score:.12e} +/- {v_score_error:.3e} (E_inf=0)")
    else:
        print("V-score not computed: only jn=jp=0 is supported, where E_inf=0.")
    print(f"final projective symmetry residuals: Tx={rx:.3e}, Ty={ry:.3e}")
    print("saved under:", out)


if __name__ == "__main__":
    main()
