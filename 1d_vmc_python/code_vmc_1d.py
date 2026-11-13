#!/usr/bin/env python3
"""Blockade-projected pBCS VMC for a generalized Lesanovsky chain.

Hamiltonian (periodic boundary conditions)

  H = sum_i P_{i-1} [ jx X_i + jn n_i + jp P_i ] P_{i+1}.

The standard Lesanovsky frustration-free line is

  jn = jx / z,   jp = jx * z,   z > 0,

with the exact zero-energy ground state

  |z> proportional to P_B product_i (|0>_i - z |1>_i).

The pBCS initializer is the Gutzwiller-projected representation of this exact
state.  In site-spin orbital order (up orbitals first, then down orbitals),

  F_ij^{ab} = s_ij v_a v_b,  s_ij = sign(j-i),  v = (1,-z)/sqrt(1+z^2).

The program reports the exact-sum variational energy before and after SR,
performs Monte Carlo stochastic reconfiguration, and compares against
constrained exact diagonalization when the system size permits.
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


@njit(cache=True)
def pfaffian(A, tol=1.0e-13):
    A = A.copy()
    n = A.shape[0]
    if n & 1:
        return 0.0
    ans = 1.0
    for k in range(0, n - 1, 2):
        p = k + 1
        vmax = abs(A[k, p])
        for j in range(k + 2, n):
            q = abs(A[k, j])
            if q > vmax:
                vmax, p = q, j
        if vmax < tol:
            return 0.0
        if p != k + 1:
            for q in range(n):
                tmp = A[k + 1, q]
                A[k + 1, q] = A[p, q]
                A[p, q] = tmp
            for q in range(n):
                tmp = A[q, k + 1]
                A[q, k + 1] = A[q, p]
                A[q, p] = tmp
            ans = -ans
        pivot = A[k, k + 1]
        ans *= pivot
        for i in range(k + 2, n):
            for j in range(i + 1, n):
                A[i, j] += (A[k + 1, i] * A[k, j] - A[k, i] * A[k + 1, j]) / pivot
                A[j, i] = -A[i, j]
    return ans


@njit(cache=True)
def is_blockaded(state, L):
    for i in range(L):
        if ((state >> i) & 1) and ((state >> ((i + 1) % L)) & 1):
            return False
    return True


@njit(cache=True)
def can_flip(state, i, L):
    if (state >> i) & 1:
        return True
    return ((state >> ((i - 1) % L)) & 1) == 0 and ((state >> ((i + 1) % L)) & 1) == 0


@njit(cache=True)
def count_particles(state, L):
    total = 0
    for i in range(L):
        total += (state >> i) & 1
    return total


@njit(cache=True)
def count_ppp_centers(state, L):
    total = 0
    for i in range(L):
        if (
            ((state >> ((i - 1) % L)) & 1) == 0
            and ((state >> i) & 1) == 0
            and ((state >> ((i + 1) % L)) & 1) == 0
        ):
            total += 1
    return total


@njit(cache=True)
def diagonal_energy(state, L, jn, jp):
    return jn * count_particles(state, L) + jp * count_ppp_centers(state, L)


@njit(cache=True)
def fill_site_orbitals(state, L, q):
    for i in range(L):
        q[i] = L + i if ((state >> i) & 1) else i


@njit(cache=True)
def build_selected_matrix(F, state, L, q, A):
    fill_site_orbitals(state, L, q)
    for i in range(L):
        for j in range(L):
            A[i, j] = F[q[i], q[j]]


@njit(cache=True)
def replacement_row(F, q, site, row):
    L = q.size
    new_orb = site if q[site] >= L else L + site
    for j in range(L):
        row[j] = F[new_orb, q[j]]
    row[site] = 0.0


@njit(cache=True)
def pfaffian_replace_ratio(Ainv, row, site):
    ratio = 0.0
    for j in range(Ainv.shape[0]):
        ratio += row[j] * Ainv[j, site]
    return ratio


@njit(cache=True)
def inverse_replace_inplace(A, Ainv, row, site, d, x, u, tol=1.0e-12):
    n = A.shape[0]
    ratio = 1.0
    for j in range(n):
        d[j] = row[j] - A[site, j]
        ratio += d[j] * Ainv[j, site]
    if (not np.isfinite(ratio)) or abs(ratio) < tol:
        return False
    for i in range(n):
        u[i] = Ainv[i, site]
        sx = 0.0
        for j in range(n):
            sx -= Ainv[i, j] * d[j]
        x[i] = sx
    invr = 1.0 / ratio
    for i in range(n):
        for j in range(n):
            Ainv[i, j] -= (u[i] * x[j] - x[i] * u[j]) * invr
    for i in range(n):
        Ainv[i, i] = 0.0
        for j in range(i + 1, n):
            value = 0.5 * (Ainv[i, j] - Ainv[j, i])
            Ainv[i, j] = value
            Ainv[j, i] = -value
    for j in range(n):
        A[site, j] = row[j]
        A[j, site] = -row[j]
    A[site, site] = 0.0
    return True


@njit(cache=True)
def log_derivatives(parameter_index, parameter_value, q, Ainv, out):
    for p in range(out.size):
        out[p] = 0.0
    L = q.size
    for i in range(L):
        qi = q[i]
        for j in range(i + 1, L):
            p = parameter_index[q[j], qi]
            if p >= 0:
                out[p] += Ainv[i, j] * parameter_value[q[j], qi]


@njit(cache=True)
def scaled_skew_inverse(A, pivot_tol=1.0e-13):
    """Inverse of a nonsingular skew matrix without an absolute-Pfaffian test.

    A global scale of a Pfaffian matrix is physically irrelevant but makes
    Pf(A) scale as scale**(L/2).  At large L an entirely regular matrix can
    therefore have an extremely small Pfaffian.  We normalize by max|A_ij|
    before inversion and undo that normalization in the inverse.
    """
    scale = 0.0
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            value = abs(A[i, j])
            if value > scale:
                scale = value
    if (not np.isfinite(scale)) or scale == 0.0:
        raise ValueError("selected Pfaffian matrix has zero or invalid scale")
    B = A / scale
    # Relative pivot test.  This detects singularity independently of the
    # physically irrelevant overall normalization of F.
    work = B.copy()
    for k in range(0, n - 1, 2):
        vmax = 0.0
        for j in range(k + 1, n):
            value = abs(work[k, j])
            if value > vmax:
                vmax = value
        if (not np.isfinite(vmax)) or vmax < pivot_tol:
            raise ValueError("selected Pfaffian matrix is singular")
        p = k + 1
        best = abs(work[k, p])
        for j in range(k + 2, n):
            value = abs(work[k, j])
            if value > best:
                best = value
                p = j
        if p != k + 1:
            for j in range(n):
                tmp = work[k + 1, j]
                work[k + 1, j] = work[p, j]
                work[p, j] = tmp
            for i in range(n):
                tmp = work[i, k + 1]
                work[i, k + 1] = work[i, p]
                work[i, p] = tmp
        pivot = work[k, k + 1]
        for i in range(k + 2, n):
            for j in range(i + 1, n):
                work[i, j] += (
                    work[k + 1, i] * work[k, j] - work[k, i] * work[k + 1, j]
                ) / pivot
                work[j, i] = -work[i, j]
    return np.linalg.inv(B) / scale


@njit(cache=True)
def sample_chain(
    F,
    parameter_index,
    parameter_value,
    nparam,
    L,
    jx,
    jn,
    jp,
    nsamples,
    burn,
    thin,
    initial_state,
    seed,
    need_O,
    rebuild_interval,
):
    np.random.seed(seed)
    E = np.empty(nsamples, np.float64)
    Mz = np.empty(nsamples, np.float64)
    Mx = np.empty(nsamples, np.float64)
    O = np.empty((nsamples, nparam), np.float64) if need_O else np.empty((0, 0))
    q = np.empty(L, np.int64)
    A = np.empty((L, L), np.float64)
    row = np.empty(L, np.float64)
    d = np.empty(L, np.float64)
    x = np.empty(L, np.float64)
    u = np.empty(L, np.float64)
    owork = np.empty(nparam, np.float64)
    state = initial_state if is_blockaded(initial_state, L) else 0
    build_selected_matrix(F, state, L, q, A)
    # Do not use |Pf(A)| as a nonsingularity test.  Pf(c A)=c**(L/2) Pf(A),
    # so at L=48 a harmless global normalization makes |Pf(A)| tiny.
    # The exact Lesanovsky initializer has a nonsingular all-zero sector.
    Ainv = scaled_skew_inverse(A)
    accepted = 0
    proposed = 0
    measured = 0
    since_rebuild = 0
    total = burn + nsamples * thin
    for sweep in range(total):
        for _ in range(L):
            i = np.random.randint(0, L)
            proposed += 1
            if not can_flip(state, i, L):
                continue
            replacement_row(F, q, i, row)
            ratio = pfaffian_replace_ratio(Ainv, row, i)
            ratio2 = ratio * ratio
            if np.isfinite(ratio2) and (ratio2 >= 1.0 or np.random.random() < ratio2):
                ok = inverse_replace_inplace(A, Ainv, row, i, d, x, u)
                state ^= 1 << i
                if ok:
                    q[i] = i if q[i] >= L else L + i
                accepted += 1
                since_rebuild += 1
                if (not ok) or (
                    rebuild_interval > 0 and since_rebuild >= rebuild_interval
                ):
                    build_selected_matrix(F, state, L, q, A)
                    Ainv = scaled_skew_inverse(A)
                    since_rebuild = 0
        if sweep == burn - 1 and burn > 0:
            build_selected_matrix(F, state, L, q, A)
            Ainv = scaled_skew_inverse(A)
            since_rebuild = 0
        if sweep >= burn and ((sweep - burn) % thin == 0):
            eloc = diagonal_energy(state, L, jn, jp)
            mx_local = 0.0
            for i in range(L):
                if can_flip(state, i, L):
                    replacement_row(F, q, i, row)
                    flip_ratio = pfaffian_replace_ratio(Ainv, row, i)
                    eloc += jx * flip_ratio
                    mx_local += flip_ratio
            E[measured] = eloc
            # Z_i|0> = +|0>, Z_i|1> = -|1>.
            Mz[measured] = (L - 2.0 * count_particles(state, L)) / L
            # Blockade-projected transverse magnetization sum_i P X_i P / L.
            Mx[measured] = mx_local / L
            if need_O:
                log_derivatives(parameter_index, parameter_value, q, Ainv, owork)
                for p in range(nparam):
                    O[measured, p] = owork[p]
            measured += 1
    return E, O, Mz, Mx, state, accepted / max(1, proposed)


def sign_matrix(L):
    i = np.arange(L)
    return np.sign(i[None, :] - i[:, None]).astype(float)


def exact_lesanovsky_pairing(L, z):
    """Mixed-spin pBCS matrix for P_B product_i (|0>-z|1>)."""
    spinor = np.array([1.0, -float(z)])
    spinor /= np.linalg.norm(spinor)
    S = sign_matrix(L)
    F = np.kron(np.outer(spinor, spinor), S)
    return 0.5 * (F - F.T)


def build_parameter_basis(L):
    """Real translation/reflection-tied distance classes, as in the source code."""
    pairs = []
    keys = set()
    for a in range(2 * L):
        i, si = a % L, a // L
        for b in range(a + 1, 2 * L):
            j, sj = b % L, b // L
            d0 = abs(j - i)
            key = (si, sj, min(d0, L - d0))
            sign = -1.0 if j < i else 1.0
            pairs.append((a, b, key, sign))
            keys.add(key)
    mats = []
    labels = []
    for key in sorted(keys):
        B = np.zeros((2 * L, 2 * L), float)
        for a, b, pair_key, sign in pairs:
            if pair_key == key:
                B[a, b] = sign
                B[b, a] = -sign
        mats.append(B)
        labels.append(f"s{key[0]}s{key[1]}[d={key[2]}]")
    return np.asarray(mats), labels


def build_parameter_lookup(basis):
    nparam, n, _ = basis.shape
    parameter_index = np.full((n, n), -1, np.int32)
    parameter_value = np.zeros((n, n), float)
    for p in range(nparam):
        rows, cols = np.nonzero(basis[p])
        for a, b in zip(rows, cols):
            if parameter_index[a, b] >= 0:
                raise ValueError("overlapping parameter basis")
            parameter_index[a, b] = p
            parameter_value[a, b] = basis[p, a, b]
    return parameter_index, parameter_value


def project_to_basis(target, basis):
    X = basis.reshape(len(basis), -1).T
    theta = np.linalg.lstsq(X, target.reshape(-1), rcond=None)[0]
    reconstructed = np.tensordot(theta, basis, axes=(0, 0))
    residual = np.max(np.abs(reconstructed - target))
    return theta, float(residual)


class BlockadePBCS:
    def __init__(self, L, theta, basis):
        self.L = L
        self.theta = theta.copy()
        self.basis = basis
        self.parameter_index, self.parameter_value = build_parameter_lookup(basis)
        self.refresh()

    def refresh(self):
        self.F = np.ascontiguousarray(np.tensordot(self.theta, self.basis, axes=(0, 0)))
        self.F = np.ascontiguousarray(0.5 * (self.F - self.F.T))

    def amplitude(self, state):
        if not is_blockaded(int(state), self.L):
            return 0.0
        q = np.empty(self.L, np.int64)
        A = np.empty((self.L, self.L))
        build_selected_matrix(self.F, int(state), self.L, q, A)
        return float(pfaffian(A))


def blockade_basis(L):
    return np.asarray(
        [state for state in range(1 << L) if is_blockaded(state, L)], np.int64
    )


def build_hamiltonian(L, jx, jn, jp):
    states = blockade_basis(L)
    index = {int(state): i for i, state in enumerate(states)}
    rows = []
    cols = []
    data = []
    for a, state0 in enumerate(states):
        state = int(state0)
        rows.append(a)
        cols.append(a)
        data.append(diagonal_energy(state, L, jn, jp))
        for i in range(L):
            if can_flip(state, i, L):
                rows.append(a)
                cols.append(index[state ^ (1 << i)])
                data.append(jx)
    H = csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))
    return H, states


def exact_ground_state(L, jx, jn, jp):
    H, states = build_hamiltonian(L, jx, jn, jp)
    if H.shape[0] <= 3:
        vals, vecs = np.linalg.eigh(H.toarray())
        return float(vals[0]), vecs[:, 0], states
    value, vector = eigsh(H, k=1, which="SA", tol=1.0e-12, maxiter=10000)
    return float(value[0]), vector[:, 0], states


def infinite_temperature_observables_transfer(L, jn, jp):
    """Exact infinite-temperature traces in the periodic blockade space.

    Pair states are (s_{i-1}, s_i) in {00, 01, 10}.  The 3x3 transfer
    matrix advances (a,b) -> (b,c), rejects b=c=1, attaches x**c, and
    attaches y for (a,b,c)=(0,0,0).  Therefore D=Tr M**L, while the x
    and y derivatives give the sums of N1 and N000.  All three matrix
    products are propagated together in O(L) time and O(1) memory.
    """
    if L < 2:
        raise ValueError("L must be at least 2")
    if L == 2:
        dimension = 3.0
        mean_n1 = 2.0 / dimension
        mean_n000 = 2.0 / dimension
        energy = jn * mean_n1 + jp * mean_n000
        return float(energy), int(dimension), float(mean_n1), float(mean_n000)

    pair_states = ((0, 0), (0, 1), (1, 0))
    pair_index = {pair: i for i, pair in enumerate(pair_states)}
    M = np.zeros((3, 3), dtype=float)
    dM_dx = np.zeros_like(M)
    dM_dy = np.zeros_like(M)
    for row, (a, b) in enumerate(pair_states):
        for c in (0, 1):
            if b == 1 and c == 1:
                continue
            col = pair_index[(b, c)]
            M[row, col] = 1.0
            if c == 1:
                dM_dx[row, col] = 1.0
            if a == 0 and b == 0 and c == 0:
                dM_dy[row, col] = 1.0

    power = np.eye(3)
    derivative_x = np.zeros((3, 3), dtype=float)
    derivative_y = np.zeros((3, 3), dtype=float)
    for _ in range(L):
        derivative_x = derivative_x @ M + power @ dM_dx
        derivative_y = derivative_y @ M + power @ dM_dy
        power = power @ M

    dimension_float = float(np.trace(power))
    sum_n1 = float(np.trace(derivative_x))
    sum_n000 = float(np.trace(derivative_y))
    dimension = int(round(dimension_float))
    mean_n1 = sum_n1 / dimension_float
    mean_n000 = sum_n000 / dimension_float
    energy = jn * mean_n1 + jp * mean_n000
    return float(energy), dimension, float(mean_n1), float(mean_n000)


def exact_variational_observables(wf, jx, jn, jp):
    H, states = build_hamiltonian(wf.L, jx, jn, jp)
    amplitudes = np.asarray([wf.amplitude(int(state)) for state in states])
    norm = float(amplitudes @ amplitudes)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("zero or invalid variational norm")
    psi = amplitudes / math.sqrt(norm)
    Hpsi = H @ psi
    energy = float(psi @ Hpsi)
    variance = float(Hpsi @ Hpsi - energy * energy)
    return energy, max(variance, 0.0), psi, states


def exact_lesanovsky_vector(states, z, L):
    amplitudes = np.asarray(
        [(-z) ** count_particles(int(state), L) for state in states], dtype=float
    )
    return amplitudes / np.linalg.norm(amplitudes)


def sample_observables(wf, args, rng, state, final=False):
    nsamples = args.final_samples if final else args.opt_samples
    burn = args.final_burn if final else args.opt_burn
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    E, O, Mz, Mx, state, acceptance = sample_chain(
        wf.F,
        wf.parameter_index,
        wf.parameter_value,
        len(wf.theta),
        wf.L,
        args.jx,
        args.jn,
        args.jp,
        nsamples,
        burn,
        args.thin,
        state,
        seed,
        not final,
        args.pf_rebuild_interval,
    )
    return E, (None if final else O), Mz, Mx, int(state), float(acceptance)


def sr_optimize(wf, args, rng, state):
    history = []
    theta_tail = []
    average_start = max(0, args.iterations - args.average_iterations)
    for iteration in range(args.iterations):
        E, O, _, _, state, acceptance = sample_observables(wf, args, rng, state)
        energy = float(E.mean())
        centered_O = O - O.mean(axis=0)
        S = (centered_O.T @ centered_O) / len(E)
        gradient = 2.0 * np.mean(centered_O * (E - energy)[:, None], axis=0)
        regularizer = args.sr_diag * max(0.1, 0.95**iteration)
        scale = np.maximum(np.diag(S), 1.0e-8)
        metric = S + regularizer * np.diag(scale) + 1.0e-10 * np.eye(len(scale))
        delta = np.linalg.solve(metric, -gradient)
        delta_norm = np.linalg.norm(delta)
        if delta_norm > args.max_step:
            delta *= args.max_step / delta_norm
        wf.theta += args.learning_rate * delta
        theta_norm = np.linalg.norm(wf.theta)
        if theta_norm > 0:
            wf.theta /= theta_norm
        wf.refresh()
        if iteration >= average_start:
            theta_tail.append(wf.theta.copy())
        history.append(
            (
                iteration,
                energy,
                float(np.var(E)),
                float(np.linalg.norm(gradient)),
                acceptance,
            )
        )
        if args.verbose and (
            iteration % args.print_every == 0 or iteration == args.iterations - 1
        ):
            print(
                f"SR {iteration:4d}: E={energy:+.12f} var={np.var(E):.3e} "
                f"|g|={np.linalg.norm(gradient):.3e} acc={acceptance:.3f}"
            )
    averaged_theta = np.mean(np.asarray(theta_tail), axis=0)
    averaged_norm = np.linalg.norm(averaged_theta)
    if not np.isfinite(averaged_norm) or averaged_norm == 0.0:
        raise ValueError("averaged variational parameters have zero or invalid norm")
    averaged_theta /= averaged_norm
    return state, history, averaged_theta, len(theta_tail)


def binned_mean_error(values, block_size):
    nblock = len(values) // block_size
    if nblock < 2:
        return float(np.mean(values)), float("nan")
    blocks = (
        np.asarray(values[: nblock * block_size])
        .reshape(nblock, block_size)
        .mean(axis=1)
    )
    return float(blocks.mean()), float(blocks.std(ddof=1) / math.sqrt(nblock))


def final_observable_statistics(E, Mz, Mx, L, block_size, infinite_temperature_energy):
    """Return means and non-overlapping-block standard errors.

    Var(H)=<|E_loc-<E>|^2>, and V-score=L*Var(H)/(<H>-E_inf)^2.
    """
    nblock = len(E) // block_size
    if nblock < 2:
        raise ValueError(
            "final-samples // block-size must be at least 2 for statistical errors"
        )
    nused = nblock * block_size
    E = E[:nused]
    Mz = Mz[:nused]
    Mx = Mx[:nused]
    Eb = E.reshape(nblock, block_size)
    Mzb = Mz.reshape(nblock, block_size)
    Mxb = Mx.reshape(nblock, block_size)
    energy = float(E.mean())
    variance = float(np.mean((E - energy) ** 2))
    block_energy = Eb.mean(axis=1)
    block_variance = np.mean((Eb - block_energy[:, None]) ** 2, axis=1)
    block_energy_difference = block_energy - infinite_temperature_energy
    block_v_score = np.full(nblock, np.nan)
    nonzero = block_energy_difference != 0.0
    block_v_score[nonzero] = (
        L * block_variance[nonzero] / block_energy_difference[nonzero] ** 2
    )
    energy_difference = energy - infinite_temperature_energy

    def sem(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) >= 2 else float("nan")

    return dict(
        energy_density=energy / L,
        energy_density_error=sem(block_energy / L),
        total_energy=energy,
        total_energy_error=sem(block_energy),
        total_energy_variance=variance,
        total_energy_variance_error=sem(block_variance),
        energy_variance_density=variance / L,
        energy_variance_density_error=sem(block_variance / L),
        infinite_temperature_energy=float(infinite_temperature_energy),
        energy_minus_infinite_temperature=energy_difference,
        v_score=(
            float(L * variance / energy_difference**2)
            if energy_difference != 0.0
            else float("nan")
        ),
        v_score_error=sem(block_v_score),
        longitudinal_magnetization=float(Mz.mean()),
        longitudinal_magnetization_error=sem(Mzb.mean(axis=1)),
        transverse_magnetization=float(Mx.mean()),
        transverse_magnetization_error=sem(Mxb.mean(axis=1)),
        final_samples_used=nused,
        final_blocks=nblock,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--jx", type=float, default=1.0, help="coefficient of P X P")
    parser.add_argument(
        "--z",
        type=float,
        default=1.0,
        help="fugacity used for the exact-state initializer",
    )
    parser.add_argument(
        "--jn", type=float, default=None, help="coefficient of P n P; default jx/z"
    )
    parser.add_argument(
        "--jp", type=float, default=None, help="coefficient of P P P; default jx*z"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--average-iterations",
        type=int,
        default=100,
        help="number of final SR iterations whose parameters are averaged",
    )
    parser.add_argument("--opt-samples", type=int, default=5000)
    parser.add_argument("--final-samples", type=int, default=40000)
    parser.add_argument("--opt-burn", type=int, default=400)
    parser.add_argument("--final-burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--sr-diag", type=float, default=1.0e-3)
    parser.add_argument("--max-step", type=float, default=0.5)
    parser.add_argument("--pf-rebuild-interval", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--ed-max-L", type=int, default=26)
    parser.add_argument("--exact-evaluate-max-L", type=int, default=20)
    parser.add_argument(
        "--output-prefix",
        default="dat",
        help="base name of files written inside the parameter directory",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="root directory under which an L/jx/jn/jp directory is created",
    )
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.L < 2 or args.L % 2:
        parser.error("L must be even and >= 2")
    if args.z <= 0:
        parser.error("--z must be positive")
    if args.jn is None:
        args.jn = args.jx / args.z
    if args.jp is None:
        args.jp = args.jx * args.z
    if args.pf_rebuild_interval < 0:
        parser.error("--pf-rebuild-interval must be nonnegative")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.average_iterations < 1:
        parser.error("--average-iterations must be positive")
    if args.average_iterations > args.iterations:
        parser.error("--average-iterations must not exceed --iterations")
    if args.block_size < 1:
        parser.error("--block-size must be positive")

    frustration_free = (
        abs(args.jn - args.jx / args.z) < 1.0e-13
        and abs(args.jp - args.jx * args.z) < 1.0e-13
        and args.jx > 0
    )
    print(f"H coefficients: jx={args.jx:g}, jn={args.jn:g}, jp={args.jp:g}")
    print(f"initializer fugacity z={args.z:g}; frustration_free={frustration_free}")

    rng = np.random.default_rng(args.seed)
    basis, labels = build_parameter_basis(args.L)
    exact_F = exact_lesanovsky_pairing(args.L, args.z)
    theta, residual = project_to_basis(exact_F, basis)
    if residual > 1.0e-10:
        raise RuntimeError(
            f"parameter basis cannot represent initializer: residual={residual}"
        )
    theta /= np.linalg.norm(theta)
    wf = BlockadePBCS(args.L, theta, basis)
    print(
        f"L={args.L}, variational parameters={len(theta)}, initializer residual={residual:.3e}"
    )

    initial_exact_energy = float("nan")
    initial_exact_variance = float("nan")
    initial_exact_fidelity = float("nan")
    ed_energy = float("nan")
    ed_vector = None
    ed_states = None
    if args.L <= args.exact_evaluate_max_L:
        initial_exact_energy, initial_exact_variance, initial_psi, initial_states = (
            exact_variational_observables(wf, args.jx, args.jn, args.jp)
        )
        if frustration_free:
            target = exact_lesanovsky_vector(initial_states, args.z, args.L)
            initial_exact_fidelity = float(abs(np.vdot(target, initial_psi)) ** 2)
        print(
            f"initial exact-sum: E={initial_exact_energy:+.12e}, "
            f"var={initial_exact_variance:.3e}, exact-state fidelity={initial_exact_fidelity:.12f}"
        )

    if args.L <= args.ed_max_L:
        ed_energy, ed_vector, ed_states = exact_ground_state(
            args.L, args.jx, args.jn, args.jp
        )
        print(
            f"ED ground state: E={ed_energy:+.12e}, blockade dimension={len(ed_states)}"
        )

    start = time.time()
    state, history, averaged_theta, averaged_count = sr_optimize(wf, args, rng, 0)
    last_theta = wf.theta.copy()
    wf.theta = averaged_theta.copy()
    wf.refresh()
    print(
        f"using normalized parameter average over final {averaged_count} SR iterations"
    )
    (
        infinite_temperature_energy,
        blockade_dimension_tm,
        infinite_mean_n1,
        infinite_mean_n000,
    ) = infinite_temperature_observables_transfer(args.L, args.jn, args.jp)
    print(
        f"infinite-temperature blockade trace: E_inf={infinite_temperature_energy:+.12e}, "
        f"dimension={blockade_dimension_tm}, <N1>_inf={infinite_mean_n1:.12e}, "
        f"<N000>_inf={infinite_mean_n000:.12e}"
    )
    E, _, Mz, Mx, state, acceptance = sample_observables(
        wf, args, rng, state, final=True
    )
    observables = final_observable_statistics(
        E, Mz, Mx, args.L, args.block_size, infinite_temperature_energy
    )
    vmc_mc_energy = observables["total_energy"]
    vmc_mc_error = observables["total_energy_error"]
    vmc_mc_variance = observables["total_energy_variance"]

    exact_energy = float("nan")
    exact_variance = float("nan")
    fidelity_ed = float("nan")
    if args.L <= args.exact_evaluate_max_L:
        exact_energy, exact_variance, psi, states = exact_variational_observables(
            wf, args.jx, args.jn, args.jp
        )
        if ed_vector is not None and np.array_equal(states, ed_states):
            fidelity_ed = float(abs(np.vdot(ed_vector, psi)) ** 2)

    print("averaged-wavefunction observables (mean +/- block standard error):")
    print(
        f"  energy density          = {observables['energy_density']:+.12e} "
        f"+/- {observables['energy_density_error']:.3e}"
    )
    print(
        f"  total energy variance   = {observables['total_energy_variance']:.12e} "
        f"+/- {observables['total_energy_variance_error']:.3e}"
    )
    print(
        f"  energy variance density = {observables['energy_variance_density']:.12e} "
        f"+/- {observables['energy_variance_density_error']:.3e}"
    )
    print(
        f"  V-score                 = {observables['v_score']:.12e} "
        f"+/- {observables['v_score_error']:.3e}"
    )
    print(
        f"  longitudinal magnetization <Z>/L = "
        f"{observables['longitudinal_magnetization']:+.12e} "
        f"+/- {observables['longitudinal_magnetization_error']:.3e}"
    )
    print(
        f"  transverse magnetization <PXP>/L = "
        f"{observables['transverse_magnetization']:+.12e} "
        f"+/- {observables['transverse_magnetization_error']:.3e}"
    )
    print(
        f"  total energy            = {vmc_mc_energy:+.12e} "
        f"+/- {vmc_mc_error:.3e}; acc={acceptance:.3f}"
    )
    if np.isfinite(exact_energy):
        print(
            f"VMC(exact sum) after SR: E={exact_energy:+.12e}, var={exact_variance:.3e}, "
            f"dE_ED={exact_energy-ed_energy:+.3e}, fidelity_ED={fidelity_ed:.12f}"
        )
    if frustration_free:
        tolerance = 1.0e-9
        verified = (
            abs(initial_exact_energy) < tolerance
            and initial_exact_variance < tolerance
            and initial_exact_fidelity > 1.0 - tolerance
        )
        print(f"Lesanovsky exact initializer verified={verified}")

    def parameter_tag(value):
        # Stable, shell-friendly representation without losing float identity.
        text = format(float(value), ".12g")
        return text.replace("-", "m").replace("+", "p").replace(".", "p")

    parameter_directory = Path(args.output_root) / (
        f"L{args.L}_jx{parameter_tag(args.jx)}_"
        f"jn{parameter_tag(args.jn)}_jp{parameter_tag(args.jp)}"
    )
    parameter_directory.mkdir(parents=True, exist_ok=True)
    output_name = Path(args.output_prefix).name
    prefix = parameter_directory / output_name
    summary = dict(
        L=args.L,
        jx=args.jx,
        jn=args.jn,
        jp=args.jp,
        z_initializer=args.z,
        frustration_free=frustration_free,
        parameters=len(theta),
        initial_exact_energy=initial_exact_energy,
        initial_exact_variance=initial_exact_variance,
        initial_exact_fidelity=initial_exact_fidelity,
        averaged_iterations=averaged_count,
        blockade_dimension_transfer=blockade_dimension_tm,
        infinite_temperature_mean_n1=infinite_mean_n1,
        infinite_temperature_mean_n000=infinite_mean_n000,
        vmc_mc_energy=vmc_mc_energy,
        vmc_mc_error=vmc_mc_error,
        vmc_mc_variance=vmc_mc_variance,
        **observables,
        vmc_exact_energy=exact_energy,
        vmc_exact_variance=exact_variance,
        ed_energy=ed_energy,
        energy_difference_exact=exact_energy - ed_energy,
        fidelity_ed=fidelity_ed,
        acceptance=acceptance,
        elapsed_seconds=time.time() - start,
    )
    with Path(str(prefix) + ".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    np.savez(
        str(prefix) + "_parameters.npz",
        theta=wf.theta,
        averaged_theta=wf.theta,
        last_theta=last_theta,
        averaged_iterations=averaged_count,
        labels=np.asarray(labels),
        F=wf.F,
        history=np.asarray(history, float),
    )
    with Path(str(prefix) + "_sr_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "energy", "variance", "grad_norm", "acceptance"])
        writer.writerows(history)
    print(
        "saved:",
        str(prefix) + ".csv",
        str(prefix) + "_parameters.npz",
        str(prefix) + "_sr_history.csv",
    )


if __name__ == "__main__":
    main()
