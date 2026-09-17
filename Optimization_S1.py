"""
Stage 1 — Classical Baseline Optimization
===========================================
Goal: get "ground truth" optimal RIS phase configurations (theta*) to
later compare the Stage 2 ML surrogate against.

Two optimizers are implemented:

1. exhaustive_search(h1, h2, n_levels)
   Brute-force grid search over discretized phases, one of n_levels values
   per element. Only tractable for small N (4-8 elements) since the search
   space is n_levels^N. Used here as ground truth to VALIDATE that the
   closed-form heuristic below is actually optimal.

2. coordinate_ascent(h1, h2, n_levels, n_iters)
   Iteratively optimizes one element's phase at a time (holding the others
   fixed), searching over a discretized grid for that element. Scales to
   much larger N since cost per iteration is only N * n_levels (not
   n_levels^N). Converges in very few iterations for this model.

Both are compared against `align_phases()` from Stage 0 (the closed-form
phase-alignment heuristic). For this single-user scalar RIS model, the
received signal is a coherent sum of N independent per-element terms, so
maximizing |sum| is achieved by aligning every term to a common phase --
meaning align_phases() is actually the TRUE global optimum here, not just
a heuristic. Stage 1's job is to confirm that numerically before trusting
it as the ML training label source in Stage 2.

"""

import itertools
import sys
from pathlib import Path

import numpy as np

from ris_channel_sim import generate_channel, compute_snr, random_phases, align_phases

def exhaustive_search(h1, h2, n_levels=8, tx_power=1.0, noise_power=1e-3):
    """
    Brute-force search over a discretized phase grid, one of n_levels
    equally spaced phase values per element. Only tractable for small N
    (search space size = n_levels ** N).

    Returns
    -------
    best_theta : complex np.ndarray, shape (N,)
    best_snr : float
    """
    N = len(h1)
    phase_grid = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)

    best_snr = -np.inf
    best_theta = None

    for combo in itertools.product(range(n_levels), repeat=N):
        phi = phase_grid[list(combo)]
        theta = np.exp(1j * phi)
        snr, _ = compute_snr(h1, h2, theta, tx_power, noise_power)
        if snr > best_snr:
            best_snr = snr
            best_theta = theta

    return best_theta, best_snr


def coordinate_ascent(h1, h2, n_levels=64, n_iters=3, tx_power=1.0, noise_power=1e-3, init_theta=None):
    """
    Coordinate-wise ascent: optimize one element's phase at a time
    (holding all others fixed) over a fine discretized grid, then move to
    the next element. Repeats for n_iters full sweeps over all elements.

    Cost per full sweep: N * n_levels (much cheaper than exhaustive search's
    n_levels ** N), so this scales to large N.

    Returns
    -------
    theta : complex np.ndarray, shape (N,)
    snr_history : list of float
        SNR achieved after each element update (useful for a convergence plot).
    """
    N = len(h1)
    phase_grid = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)

    if init_theta is None:
        theta = random_phases(N)
    else:
        theta = init_theta.copy()

    snr_history = []
    for _ in range(n_iters):
        for n in range(N):
            best_snr_n = -np.inf
            best_phi_n = np.angle(theta[n])
            for phi in phase_grid:
                theta[n] = np.exp(1j * phi)
                snr, _ = compute_snr(h1, h2, theta, tx_power, noise_power)
                if snr > best_snr_n:
                    best_snr_n = snr
                    best_phi_n = phi
            theta[n] = np.exp(1j * best_phi_n)
            snr_history.append(best_snr_n)

    final_snr, _ = compute_snr(h1, h2, theta, tx_power, noise_power)
    return theta, snr_history, final_snr


def _validate_against_closed_form():
    """
    For small N, confirm exhaustive search and coordinate ascent both
    recover (approximately, given grid resolution) the same SNR as the
    closed-form align_phases() heuristic -- i.e., confirm align_phases is
    actually the global optimum for this model, not just a good guess.

    n_levels for exhaustive search is kept small enough to stay tractable
    (search space = n_levels ** N), shrinking as N grows.
    """
    rng = np.random.default_rng(1)
    # print("Validating optimizers against closed-form align_phases()")
    # N |  SNR (closed-form) |   SNR (exhaustive) |  SNR (coord. ascent)
    print(f"{'N':>4} | {'SNR-Mathematical Analysis':>18} | {'SNR-Try all Phases':>18} | {'SNR-Stepwise Optimization':>20}")
    print("-" * 80)

    # (N, n_levels_for_exhaustive) -- chosen so n_levels**N stays under ~200k
    configs = [(2, 12), (4, 10), (6, 6), (8, 4)]

    for N, ex_levels in configs:
        h1, h2 = generate_channel(N, k_factor=3.0, rng=rng)

        theta_cf = align_phases(h1, h2)
        snr_cf, _ = compute_snr(h1, h2, theta_cf)
        SNRcf_dB = 10 * np.log10(snr_cf)

        theta_ex, snr_ex = exhaustive_search(h1, h2, n_levels=ex_levels)
        theta_ca, _, snr_ca = coordinate_ascent(h1, h2, n_levels=64, n_iters=3)

        print(f"{N:>4} |     {SNRcf_dB:>18.2f} dB|{10 * np.log10(snr_ex):>18.2f} dB|{10 * np.log10(snr_ca):>20.2f} dB")

    # print("\n(Small gaps are expected -- exhaustive/coordinate ascent use a")
    # print(" discretized phase grid, while align_phases is exact/continuous.")
    # print(" Coarser grids at higher N explain any larger gap there.)")


def _plot_stage1(n_trials=100, N_values=(2, 4, 8, 16, 32, 64, 128)):
    """
    Plot achieved SNR (dB) and spectral efficiency / rate (bits/s/Hz)
    vs N, comparing the optimizer (closed-form / coordinate ascent) against
    random phases -- confirms the ~N^2 SNR scaling and shows the resulting
    rate gain.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    snr_opt_avg, snr_rand_avg = [], []
    rate_opt_avg, rate_rand_avg = [], []

    for N in N_values:
        opt_trials, rand_trials = [], []
        for _ in range(n_trials):
            h1, h2 = generate_channel(N, k_factor=3.0, rng=rng)

            # Use closed-form for N <= 8 sanity range and larger N alike --
            # Stage 1's validation above confirms it IS the optimum here.
            theta_opt = align_phases(h1, h2)
            snr_opt, _ = compute_snr(h1, h2, theta_opt)
            opt_trials.append(snr_opt)

            theta_rand = random_phases(N, rng=rng)
            snr_rand, _ = compute_snr(h1, h2, theta_rand)
            rand_trials.append(snr_rand)

        snr_opt_avg.append(np.mean(opt_trials))
        snr_rand_avg.append(np.mean(rand_trials))
        rate_opt_avg.append(np.mean(np.log2(1 + np.array(opt_trials))))
        rate_rand_avg.append(np.mean(np.log2(1 + np.array(rand_trials))))

    N_arr = np.array(N_values)
    snr_opt_db = 10 * np.log10(snr_opt_avg)
    snr_rand_db = 10 * np.log10(snr_rand_avg)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(N_arr, snr_opt_db, marker='o', label='Optimal aligned phases')
    axes[0].plot(N_arr, snr_rand_db, marker='s', label='Random phases')
    axes[0].set_xscale('log', base=2)
    axes[0].set_xlabel('N (number of RIS elements)')
    axes[0].set_ylabel('Average SNR (dB)')
    axes[0].set_title(f'Stage 1: Optimized SNR vs N ({n_trials} trials/N)')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].plot(N_arr, rate_opt_avg, marker='o', label='Optimal aligned phases')
    axes[1].plot(N_arr, rate_rand_avg, marker='s', label='Random phases')
    axes[1].set_xscale('log', base=2)
    axes[1].set_xlabel('N (number of RIS elements)')
    axes[1].set_ylabel('Data Rate (bits/s/Hz)') #log2(1+SNR)
    axes[1].set_title('Achieved rate vs N')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _validate_against_closed_form()
    _plot_stage1()