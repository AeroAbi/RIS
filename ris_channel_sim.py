"""
Stage 0 — RIS-Assisted Channel Simulator
==========================================
Single BS -> RIS -> single User link.

Model:
    y = (h2^H diag(theta) h1) x + n

    h1 : BS -> RIS channel, shape (N,)   (N = number of RIS elements)
    h2 : RIS -> user channel, shape (N,)
    theta : RIS phase-shift vector, shape (N,), theta_n = exp(j * phi_n)

This module gives you:
    - generate_channel(N, k_factor): returns h1, h2 (Rician fading)
    - compute_snr(h1, h2, theta, tx_power, noise_power): SNR for a given phase config
    - random_phases(N): a random baseline phase vector
    - align_phases(h1, h2): a simple phase-alignment heuristic 
"""

from turtle import lt
import numpy as np


def generate_channel(N, k_factor=3.0, rng=None):
    """
    Generate Rician-faded channel vectors for BS->RIS (h1) and RIS->user (h2).

    Parameters
    ----------
    N : int
        Number of RIS elements.
    k_factor : float
        Rician K-factor (ratio of LOS power to scattered power). Higher = more
        LOS-dominant, which is realistic for near-LOS mmWave RIS links.
    rng : np.random.Generator, optional
        Pass your own generator for reproducibility; otherwise a fresh one is used.

    Returns
    -------
    h1, h2 : complex np.ndarray, shape (N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    def rician_vector(n):
        # LOS component: unit-magnitude, deterministic phase (e.g., zero here;
        # in a full model this would come from array steering vectors)
        los = np.ones(n, dtype=complex)
        # NLOS scattered component: complex Gaussian
        nlos = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)

        k = k_factor
        los_scale = np.sqrt(k / (k + 1))
        nlos_scale = np.sqrt(1 / (k + 1))
        return los_scale * los + nlos_scale * nlos

    h1 = rician_vector(N)
    h2 = rician_vector(N)
    return h1, h2


def compute_snr(h1, h2, theta, tx_power=1.0, noise_power=1e-3):
    """
    Compute the received SNR for a given RIS phase configuration.

    Parameters
    ----------
    h1, h2 : complex np.ndarray, shape (N,)
    theta : complex np.ndarray, shape (N,)
        RIS phase-shift vector, elements should have unit magnitude
        (e.g., theta_n = exp(1j * phi_n)).
    tx_power : float
        Transmit power (linear scale).
    noise_power : float
        Noise power (linear scale).

    Returns
    -------
    snr : float
    effective_gain : complex
        The scalar cascaded channel gain (h2^H diag(theta) h1), useful for debugging.
    """
    effective_gain = np.conj(h2) @ (theta * h1)
    signal_power = tx_power * (np.abs(effective_gain) ** 2)
    snr = signal_power / noise_power
    SNR_dB = 10 * np.log10(snr)
    return snr, effective_gain


def random_phases(N, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, size=N)
    return np.exp(1j * phi)

def align_phases(h1, h2):
    """
    Simple phase-alignment heuristic: choose each element's phase to cancel
    the combined phase of h1 and h2, so every element's contribution adds
    coherently (in-phase) at the receiver. This is NOT the full optimizer
    (that's Stage 1) but gives a quick, correct-in-spirit baseline.
    """
    combined_phase = np.angle(h1) + np.angle(np.conj(h2))
    phi = -combined_phase
    return np.exp(1j * phi)
#
def _sanity_check():
    """
    Quick check: as N grows, SNR under phase-aligned RIS should scale
    roughly as N^2, while random phases should scale much more slowly
    (roughly N due to incoherent combining).
    """
    rng = np.random.default_rng(42)
    
    print(f"{'N':>4} | {'angle-random_deg':>20} | {'SNR-random_dB':>15} | {'angle-Aligned_deg':>20} | {'SNR-aligned_dB':>15}")
    print("-" * 95)

    # print(f"{'N':>4} | {'angle-Aligned_deg':>20} | {'SNR-aligned_dB':>15}")
    # print("-" * 55)

    for N in [2, 4, 8, 16, 32, 64]:
        h1, h2 = generate_channel(N, k_factor=3.0, rng=rng)

        theta_aligned = align_phases(h1, h2)
        snr_aligned, _ = compute_snr(h1, h2, theta_aligned)
        snr_aligned_db = 10*np.log10(snr_aligned)
        anglealigned_deg = np.degrees(np.angle(theta_aligned[0]))
        anglesaligned_deg = np.degrees(np.angle(theta_aligned))

        theta_random = random_phases(N, rng=rng)
        snr_random, _ = compute_snr(h1, h2, theta_random)
        snr_random_db = 10*np.log10(snr_random)
        angle_deg = np.degrees(np.angle(theta_random[0]))
        angles_deg = np.degrees(np.angle(theta_random))
        
        #ratio = snr_aligned / (N ** 2)
        print(f"{N:>4}| {angle_deg:>20.2f} | {snr_random_db:>15.2f} |{anglealigned_deg:>20.2f} | {snr_aligned_db:>15.2f} ")
        # print(f"{N:>4} | {angle_deg:>20.2f} | {snr_random_db:>15.2f}")
        # print(f"{N:>4} | {anglealigned_deg:>20.2f} | {snr_aligned_db:>15.2f}")
        # print(angles_deg)
        
        
def _plot_scaling(n_trials=200, N_values=(2, 4, 8, 16, 32, 64, 128)):
    """
    Monte Carlo average over many channel realizations per N, then plot:
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    snr_aligned_avg = []
    snr_random_avg = []
    all_angles = []

    for N in N_values:
        aligned_trials = []
        random_trials = []
        for _ in range(n_trials):
            h1, h2 = generate_channel(N, k_factor=3.0, rng=rng)

            theta_aligned = align_phases(h1, h2)
            snr_a, _ = compute_snr(h1, h2, theta_aligned)
            aligned_trials.append(snr_a)

            theta_random = random_phases(N, rng=rng)
            all_angles.extend(np.degrees(np.angle(theta_random)))
            snr_r, _ = compute_snr(h1, h2, theta_random)
            random_trials.append(snr_r)

        snr_aligned_avg.append(np.mean(aligned_trials))
        snr_random_avg.append(np.mean(random_trials))

    snr_aligned_avg = np.array(snr_aligned_avg)
    snr_random_avg = np.array(snr_random_avg)
    N_arr = np.array(N_values)


    snr_aligned_db = 10 * np.log10(snr_aligned_avg)
    snr_random_db = 10 * np.log10(snr_random_avg)
    #ratio = snr_aligned_avg / (N_arr ** 2)

    tx_power_mw = 1.0      # mW
    noise_power_mw = 1e-3  # mW
    #signal_power_aligned_mw = snr_aligned_avg * noise_power_mw
    signal_power_random_mw = snr_random_avg * noise_power_mw
    #signal_power_aligned_dbm = 10 * np.log10(signal_power_aligned_mw)
    signal_power_random_dbm = 10 * np.log10(signal_power_random_mw)
    noise_power_dbm = 10 * np.log10(noise_power_mw)
#
    plt.figure(figsize=(7,5))
    plt.plot(N_arr, snr_random_db, marker='s', label='Random RIS phases')
    plt.xscale('log', base=2)
    plt.xlabel('N (number of RIS elements)')
    plt.ylabel('Average SNR (dB)')
    plt.title(f'Random SNR vs N ({n_trials} trials Monte Carlo per N)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(7,5))
    plt.plot(N_arr, snr_aligned_db, marker='s', label='Aligned RIS phases')
    plt.xscale('log', base=2)
    plt.xlabel('N (number of RIS elements)')
    plt.ylabel('Average SNR (dB)')
    plt.title(f'Aligned SNR vs N ({n_trials} trials Monte Carlo per N)')
    plt.legend()
    plt.grid()

    plt.figure(figsize=(9,5))
    elements = np.arange(1, len(theta_random)+1)
    plt.plot(elements,np.degrees(np.angle(theta_random)),'o-',label='Random')
    plt.plot(elements,np.degrees(np.angle(theta_aligned)),'s-',label='Aligned')
    plt.xlabel("RIS Element Index")
    plt.ylabel("Phase Angle (degrees)")
    plt.title(f"RIS Phase Configuration (N={len(theta_random)})")
    plt.legend()
    plt.grid(True)
    plt.show()

   
# Histogram

    # plt.figure(figsize=(10,10))
    # plt.hist(all_angles, bins=36, edgecolor='black')
    # plt.xlabel("Phase (degrees)")
    # plt.ylabel("Number of RIS Elements")
    # plt.title("Distribution of All Random RIS Phases")
    # plt.grid(True, alpha=0.3)
    # plt.show()
   
if __name__ == "__main__":
    _sanity_check()
    _plot_scaling()   
   