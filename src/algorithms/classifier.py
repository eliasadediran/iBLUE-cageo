import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats

def compute_2d_psd(bathy, dx=1.0, dy=1.0, window=True):
    """
    Compute 2D PSD of bathymetry (power per unit area vs wavenumber).
    Returns:
      Pk2d: 2D PSD array (same shape as bathy)
      kx, ky: 2D wavenumber grids (rad/m)
      K: radial wavenumber grid
    Notes:
      - bathy should be detrended (we detrend with a 2D linear fit here).
      - windowing reduces spectral leakage.
    """
    ny, nx = bathy.shape
    # detrend (remove plane)
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dy)
    # fit plane
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, bathy.ravel(), rcond=None)
    plane = (A @ coeffs).reshape(bathy.shape)
    z = bathy - plane
    z_detrend = z.copy()

    if window:
        wx = scipy.signal.windows.hann(nx, sym=False)
        wy = scipy.signal.windows.hann(ny, sym=False)
        w2d = (np.outer(wy, wx))
        z = z * w2d
        # U = np.mean(w2d ** 2)  # window power correction
        U = np.sum(w2d ** 2) / (nx * ny)  # exact discrete average of squared window
    else:
        U = 1.0

    # 2D FFT
    Z = np.fft.fft2(z)
    Z = np.fft.fftshift(Z)
    # P2 = (np.abs(Z)**2) / (nx*ny)
    P2 = (dx * dy / (nx * ny * U)) * (np.abs(Z) ** 2)

    # build k grids (rad/m)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    Kx, Ky = np.meshgrid(kx, ky, indexing='xy')
    K = np.sqrt(Kx**2 + Ky**2)

    # Parseval
    dkx = 1 / (nx * dx)
    dky = 1 / (ny * dy)

    variance_psd = np.sum(P2) * dkx * dky
    variance_spatial = np.var(z)
    # variance_spatial = np.var(z)*1/np.sqrt(U)
    # if window:
    #     variance_spatial = np.var(z_detrend)
    # else:
    #     variance_spatial = np.var(z_detrend)

    # print(variance_spatial, variance_psd)

    return P2, Kx, Ky, K


def radial_average(P2, K, nbins=200, min_counts=10, verbose=False):
    """Compute radial average of 2D PSD -> 1D PSD(Pk) vs k (rad/m) with bin counts."""
    kflat = K.ravel()
    Pflat = P2.ravel()
    # mask zero
    mask = kflat > 0
    kflat = kflat[mask];
    Pflat = Pflat[mask]
    kmin = kflat.min()
    kmax = kflat.max()
    bins = np.logspace(np.log10(kmin), np.log10(kmax), nbins)
    kb = 0.5 * (bins[:-1] + bins[1:])

    counts = np.array([((kflat >= bins[i]) & (kflat < bins[i + 1])).sum() for i in range(len(bins) - 1)])
    if verbose:
        print("Min modes per bin:", counts.min())
        print("Max modes per bin:", counts.max())

    Pbin = np.zeros_like(kb)
    for i in range(len(kb)):
        if counts[i] >= min_counts:
            sel = (kflat >= bins[i]) & (kflat < bins[i + 1])
            Pbin[i] = Pflat[sel].mean()
        else:
            Pbin[i] = np.nan

    # remove nan
    ok = ~np.isnan(Pbin) & (Pbin > 0)
    return kb[ok], Pbin[ok], counts[ok], kmin, kmax


def fit_powerlaw(k, Pk, counts=None, kmin_pct=10, kmax_pct=85, min_counts=10):
    """
    Fit a linear fit to log10(Pk) = intercept + slope * log10(k) using a stable inertial range.
    Automatically selects k_fit_min/k_fit_max based on percentiles and minimum bin counts.

    Returns slope_beta (positive) such that P ~ k^-beta
    """
    if counts is None:
        counts = np.ones_like(Pk)  # fallback if no info

    # percentile-based trimming
    k_sorted = np.sort(k)
    kmin_fit = np.percentile(k_sorted, kmin_pct)
    kmax_fit = np.percentile(k_sorted, kmax_pct)

    # also apply minimum counts filter
    sel = (k >= kmin_fit) & (k <= kmax_fit) & (counts >= min_counts)
    x = np.log10(k[sel])
    y = np.log10(Pk[sel])

    slope, intercept, r, p, std = stats.linregress(x, y)
    beta = -slope  # because P ~ k^-beta

    return beta, intercept, r, k[sel].min(), k[sel].max()

def seabed_classifier(bathy, outdir, resolution, seabed, title, nbins=200, kmin_pct=10, kmax_pct=85, min_counts=15,  plot=True):
    P2, Kx, Ky, K = compute_2d_psd(bathy, dx=resolution, dy=resolution, window=True)
    k, Pk, counts, kmin_raw, kmax_raw = radial_average(P2, K, nbins=nbins, min_counts=10, verbose=False)
    # pick fit band visually or automatically: skip extreme low and high k
    beta, intercept_log10, r, kmin_fit, kmax_fit = fit_powerlaw(k, Pk, counts=counts, kmin_pct=kmin_pct, kmax_pct=kmax_pct, min_counts=min_counts)
    depth_mean = np.nanmean(bathy)
    depth_std  = np.nanstd(bathy)
    depth_range = np.ptp(bathy)
    depth_min = np.min(bathy)
    depth_max = np.min(bathy)

    if plot:
        fig, ax = plt.subplots(1,2, figsize=(9,3.5))
        ax[0].imshow(np.log10(P2 + 1e-20), origin='lower', extent=[Kx.min(), Kx.max(), Ky.min(), Ky.max()])
        ax[0].set_title('log10 2D PSD in k-space (rad/m)')
        ax[0].set_xlabel('kx (rad/m)')
        ax[0].set_ylabel('ky (rad/m)')
        # plot fitted line
        ax[1].loglog(k, Pk, '-k', label='radial PSD')
        xfit = np.array([kmin_fit, kmax_fit])
        yfit = 10**(intercept_log10) * xfit**(-beta)
        ax[1].loglog(xfit, yfit, '--r', label=f'fit beta={beta:.2f}\nintercept={intercept_log10:.2f}')
        ax[1].axvline(kmin_fit, color='b', linestyle=':', label=f'lower band')
        ax[1].axvline(kmax_fit, color='g', linestyle=':', label=f'upper band')
        ax[1].legend()
        ax[1].set_title(f'{seabed.capitalize()} Radial PSD and fit ({nbins}bins)')
        ax[1].set_xlabel('Wavenumber k (rad/m)')
        # ax[1].set_ylabel('P(k) (m^3·rad^-1)')
        # ax[1].set_ylabel(r'$\mathregular{P(k)\;(m^3\,rad^{-1})}$')
        ax[1].set_ylabel(r'$P(k)$ (m$^3$ rad$^{-1}$)')
        # ax[1].set_ylabel(r'$P(k)$ (m$^3$ rad$^{-1}$)')
        plt.tight_layout()
        plt.savefig(f"{outdir}{title}_{nbins}bins_selectk.png",dpi=300, bbox_inches="tight")
        plt.close(fig)

    out = dict(
        seabed = seabed,
        beta_main = beta,
        intercept_log10 = intercept_log10, r=r, kmin_fit=kmin_fit, kmax_fit=kmax_fit, kmin_raw=kmin_raw, kmax_raw=kmax_raw,
        depth_mean = depth_mean, depth_std = depth_std, depth_range=depth_range, depth_max=depth_max, depth_min=depth_min)

    df = pd.DataFrame([out])
    df.to_csv(f'{outdir}{title}_{nbins}bins_allk.csv', index=False)

    # df = pd.DataFrame([out])
    # df.to_csv(f'{outdir}{title}_{nbins}bins_selectk.csv', index=False)
    return np.round(beta,3)