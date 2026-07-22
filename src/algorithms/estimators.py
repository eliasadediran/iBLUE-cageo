import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from src.helpers import line_utils, data_utils, matrix_utils
from scipy.stats import genextreme, norm
from numpy.lib.stride_tricks import sliding_window_view

# Spectral and spatial statistical estimators for quantifying interpolation uncertainty.
# These functions transform bathymetric strips into uncertainty estimates by
# comparing observed values with a locally interpolated reference signal.
def compute_energy(data: np.ndarray,
                   resolution: int,
                   method: str,
                   window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency-domain energy of a signal after tapering.

    The result is used to estimate how much variance is present at each
    wavelength scale, which is then converted into an uncertainty surface.

    Parameters
    ----------
    data : np.ndarray
        One or more bathymetric strips that have been tapered with a window.
    resolution : int
        Spatial resolution of the data in meters per sample.
    method : str
        FFT-based energy method to use. The implementation supports the
        spectral estimators used in the manuscript workflow.
    window_values : np.ndarray
        Tapering window applied to the signal prior to the transform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Spectral energy values and the corresponding frequency vector.
    """

    rfft_values = np.abs(np.fft.rfft(data, axis=1))
    _, num_cols = rfft_values.shape
    r_frequencies = np.fft.rfftfreq(data.shape[1], d=resolution)

    cden = np.sqrt(np.sum(window_values** 2))
    energy =  resolution * (np.abs(rfft_values / cden)**2)

    if num_cols % 2 == 0:  # even length → Nyquist bin exists
        energy[:, 1:-1] *= 2
    else:  # odd length → no Nyquist bin
        energy[:, 1:] *= 2

    return energy, r_frequencies
#
def create_spatial_signal(resolution: int, max_cell_number: int, line_spacing):
    """
    Create a wavelength-dependent spatial scaling signal.

    The returned signal is used to weight spectral energy according to the
    expected structure of the seabed at different spatial scales.
    """
    frequencies = np.fft.rfftfreq(max_cell_number, resolution)
    distances = np.arange(max_cell_number) * resolution
    distances_2d, freq_2d = np.meshgrid(distances, frequencies)

    # Sine Kernel
    spatial_scale = distances_2d * freq_2d
    spatial_scale = np.where(spatial_scale < 0.25, spatial_scale, 0.25)
    spatial_signal = np.sin(spatial_scale * 2 * np.pi)

    return spatial_signal


def spectral_estimator(
    data: np.ndarray,
    multiple: int,
    resolution: int,
    windowing: str = 'hann',
    method: str = "PSD95",
    selection: str = "half",
) -> np.ndarray:
    """
    Estimate uncertainty from the spectral structure of a bathymetric strip.

    The input strip is tapered, transformed into the frequency domain, and then
    converted into a strip of uncertainty values that can be compared against
    residuals from the interpolation workflow.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array of bathymetric values arranged as strips.
    multiple : int
        Window length expressed as a multiple of the line spacing.
    resolution : int
        Spatial resolution of the input data in meters per sample.
    windowing : str
        Windowing function used to reduce edge effects before the transform.
    method : str
        Spectral method to use. Supported values are "PSD95" and "PSD99".
    selection : str
        Strategy for selecting frequency contributions. The default uses a
        symmetric half-spectrum representation.

    Returns
    -------
    np.ndarray
        A strip of spectral uncertainty estimates with the same structure as
        the input data strip.
    """

    if data.ndim < 2:
        data = data.reshape(1, -1)

    # create window
    segment_window = signal.windows.get_window(
        window=windowing, Nx=data.shape[1], fftbins=True)

    # preprocess_signal
    preprocessed_signal = data * segment_window

    energy, energy_freqs = compute_energy(preprocessed_signal,
                                            resolution,
                                            method,
                                            segment_window)

    # frequency resolution
    df = 1.0 / (data.shape[1] * resolution)

    # compute contribution per frequency
    linespacing = int((data.shape[1] - 2) / multiple) * resolution
    spatial_signal = create_spatial_signal(resolution, data.shape[1], linespacing)
    variance = (energy * df) @ spatial_signal
    if method == 'PSD95':
        window_uncertainty = np.sqrt(variance) * 1.96
    elif method == 'PSD99':
        window_uncertainty = np.sqrt(variance) * 2.6
    else:
        raise ValueError('method must be "PSD95" or "PSD99"')

    # Remove edges when computing the original linespacing
    linespacing_width = int((data.shape[1]-2) / multiple)

    # Include edges again for the output strip
    output = np.zeros(shape=(data.shape[0], linespacing_width + 2))
    num_cols = output.shape[1]

    if selection == "half":

        half = num_cols // 2
        if num_cols % 2:  # odd
            selected_data = window_uncertainty[:, :half + 1]
            output[:, :half + 1] = selected_data
            output[:, half + 1:] = np.fliplr(selected_data[:, :-1])

        else:  # even
            selected_data = window_uncertainty[:, :half]
            output[:, :half] = selected_data
            output[:, half:] = np.fliplr(selected_data)
    else:
        # pick energy from frequencies only present in original data
        freqs_window = np.fft.rfftfreq(int(data.shape[1] / multiple),
                                       resolution)
        freq_idxs = np.where(np.isin(freqs_window, energy_freqs))[0]
        selected_data = window_uncertainty[:, freq_idxs]
        output[:, :int(num_cols/2)] = selected_data
        output[:, int(num_cols/2):] = np.fliplr(selected_data)

    return output


def statistical_estimator(
    data: np.ndarray,
    line_spacing,
    min_window: int = 2,
    multiple: int = 1,
    method: str = "SAR",
):
    """
    Estimate uncertainty from spatial statistics in a sliding-window fashion.

    This method evaluates local variability in the bathymetric strip by
    computing range-based statistics across multiple window sizes. The
    resulting estimates are used as a complementary uncertainty measure to the
    spectral approach.

    Parameters
    ----------
    data : np.ndarray
        Two-dimensional array of shape (num_lines, num_samples).
    line_spacing : float or int
        The target line spacing used to interpret the spatial scale.
    min_window : int
        Minimum window length to begin the sliding-window calculation.
    multiple : int
        Multiplier controlling the spacing of the windows.
    method : str
        One of "SAR", "SER", "SMR", or "ALL".

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray]
        The requested uncertainty statistic or all three statistics.
    """

    num_lines, num_samples = data.shape
    interpolation_cell_distance = ((num_samples - 2) // multiple) + 2
    half = interpolation_cell_distance // 2

    # Preallocate arrays
    shape = (num_lines, interpolation_cell_distance)
    SAR = np.zeros(shape)
    SMR = np.zeros(shape)
    SER = np.zeros(shape)

    # --- Main computation loop ---
    if interpolation_cell_distance % 2:
        max_len = interpolation_cell_distance // 2 + 1
    else:
        max_len = interpolation_cell_distance // 2
    for win_len in range(min_window, max_len + 1):
        windows = sliding_window_view(data, window_shape=win_len, axis=-1)
        # windows shape: (num_lines, num_convolutions, win_len)
        mins = np.min(windows, axis=-1)
        maxs = np.max(windows, axis=-1)
        ranges = maxs - mins

        range_mean = np.mean(ranges, axis=-1)
        range_std = np.std(ranges, axis=-1)
        range_max = np.max(ranges, axis=-1)
        SAR[:, win_len-1] = range_mean
        SER[:, win_len - 1] = range_mean + range_std
        SMR[:, win_len-1] = range_max

    # --- Mirror the ends for symmetry (using half instead of win_len) ---
    width = interpolation_cell_distance
    is_odd = width % 2

    if is_odd:
        SAR[:, half + 1:] = np.fliplr(SAR[:, :half])
        SER[:, half + 1:] = np.fliplr(SER[:, :half])
        SMR[:, half + 1:] = np.fliplr(SMR[:, :half])
    else:
        SAR[:, half:] = np.fliplr(SAR[:, :half])
        SER[:, half:] = np.fliplr(SER[:, :half])
        SMR[:, half:] = np.fliplr(SMR[:, :half])

    if method == 'SAR':
        return SAR
    elif method == 'SER':
        return SER
    elif method == 'SMR':
        return SMR
    elif method == 'ALL':
        return SAR, SER, SMR

def compute_residual(data_strip: np.ndarray, normalize_residual=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the residual error between the observed strip and a linear interpolation.

    The function uses the first and last values of each row as the endpoints of a
    linear interpolation and returns the resulting residual. These residuals are
    later used as the reference signal for uncertainty estimation.

    Parameters
    ----------
    data_strip : np.ndarray
        Bathymetric data arranged as a strip with one value per sample along the
        across-track direction.
    normalize_residual : bool
        If True, convert the residual to a percent-based value relative to the
        observed data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Residual values and the corresponding linearly interpolated values.
    """

    interpolated_strip = np.linspace(start=data_strip[:, 0],
                                     stop=data_strip[:, -1],
                                     num=data_strip.shape[1])

    interpolated_strip = interpolated_strip.T
    residual = data_strip - interpolated_strip
    if normalize_residual:
        residual = (residual/data_strip)*100
    return residual, interpolated_strip