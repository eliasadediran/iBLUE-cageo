import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from src.helpers import line_utils, data_utils, matrix_utils
from scipy.stats import genextreme, norm
from numpy.lib.stride_tricks import sliding_window_view

# Collection of utility functions for FFT Computations for Uncertainty


def compute_residual(data_strip: np.ndarray, normalize_residual=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the residual error from estimating the data using
    linear interpolation

    This function computes the estimate for the data strip
    using the edge values and returns the residual error

    Parameters
    ----------

    data_strip : np.array
                 Bathymetric data strips re-aranged to a single strip

    Returns
    -------
    residual : np.array
               Difference of the interpolation from the input data strip

    """

    interpolated_strip = np.linspace(start=data_strip[:, 0],
                                     stop=data_strip[:, -1],
                                     num=data_strip.shape[1])

    interpolated_strip = interpolated_strip.T
    residual = data_strip - interpolated_strip
    if normalize_residual:
        residual = (residual/data_strip)*100
    return residual, interpolated_strip

def compute_energy(data: np.ndarray,
                   resolution: int,
                   method: str,
                   window_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT energy using 'method' process

    Parameters
    ----------
    data : np.array
           Input data
    resolution : int
                 Spatial resoluton of the array
    method : str
             FFT Method used to estimate signal energy

    Returns
    -------
    np.array
            Spectral energy in the signal
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
    Create the distance and frequency dependent scaling factors.
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
    Estimate the uncertainty using FFT

    Parameters
    ----------
    data : np.array
           Input data for FFT estimation
    multiple : int
               Window length as multiple of the linespacing
    resolution : int
                 Input data resolution for frequency calculation
    windowing : str
                Type of window to taper input
                options: scipy.signal.windows type
    method : str
             Type of FFT to estimate energy, defaults to 'amplitude'
             options: ['amplitude', 'psd', 'spectrum']


    Returns
    -------
    output : np.ndarray
        Uncertainty estimate from the FFT method
        To be compared with the residual error

    """

    if data.ndim < 2:
        data = data.reshape(1, -1)

    # create window
    segment_window = signal.windows.get_window(
        window=windowing, Nx=data.shape[1], fftbins=True)

    # preprocess_signal, could be modified later
    preprocessed_signal = data * segment_window

    energy, energy_freqs = compute_energy(preprocessed_signal,
                                            resolution,
                                            method,
                                            segment_window)

    df = 1.0 / (data.shape[1] * resolution)

    # compute contribution per frequency
    linespacing = int((data.shape[1] - 2) / multiple) * resolution
    spatial_signal = create_spatial_signal(resolution, data.shape[1], linespacing)
    variance = (energy * df) @ spatial_signal
    if method == 'PSD95':
        window_uncertainty = np.sqrt(variance) * 1.96 #2.6 #1.96 #2 #2.17
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
    Efficiently compute uncertainty estimates (diff, std, GEV, Gaussian)
    using sliding window views instead of explicit loops.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (num_lines, num_samples).
    min_window : int
        Minimum window length to start sliding over.
    multiple : int
        Multiplier controlling the spacing of the windows.
    method : str
        One of ['diff', 'std', 'gev', 'gaussian'].

    Returns
    -------
    Tuple of np.ndarray
        Depending on `method`, returns corresponding uncertainty statistics.
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
