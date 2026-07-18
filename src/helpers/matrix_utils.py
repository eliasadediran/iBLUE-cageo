import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def get_column_indices(
    array_len: int,
    resolution: int,
    linespacing_meters: int,
    max_multiple: int,
) -> np.ndarray:
    """
    Determine indices of the columns to be used as sampling lines

    Parameters:
    -------------
    array_len : int
                Length of the array containing bathymetry data
    resolution : float
                    Spatial Resolution of the bathymetry data
    linespacing_meters : int
                            distance between 2 vertical sample lines (m)
    max_multiple : int
                    maximum multiple of the linespacing to be used
                    as window size for FFT


    Returns:
    -----------
    col_indxs : np.array
                array indices corresponding to sample points/lines

    """

    # Round the desired linespacing to the nearest even integer
    # for computational convenience
    linespacing_in_pixels = np.round(linespacing_meters / resolution)

    if array_len < linespacing_in_pixels:
        raise ValueError(
            f"""
            Desired linespacing should be less than the Spatial coverage.
            Entered Linespacing: {linespacing_meters}m
            Bathemetric coverage: {array_len * resolution}m"""
        )

    # Valid sampling columns is determined by the window length
    # Window lengths are multiples of the linespacing
    window_size_pixels = linespacing_in_pixels * max_multiple
    start_col = int(
        (window_size_pixels // 2) -
        (linespacing_in_pixels // 2)
                    )

    last_col = int(
        array_len
        - (window_size_pixels // 2)
        + (linespacing_in_pixels // 2)
        - 1
    )

    # actual sampling indices will be determined by the desired linespacing
    col_indxs = np.arange(start_col,
                          last_col,
                          (linespacing_in_pixels)).astype(int)

    return col_indxs


def get_strip(
    row: np.ndarray,
    column_indices: tuple[int, int],
    multiple: int,
) -> np.ndarray:
    """
    Retrieve section of the bathymetric data for FFT processing given
    column indices and multiple of the linespacing

    This function computes the strip of data for FFT processing. The width
    of the strip is the effective window size. The window size is the product
    of the linespacing and the value of the multiple parameter.

    Parameters
    ----------
    depth : np.array
            Bathymetric/depth data
    column_indices : np.array
                    column indices/location of the sampling lines
    multiple : int
            Multiple of the linespacing used to define the window size

    Returns
    --------
    strip : np.array
            a segment of the bathymetric data for further FFT processing

    """

    current_multiple = multiple
    start, end = column_indices[0], column_indices[1]
    linespacing = end - start - 1
    window_size = linespacing * current_multiple
    midpoint = start + (linespacing // 2) + 1

    # Determine indices for the window segment
    # +1 / -1 will include sampling columns at the edges
    start_col = int(midpoint - (window_size // 2)) - 1
    end_col = int(midpoint + (window_size // 2)) + 1
    strip = np.array([x[start_col:end_col] for x in row])

    return strip


def matrix2strip(
    depth: np.ndarray,
    column_indices: np.ndarray,
    multiple: int,
) -> np.ndarray:
    """
    Retrieve strip of the bathymetric data given column indices

    This function computes the strips of data for FFT processing. The width
    of a strip defined by the window size (which is a multiple of the
    linespacing).  All strips are concatenated vertically to take advantage
    of numpy's vectorization

    Parameters
    ----------
    depth : np.ndarray
            1d vector or 2d array of bathymetric values
    column_indices : np.array
                    column indices/location of the sampling lines
    multiple : int
            Multiple of the linespacing used to define the window size

    Returns
    --------
    strip : np.array
            a segment of the bathymetric data for further FFT processing


    """

    # if depth is a vector, convert to matrix
    if len(depth.shape) < 1:
        depth = np.expand_dims(depth, axis=0)

    current_multiple = multiple
    start, end = column_indices[0], column_indices[1]
    linespacing = end - start - 1
    window_size = linespacing * current_multiple
    midpoint = start + (linespacing // 2) + 1

    # Determine column boundaries for window segment
    # -1 / +1 will include sampling columns at the edges
    start_col = int(midpoint - (window_size // 2)) - 1
    end_col = int(column_indices[-1] + (window_size // 2) + 1)

    # Get sliding window view of data using window_size
    # +2 will compensate for the additional pixels on the edges
    window_views = sliding_window_view(depth[:, start_col:end_col],
                                       window_shape=(depth.shape[0],
                                                     window_size + 2))

    # remove extra dimension and only retain views of the window size
    stride = linespacing + 1
    window_views = window_views.squeeze()[::stride]

    # reshape to 2d matrix
    strips = window_views.reshape(-1, window_size + 2)

    return strips


def strip2matrix(
    data_strip: np.ndarray,
    original_shape: tuple,
    column_indices: np.ndarray,
) -> np.ndarray:
    """
    Reverses the get_strip function

    Parameters
    ----------
    data_strip : np.ndarray
                processed data in strip form
    original_shape : np.array
                    original dimensions of the bathymetric data
    column_indices : np.array
                     column indices/location of the sampling lines

    Returns
    --------
    unstripped : np.array
                values in their original spatial location


    """
    # placeholder for reconstructed matrix
    output = np.zeros(shape=original_shape)
    output[:] = np.nan

    # "Cut" the long strip into vertical segments with length (rows)
    # equal to the original data matrix
    num_rows, num_cols = original_shape[0], data_strip.shape[1]
    window_views = sliding_window_view(data_strip,
                                       window_shape=(num_rows,
                                                     num_cols))

    # remove extra dimension and only retain views of the window size
    stride = num_rows
    segment_strips = window_views.squeeze()[::stride]

    # Start with the first slice of the segment_strips
    strip_0 = segment_strips[0, :, :]

    # Remove the first column of the succeeding slides as as they overlap
    # with the last column of the previous slice
    strip_rest = segment_strips[1:, :, 1:]

    # concatenate succeeding slices in the 2nd dimension
    strip_rest = np.transpose(strip_rest, (0, 2, 1))
    strip_rest = strip_rest.reshape(-1, strip_rest.shape[2]).T

    # concatenate first slice with the rest of the segments
    unstripped = np.concatenate((strip_0, strip_rest), axis=1)
    # print(f"strip_0 shape: {strip_0.shape}")
    # print(f"strip_rest shape: {strip_rest.shape}")
    # print(f"target shape: {output[:, column_indices[0]:column_indices[-1]+1].shape}")

    # Place reconstructed array into proper columns
    output[:, column_indices[0]:column_indices[-1]+1] = unstripped

    return output
