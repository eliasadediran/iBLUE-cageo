from osgeo import gdal
import pandas as pd
from pathlib import Path
from typing import Union, Any, Dict, Optional, Callable
import numpy as np

def select_lines(data, Mline_spacings, x_res, ndv=-9999, MlineOnly=True):
    """
    Automates the calculation of main line and cross line spacings, increments,
    and selection of lines for interpolation.

    Parameters:
    - data: 2D array representing the dataset.
    - Mline_spacings: Array of main line spacings (meters).
    - x_res: Resolution in meters.
    - xsize: Size of the data in the x-direction.
    - ndv: No-data value (default is -9999).
    - MlineOnly: Boolean indicating if only main lines should be selected (default is True).

    Returns:
    - train_selection: List of selected lines for training.
    - test_selection: List of selected lines for testing.
    """

    # Step 1: Calculate main line increments and spacings
    # xsize: Size of the data in the x-direction.
    xsize = data.shape[0]

    # Edited to align with Francis
    # Mline_incs = []
    # for Mline_spacing in Mline_spacings:
    #     Mline_incs.append(int(Mline_spacing / x_res))

    Mline_incs = []
    for Mline_spacing in Mline_spacings:
        linespacing_in_pixels = int(Mline_spacing / x_res)
        # if (linespacing_in_pixels % 2) != 0:
        #     linespacing_in_pixels = linespacing_in_pixels - 1 # no need for this too
        # Mline_incs.append(linespacing_in_pixels+1) # need to clarify +1 with Francis; figured it out, no need for +1
        Mline_incs.append(linespacing_in_pixels)  #

    Mline_spacings_updated = []
    for Mline_inc in Mline_incs:
        a = Mline_inc * x_res
        Mline_spacings_updated.append(a)

    # Check if main line spacing is appropriate
    for Mline_spacing in Mline_spacings_updated:
        # if Mline_spacing >= xsize:
        if Mline_spacing >= int(xsize*x_res):
            raise RuntimeError(
                "Main line spacing specified is greater than the data size in x axis, data length in x direction is %s" % int(
                    xsize * x_res) + "m")
        else:
            print("Main line spacing " + str(Mline_spacing) + "m specified is appropriate")

    # Step 2: Calculate cross-line increments and spacings
    Mline_idxs = []
    Xline_idxs = []
    Xline_incs = []
    Xline_spacings = []
    xmax = data.shape[1]
    ymax = data.shape[0]

    for Mline_inc in Mline_incs:
        Mlines_idx = np.arange(0, xmax, Mline_inc)
        Mline_idxs.append(Mlines_idx)
        Mlines_selected = data[:, Mlines_idx]
        Mlines_total = Mlines_selected.size
        Xlines_total = int(Mlines_total * 0.09)  # 9% of total Mlines for Xlines
        num = int(np.floor(Xlines_total / xmax))
        # Avoid zero division error
        if num == 0:
            num = 1
        Xline_inc = int(ymax / num)
        Xline_incs.append(Xline_inc)
        Xlines_idx = np.arange(0, ymax, Xline_inc)
        Xline_idxs.append(Xlines_idx)
        Xline_spacings.append(int(Xline_inc * x_res))

    # Step 3: Line selection process
    def line_select(array, Mline_inc, Xline_inc, MlineOnly=True, ndv=-9999):
        xsize, ysize = array.shape

        if xsize == ysize:
            mask = np.zeros((array.shape), dtype=bool)
            for x in range(array.shape[1]):
                for y in range(array.shape[0]):
                    if x % Mline_inc == 0 or (y % Xline_inc == 0 and not MlineOnly):
                        mask[x, y] = True
            mask = mask.T
            train_sample = np.ma.array(array, mask=~mask, fill_value=ndv).filled()
            test_sample = np.ma.array(array, mask=mask, fill_value=ndv).filled()

        elif xsize > ysize or xsize < ysize:
            mask_x = np.zeros([max(array.shape), max(array.shape)], dtype=bool)
            for x in range(max(array.shape)):
                if x % Mline_inc == 0:
                    mask_x[x] = True
            mask_x = mask_x.T

            mask_y = np.zeros([max(array.shape), max(array.shape)], dtype=bool)
            for y in range(max(array.shape)):
                if y % Xline_inc == 0 and not MlineOnly:
                    mask_y[y] = True

            mask = mask_x + mask_y
            mask = mask[:, :min(array.shape)] if xsize > ysize else mask[:min(array.shape), :]

            train_sample = np.ma.array(array, mask=~mask, fill_value=ndv).filled()
            test_sample = np.ma.array(array, mask=mask, fill_value=ndv).filled()

        train_sample[-1, :] = array[-1, :]
        train_sample[:, -1] = array[:, -1]
        test_sample[-1, :] = ndv
        test_sample[:, -1] = ndv

        if MlineOnly:
            for i in range(train_sample.shape[1]):
                if any(train_sample[:, i] == ndv):
                    train_sample[:, i] = ndv
            for i in range(test_sample.shape[1]):
                if all(test_sample[:, i] == ndv):
                    test_sample[:, i] = ndv
                else:
                    test_sample[:, i] = array[:, i]

        train_sample[:, -1] = array[:, -1]
        test_sample[:, -1] = ndv

        return train_sample, test_sample

    # Step 4: Automate the selection of lines with different spacings
    train_selection = []
    test_selection = []
    line_increments = []
    Xline_increments = []
    inc = 0
    keep = True

    while keep:
        for i, j in enumerate(Mline_incs):
            for k, l in enumerate(Xline_incs):
                if i != k:
                    continue
                else:
                    print('Mline increment:'+str(j), 'Xline increment:'+str(l))
                    line_increments.append(j)
                    Xline_increments.append(l)
                    train_selection.append(line_select(data, j, l, MlineOnly, ndv)[0])
                    test_selection.append(line_select(data, j, l, MlineOnly, ndv)[1])
                    inc += 1
                    if inc >= len(Mline_incs):
                        keep = False

    return train_selection, test_selection, line_increments, Mline_idxs, Xline_increments

def fill_across_track(data, across_idx, method='both', overlap_bias='mean'):

    Z = data
    Z_filled = Z.copy()
    N_along, N_across = Z.shape

    # ============================================================
    # BOTH-SIDED FILL
    # ============================================================
    if method == 'both':

        # ---------- Interior gaps ----------
        for k in range(len(across_idx) - 1):

            col_left = across_idx[k]
            col_right = across_idx[k + 1]

            gap_width = col_right - col_left - 1
            if gap_width <= 0:
                continue

            # ---- FIX ONLY HERE ----
            half_gap = gap_width // 2
            is_odd = (gap_width % 2 == 1)

            if is_odd:
                row_radius = (gap_width - 1) // 4
            else:
                row_radius = gap_width // 4

            # ensure we generate at least gap_width values
            if is_odd:
                test_len = 2 * (2 * row_radius + 1) + 1
            else:
                test_len = 4 * row_radius

            if test_len < gap_width:
                row_radius += 1
            # -----------------------

            row_start = row_radius
            row_end = N_along - row_radius - 1

            for row in range(row_start, row_end):

                if is_odd:
                    row_window = np.arange(
                        row - row_radius,
                        row + row_radius + 1
                    )
                else:
                    row_window = np.arange(
                        row - row_radius,
                        row + row_radius
                    )

                left_vals = Z[row_window, col_left]
                right_vals = Z[row_window, col_right][::-1]

                if is_odd:

                    center_row = row

                    if overlap_bias == "left":
                        center_val = Z[center_row, col_left]

                    elif overlap_bias == "right":
                        center_val = Z[center_row, col_right]

                    elif overlap_bias == "mean":
                        center_val = 0.5 * (
                                Z[center_row, col_left] +
                                Z[center_row, col_right]
                        )
                    else:
                        raise ValueError(
                            "overlap_bias must be 'left', 'right', or 'mean'"
                        )

                    fill_values = np.concatenate(
                        (left_vals, [center_val], right_vals)
                    )

                else:
                    fill_values = np.concatenate(
                        (left_vals, right_vals)
                    )

                # Safety check
                # Adjust to exact gap size
                if len(fill_values) > gap_width:
                    extra = len(fill_values) - gap_width
                    left_trim = extra // 2
                    right_trim = extra - left_trim

                    fill_values = fill_values[left_trim:len(fill_values) - right_trim]

                elif len(fill_values) < gap_width:
                    raise ValueError(
                        f"Length mismatch: fill={len(fill_values)}, gap={gap_width}"
                    )

                # Fill the gap
                Z_filled[row, col_left + 1: col_right] = fill_values

        # ---------- Right boundary gap ----------
        col_src = across_idx[-1]
        gap_width = (N_across - 1) - col_src

        if gap_width > 0:
            half_gap = gap_width // 2

            for row in range(half_gap, N_along - half_gap):
                Z_filled[row, col_src + 1 :] = \
                    Z[row - half_gap : row - half_gap + gap_width, col_src]

        return Z_filled

    # ============================================================
    # RIGHT-ONLY FILL
    # ============================================================
    elif method == 'right':

        n_sparse = len(across_idx)

        # ---------- Interior gaps ----------
        for k in range(n_sparse - 1):

            col_src  = across_idx[k]
            col_next = across_idx[k + 1]

            gap_width = col_next - col_src - 1
            if gap_width <= 0:
                continue

            half_gap = gap_width // 2

            for row in range(half_gap, N_along - half_gap):
                Z_filled[row, col_src + 1 : col_next] = \
                    Z[row - half_gap : row - half_gap + gap_width, col_src]

        # ---------- Right boundary gap ----------
        col_src = across_idx[-1]
        gap_width = (N_across - 1) - col_src

        if gap_width > 0:
            half_gap = gap_width // 2

            for row in range(half_gap, N_along - half_gap):
                Z_filled[row, col_src + 1 :] = \
                    Z[row - half_gap : row - half_gap + gap_width, col_src]

        return Z_filled

    # ============================================================
    # LEFT-ONLY FILL
    # ============================================================
    elif method == 'left':

        n_sparse = len(across_idx)

        # ---------- Interior gaps ----------
        for k in range(1, n_sparse):

            col_prev = across_idx[k - 1]
            col_src  = across_idx[k]

            gap_width = col_src - col_prev - 1
            if gap_width <= 0:
                continue

            half_gap = gap_width // 2

            for row in range(half_gap, N_along - half_gap):
                Z_filled[row, col_prev + 1 : col_src] = \
                    Z[row - half_gap : row - half_gap + gap_width, col_src][::-1]

        # ---------- right boundary gap ---------- # need to correct this
        col_src = across_idx[-1]
        gap_width = (N_across - 1) - col_src

        if gap_width > 0:
            half_gap = gap_width // 2

            for row in range(half_gap, N_along - half_gap):
                Z_filled[row, col_src + 1 :] = \
                    Z[row - half_gap : row - half_gap + gap_width, col_src]

        return Z_filled

    else:
        raise ValueError("method must be 'left', 'right', or 'both'")

def clip_rows_with_ndvs_indices(Z, ndv):
    """
    Determine row indices to keep after:
    1) Removing leading NaN rows (top trim)
    2) Truncating at first NaN row after valid data (bottom cascade)

    Returns
    -------
    keep_rows : ndarray
        Indices into the original Z to keep
    """

    if not np.isnan(ndv):
        Z = Z.copy()
        Z[Z==ndv] = np.nan

    # Row is valid ONLY if it contains no NaNs
    row_fully_valid = ~np.isnan(Z).any(axis=1)

    # If no valid rows exist, return empty index array
    if not np.any(row_fully_valid):
        return np.array([], dtype=int)

    # ---- Top trim ----
    first_valid = np.argmax(row_fully_valid)

    # Work only on rows after top trim
    valid_after = row_fully_valid[first_valid:]

    # ---- Bottom cascade ----
    invalid_after = np.where(~valid_after)[0]

    if invalid_after.size > 0:
        last_keep = first_valid + invalid_after[0]
    else:
        last_keep = Z.shape[0]

    # Indices of rows to keep in ORIGINAL array
    keep_rows = np.arange(first_valid, last_keep)

    return keep_rows


def extract_pixel_coordinates(filename: Union[str, Path],
        folder: Optional[str] = None):
    if not isinstance(filename, (str, Path)):
        # Using TypeError for incorrect type, more conventional than ValueError
        raise TypeError("Filename must be a String or Path.")

    if folder:
        file_path = Path(folder) / filename
    else:
        file_path = Path(filename)

    # Check if the file exists before trying to open with GDAL
    if not file_path.exists():
        raise FileNotFoundError(f"TIFF file not found at: {file_path}")

    # Open the raster file
    dataset = gdal.Open(file_path)

    # Get the geotransformation matrix
    geotransform = dataset.GetGeoTransform()

    # Get the number of rows and columns
    num_rows = dataset.RasterYSize
    num_cols = dataset.RasterXSize

    # Create lists to store the coordinates
    eastings = []
    northings = []

    # Iterate through each pixel
    for row in range(num_rows):
        for col in range(num_cols):
            # Convert pixel coordinates to northing and easting
            x = geotransform[0] + (col + 0.5) * geotransform[1] + (row + 0.5) * geotransform[2]
            y = geotransform[3] + (col + 0.5) * geotransform[4] + (row + 0.5) * geotransform[5]

            # Append the coordinates to the respective lists
            eastings.append(x)
            northings.append(y)

    # Create a DataFrame with the coordinates
    df = pd.DataFrame({'easting': eastings, 'northing': northings})

    # Close the dataset
    dataset = None

    # Return the DataFrame
    return df