import numpy as np

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