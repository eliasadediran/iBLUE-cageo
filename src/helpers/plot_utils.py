import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from pathlib import Path

# mpl.use("Qt5Agg")
parameters = {'axes.labelsize': 14,
          'axes.titlesize': 14,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
plt.rcParams.update(parameters)

plt.rcParams.update({
    'axes.edgecolor': 'black',
    # 'axes.linewidth': 0.8,
    'xtick.color': 'black',
    'ytick.color': 'black'
})

def plot_figure_scatter(array, percentile, ndv, outdir, coordinate, marker, extent, xsize, ysize, res, seabed, spacing, title, cb_label, vmin = None, vmax = None, cm_name='viridis', scale_colour='white', north_colour='white', utm_zone = "NAD83 UTM 19N, 8m resolution", bound=False, outline=False):
    plt.figure(figsize=(4.5,3), dpi=600, tight_layout=True)
    # plt.grid(False)

    # Ensure output directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)
    output_extent = extent
    ax = plt.axes()
    plt.grid(False)
    ax.grid(False, which='both')
    plt.ticklabel_format(useOffset=False, style='plain')  # to avoid the display of an offset for the labels

    # Convert data to km
    m2km = lambda x, _: f'{x / 1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)


    # Set axis intervals
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3000))

    ax.tick_params(axis='both', which='major', length=2, width=0.5, bottom=True, left=True)

    array_nan = array.copy().astype('float')
    array_nan[array_nan == ndv] = np.nan  # set the ndv to nan so as to dispace the ndv as white color
    masked_array = np.ma.array(array_nan, mask=np.isnan(array_nan))

    # new
    masked_array = masked_array.flatten().reshape(-1,1)

    # Combine the coordinate and array data into a DataFrame
    xy_coordinates = np.hstack((coordinate.easting.values.reshape(-1,1),coordinate.northing.values.reshape(-1,1)))
    df = pd.DataFrame(np.hstack((xy_coordinates, masked_array)), columns=['x', 'y', 'z']).dropna()

    # Sort the DataFrame by the 'z' column
    df_sorted = df.sort_values(by='z', ignore_index=True)

    # Calculate the percentile value
    percentile_value = df_sorted['z'].quantile(percentile / 100)

    # Filter the DataFrame based on the percentile
    idx = df_sorted.loc[df_sorted['z'] >= percentile_value].index
    sliced_df = df_sorted.iloc[0:idx[0]] if not idx.empty else df_sorted

    cmap = plt.get_cmap(cm_name)
    cmap.set_bad('white', 1.)
    if vmin is not None and vmax is not None:
        scatter = plt.scatter(sliced_df.x.values, sliced_df.y.values, c=sliced_df.z.values, cmap=cmap, s=0.2,
                          marker=marker, vmin=vmin, vmax=vmax)
    else:
        scatter = plt.scatter(sliced_df.x.values, sliced_df.y.values, c=sliced_df.z.values, cmap=cmap, s=0.2,
                          marker=marker)
    # plt.title(title, fontsize=20)  # put the filename and the projection as part of the title
    plt.xlabel("Easting (km)")
    plt.ylabel("Northing (km)")
    cb = plt.colorbar(scatter, pad=0.01)  # pass the 'img' object used to create the colorbar
    cb.set_label(cb_label, size=14, labelpad=0.01)

    # Set plot extent
    ulx, lrx, lry, uly = extent
    plt.xlim(ulx, lrx)
    plt.ylim(lry, uly)

    # Add the north arrow
    ax.annotate('N', xy=(0.05, 0.94), xycoords='axes fraction',
                fontsize=8, ha='center', va='center', color=north_colour,
                bbox=dict(boxstyle='round', fc='none', ec=north_colour, color=north_colour))

    # Add the scale bar
    scalebar = ScaleBar(1, "m", length_fraction=0.4, height_fraction=0.01, location="lower center", font_properties={"size": 8}, sep=1)
    
    # Customize the scale bar properties
    scalebar.set_color(scale_colour)
    scalebar.set_box_color(scale_colour)
    scalebar.set_box_alpha(0)

    ax.add_artist(scalebar)

    # Add UTM zone above scalebar
    if utm_zone:
        ax.text(0.5, 0.1, utm_zone,transform=ax.transAxes, ha='center', va='bottom',fontsize=8, color=scale_colour,
                bbox=dict(boxstyle='round', fc='none', ec=scale_colour))


    plt.savefig(os.path.join(outdir, f"{seabed}_{title}_{spacing}m_scatter.png"), bbox_inches='tight', dpi=600)
    plt.interactive(False)
    plt.show()
    # plt.close()


def plot_figure(array, extent, ndv, outdir, xsize, ysize, res, seabed, spacing, title, cb_label, vmin = None, vmax = None, cm_name='viridis', scale_colour='white', utm_zone = "NAD83 UTM 19N, 8m resolution", bound=False, outline=False):

    plt.figure(figsize=(4.5,3), dpi=600, tight_layout=True)

    # Ensure output directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)
    output_extent = extent
    ax = plt.axes()
    plt.grid(False)
    ax.grid(False, which='both')
    plt.ticklabel_format(useOffset=False, style='plain')  # to avoid the display of an offset for the labels

    # Convert data to km
    m2km = lambda x, _: f'{x / 1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)

    # Set axis intervals
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3000))


    ax.tick_params(axis='both', which='major', length=2, width=0.5, bottom=True, left=True)

    array_nan = array.copy().astype('float')
    array_nan[array_nan == ndv] = np.nan  # set the ndv to nan so as to dispace the ndv as white color
    masked_array = np.ma.array(array_nan, mask=np.isnan(array_nan))


    cmap = plt.get_cmap(cm_name)
    cmap.set_bad('white', 1.)
    if vmin is not None and vmax is not None:
        img = plt.imshow(masked_array, origin='upper', extent=output_extent, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        img = plt.imshow(masked_array, origin='upper', extent=output_extent, cmap=cmap)
    # plt.title(title, fontsize=20)  # put the filename and the projection as part of the title
    plt.xlabel("Easting (km)")
    plt.ylabel("Northing (km)")
    cb = plt.colorbar(img, pad=0.01)  # pass the 'img' object used to create the colorbar
    cb.set_label(cb_label, size=8, labelpad=0.01)

    # Add the north arrow
    ax.annotate('N', xy=(0.05, 0.94), xycoords='axes fraction',
                fontsize=8, ha='center', va='center', color=scale_colour,
                bbox=dict(boxstyle='round', fc='none', ec=scale_colour, color=scale_colour))

    # Add the scale bar

    scalebar = ScaleBar(1, "m", length_fraction=0.4, height_fraction=0.01, location="lower center", font_properties={"size": 8}, sep=1)
    # Customize the scale bar properties
    scalebar.set_color(scale_colour)
    scalebar.set_box_color(scale_colour)
    scalebar.set_box_alpha(0)

    ax.add_artist(scalebar)

    # Add UTM zone above scalebar
    if utm_zone:
        ax.text(0.5, 0.1, utm_zone,transform=ax.transAxes, ha='center', va='bottom',fontsize=8, color=scale_colour,
                bbox=dict(boxstyle='round', fc='none', ec=scale_colour))


    plt.savefig(os.path.join(outdir, f"{seabed}_{title}_{spacing}m.png"), bbox_inches='tight', dpi=600)
    plt.interactive(False)
    plt.show()
    # plt.close()

def plot_figure_mask(array, ndv, outdir, extent, xsize, ysize, res, seabed, spacing, title, cb_label, vmin = None, vmax = None, cm_name='viridis', scale_colour='white', utm_zone = "NAD83 UTM 19N, 8m resolution", bound=False, outline=False):
    plt.figure(figsize=(4.5,3), dpi=600, tight_layout=True)
    
    # Ensure output directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    output_extent = extent
    ax = plt.axes()
    plt.grid(False)
    ax.grid(False, which='both')
    plt.ticklabel_format(useOffset=False, style='plain')  # to avoid the display of an offset for the labels

    # Convert data to km
    m2km = lambda x, _: f'{x / 1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)

    # Set axis intervals
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3000))

    ax.tick_params(axis='both', which='major', length=2, width=0.5, bottom=True, left=True)

    array_nan = array.copy().astype('float')
    array_nan[array_nan == ndv] = np.nan  # set the ndv to nan so as to dispace the ndv as white color
    masked_array = np.ma.array(array_nan, mask=np.isnan(array_nan))

    cmap = ListedColormap(['lightgray', 'crimson'])  # 0=No, 1=Yes
    bnorm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    if vmin is not None and vmax is not None:
        img = plt.imshow(masked_array, origin='upper', extent=output_extent, cmap=cmap, norm=bnorm, vmin=vmin, vmax=vmax)
    else:
        img = plt.imshow(masked_array, origin='upper', extent=output_extent, norm=bnorm, cmap=cmap)
    # plt.title(title, fontsize=20)  # put the filename and the projection as part of the title
    plt.xlabel("Easting (km)")
    plt.ylabel("Northing (km)")
    cb = plt.colorbar(img, ticks=[0, 1], pad=0.01)  # pass the 'img' object used to create the colorbar
    cb.set_label(f'|z| > {cb_label}', size=14, labelpad=0.01)
    cb.set_ticklabels(['No', 'Yes'])

    # Add the north arrow
    ax.annotate('N', xy=(0.05, 0.94), xycoords='axes fraction',
                fontsize=8, ha='center', va='center', color=scale_colour,
                bbox=dict(boxstyle='round', fc='none', ec=scale_colour, color=scale_colour))

    # Add the scale bar
    scalebar = ScaleBar(1, "m", length_fraction=0.4, height_fraction=0.01, location="lower center", font_properties={"size": 8}, sep=1)

    # Customize the scale bar properties
    scalebar.set_color(scale_colour)
    scalebar.set_box_color(scale_colour)
    scalebar.set_box_alpha(0)

    ax.add_artist(scalebar)

    # Add UTM zone above scalebar
    if utm_zone:
        ax.text(0.5, 0.1, utm_zone,transform=ax.transAxes, ha='center', va='bottom',fontsize=8, color=scale_colour,
                bbox=dict(boxstyle='round', fc='none', ec=scale_colour))

    plt.savefig(os.path.join(outdir, f"{seabed}_{spacing}m_{title}_calibration_spatial_plot.png"), bbox_inches='tight', dpi=600)
    plt.interactive(False)
    plt.show()
    # plt.close()

def plot_cross_sections(depth,residuals,PSD_uncertainty, SSR_uncertainty, spectral_method, statistical_method, rows_keep, select_rows, column_indices, desired_linespacing_meters, resolution, outdir, seabed, normalize_residual=True, plot_location=True):
    """
    Plot uncertainty cross-sections and optionally plot their locations.

    Parameters
    ----------
    depth : ndarray
        Bathymetric grid.
    residuals : ndarray
        Residual grid.
    PSD_uncertainty : ndarray
        Spectral uncertainty estimates.
    SSR_uncertainty : ndarray
        Statistical uncertainty estimates.
    """

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
    })

    # Ensure output directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Determine section boundaries
    if desired_linespacing_meters < 50:
        n_segments = 5
        indices = np.linspace(0, len(column_indices) - 1, n_segments + 1, dtype=int)[1:]
        x_lim_rights = column_indices[indices]

    elif desired_linespacing_meters < 100:
        n_segments = 4
        indices = np.linspace(0, len(column_indices) - 1, n_segments + 1, dtype=int)[1:]
        x_lim_rights = column_indices[indices]

    elif desired_linespacing_meters < 200:
        n_segments = 3
        indices = np.linspace(0, len(column_indices) - 1, n_segments + 1, dtype=int)[1:]
        x_lim_rights = column_indices[indices]

    elif desired_linespacing_meters < 400:
        x_lim_rights = [column_indices[len(column_indices) // 2],column_indices[-1]]

    else:
        x_lim_rights = [column_indices[-1]]

    for select_row in select_rows:

        # --------------------------------------------------------------
        # Optional location plot
        # --------------------------------------------------------------
        if plot_location:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.grid(False)

            im = ax.imshow(depth, cmap="viridis", aspect="equal")

            cbar = fig.colorbar(im)
            cbar.set_label("Depth (m)", fontsize=16)
            cbar.ax.tick_params(labelsize=14)

            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x * resolution)) for x in xticks])

            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(int(y * resolution)) for y in yticks])

            ax.tick_params(axis="x", labelsize=16)
            ax.tick_params(axis="y", labelsize=16)

            ax.set_xlabel("West-East (m)", fontsize=18)
            ax.set_ylabel("North-South (m)", fontsize=18)

            ax.tick_params(axis="both",which="major",length=3,width=1,direction="out")

            ax.hlines(y=select_row, xmin=0, xmax=depth.shape[1], color="red")

            ax.set_xlim(0, depth.shape[1])
            ax.set_ylim(depth.shape[0], 0)

            ax.set_title(f"Row {int(select_row * resolution)}")

            plt.savefig(os.path.join(outdir, f"{seabed}_{desired_linespacing_meters}m_row_{int(select_row * resolution)}_spatial_plot.png"), dpi=300, bbox_inches="tight")
            # plt.close()

        # --------------------------------------------------------------
        # Cross-section plots
        # --------------------------------------------------------------
        row_residual = np.abs(residuals[rows_keep, :][select_row])
        row_psd = PSD_uncertainty[select_row]
        row_ssr = SSR_uncertainty[select_row]

        max_value = np.nanmax([np.nanmax(row_residual),np.nanmax(row_psd),np.nanmax(row_ssr)])

        x_lim_left = 0

        for i, x_lim_right in enumerate(x_lim_rights):

            fig, ax = plt.subplots(figsize=(15, 5),constrained_layout=True)

            ax.plot(row_residual, lw=2, label="Residuals")
            ax.plot(row_psd, lw=2, label=spectral_method)
            ax.plot(row_ssr, lw=2, label=statistical_method)

            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(x * resolution)) for x in xticks])

            ax.set_xlim(x_lim_left, x_lim_right)
            ax.set_ylim(0, max_value)
            ax.set_xlabel("West-East (m)")
            ax.set_ylabel( "Uncertainty (% of depth)" if normalize_residual else "Uncertainty (m)")
            ax.set_title(f"Residual vs Estimated Uncertainty (Section {i + 1})")
            ax.legend()
            plt.savefig(os.path.join(outdir, f"{seabed}_{desired_linespacing_meters}m_row_{int(select_row * resolution)}_cross_section_{i + 1}.png"), bbox_inches="tight")
            # plt.close()
            x_lim_left = x_lim_right