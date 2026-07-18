import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

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
    # plt.figure(figsize=(15, 10))
    plt.figure(figsize=(4.5,3), dpi=600, tight_layout=True)
    # plt.grid(False)
    output_extent = extent
    ax = plt.axes()
    plt.grid(False)
    ax.grid(False, which='both')
    plt.ticklabel_format(useOffset=False, style='plain')  # to avoid the display of an offset for the labels

    # Convert data to km
    m2km = lambda x, _: f'{x / 1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(round(xsize * res / 3 / 1000) * 1000))  # to set the distance between labels (x axis) and round to nearest thousand
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(round(ysize * res / 3 / 1000) * 1000))

    # Set axis intervals
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3000))

    # Set plot extent
    #extent = [ulx, lrx, lry, uly]
    # plt.xlim(extent[0], extent[1])
    # plt.ylim(extent[2], extent[3])

    ax.tick_params(axis='both', which='major', length=2, width=0.5, bottom=True, left=True)

    array_nan = array.copy().astype('float')
    # if np.any(array_nan == 0):
    #     array_nan[array_nan == 0] = np.nan  # set the ndv to nan so as to dispace the ndv as white color
    # else:
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

    # import cmocean
    # cmap = cmocean.cm.deep
    cmap = plt.get_cmap(cm_name)
    # cmap = mpl.cm.jet
    # # cmap = mpl.cm.hsv # for roughness
    # # cmap = mpl.cm.rainbow # for slope
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
    # add_north_arrow(ax)

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


    if outline:
        from matplotlib.patches import Rectangle
        fig = plt.gcf()
        # fig.canvas.draw()
        axes = fig.get_axes()

        bboxes = [ax.get_position() for ax in axes]

        x0 = min(b.x0 for b in bboxes)
        y0 = min(b.y0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)

        # independent padding (tune these)
        # pad_left   = 0.12 #0.055
        # pad_right  = 0.06 #0.017  # usually larger because of colorbar
        # pad_bottom = 0.08 #0.005
        # pad_top    = 0.08 #0.015

        pad_left   = 0.055
        pad_right  = 0.017  # usually larger because of colorbar
        pad_bottom = 0.005
        pad_top    = 0.017

        fig.add_artist(
            Rectangle(
                (x0 - pad_left, y0 - pad_bottom),
                (x1 - x0) + pad_left + pad_right,
                (y1 - y0) + pad_bottom + pad_top,
                transform=fig.transFigure,
                fill=False,
                edgecolor='black',
                linewidth=1.2
            )
        )


    plt.savefig(f"{outdir}{seabed}_{title}_{spacing}m_scatter.png", bbox_inches='tight', dpi=600)
    plt.interactive(False)
    plt.show()
    # plt.close()


# # mpl.use("Qt5Agg")
# parameters = {'axes.labelsize': 14,
#           'axes.titlesize': 14,
#           'xtick.labelsize': 12,
#           'ytick.labelsize': 12}
# plt.rcParams.update(parameters)
#
# plt.rcParams.update({
#     'axes.edgecolor': 'black',
#     # 'axes.linewidth': 0.8,
#     'xtick.color': 'black',
#     'ytick.color': 'black'
# })

def plot_figure(array, extent, ndv, outdir, xsize, ysize, res, seabed, spacing, title, cb_label, vmin = None, vmax = None, cm_name='viridis', scale_colour='white', utm_zone = "NAD83 UTM 19N, 8m resolution", bound=False, outline=False):
    # plt.figure(figsize=(15, 10))
    plt.figure(figsize=(4.5,3), dpi=600, tight_layout=True)
    # plt.grid(False)
    output_extent = extent
    ax = plt.axes()
    plt.grid(False)
    ax.grid(False, which='both')
    plt.ticklabel_format(useOffset=False, style='plain')  # to avoid the display of an offset for the labels

    # Convert data to km
    m2km = lambda x, _: f'{x / 1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(round(xsize * res / 3 / 1000) * 1000))  # to set the distance between labels (x axis) and round to nearest thousand
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(round(ysize * res / 3 / 1000) * 1000))

    # Set axis intervals
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3000))

    # Set plot extent
    #extent = [ulx, lrx, lry, uly]
    # plt.xlim(extent[0], extent[1])
    # plt.ylim(extent[2], extent[3])

    ax.tick_params(axis='both', which='major', length=2, width=0.5, bottom=True, left=True)

    array_nan = array.copy().astype('float')
    # if np.any(array_nan == 0):
    #     array_nan[array_nan == 0] = np.nan  # set the ndv to nan so as to dispace the ndv as white color
    # else:
    array_nan[array_nan == ndv] = np.nan  # set the ndv to nan so as to dispace the ndv as white color
    masked_array = np.ma.array(array_nan, mask=np.isnan(array_nan))


    cmap = plt.get_cmap(cm_name)
    # cmap = mpl.cm.jet
    # # cmap = mpl.cm.hsv # for roughness
    # # cmap = mpl.cm.rainbow # for slope
    cmap.set_bad('white', 1.)
    if vmin is not None and vmax is not None:
        img = plt.imshow(masked_array, origin='upper', extent=output_extent, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        img = plt.imshow(masked_array, origin='upper', extent=output_extent, cmap=cmap)
    # plt.title(title, fontsize=20)  # put the filename and the projection as part of the title
    plt.xlabel("Easting (km)")
    plt.ylabel("Northing (km)")
    cb = plt.colorbar(img, pad=0.01)  # pass the 'img' object used to create the colorbar
    cb.set_label(cb_label, size=12, labelpad=0.01)
    # add_north_arrow(ax)

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


    if outline:
        from matplotlib.patches import Rectangle
        fig = plt.gcf()
        # fig.canvas.draw()
        axes = fig.get_axes()

        bboxes = [ax.get_position() for ax in axes]

        x0 = min(b.x0 for b in bboxes)
        y0 = min(b.y0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)

        # independent padding (tune these)
        # pad_left   = 0.12 #0.055
        # pad_right  = 0.06 #0.017  # usually larger because of colorbar
        # pad_bottom = 0.08 #0.005
        # pad_top    = 0.08 #0.015

        pad_left   = 0.055
        pad_right  = 0.017  # usually larger because of colorbar
        pad_bottom = 0.005
        pad_top    = 0.017

        fig.add_artist(
            Rectangle(
                (x0 - pad_left, y0 - pad_bottom),
                (x1 - x0) + pad_left + pad_right,
                (y1 - y0) + pad_bottom + pad_top,
                transform=fig.transFigure,
                fill=False,
                edgecolor='black',
                linewidth=1.2
            )
        )


    plt.savefig(f"{outdir}{seabed}_{title}_{spacing}m.png", bbox_inches='tight', dpi=600)
    plt.interactive(False)
    plt.show()
    # plt.close()

def plot_figure_mask(array, ndv, outdir, extent, xsize, ysize, res, seabed, spacing, title, cb_label, vmin = None, vmax = None, cm_name='viridis', scale_colour='white', utm_zone = "NAD83 UTM 19N, 8m resolution", bound=False, outline=False):
    plt.figure(figsize=(4.5,3), dpi=600, tight_layout=True)
    output_extent = extent
    ax = plt.axes()
    plt.grid(False)
    ax.grid(False, which='both')
    plt.ticklabel_format(useOffset=False, style='plain')  # to avoid the display of an offset for the labels

    # Convert data to km
    m2km = lambda x, _: f'{x / 1000:g}'
    ax.xaxis.set_major_formatter(m2km)
    ax.yaxis.set_major_formatter(m2km)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(round(xsize * res / 3 / 1000) * 1000))  # to set the distance between labels (x axis) and round to nearest thousand
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(round(ysize * res / 3 / 1000) * 1000))

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
    # cb.set_label(cb_label, size=14, labelpad=0.01)
    cb.set_label(f'|z| > {cb_label}', size=14, labelpad=0.01)
    cb.set_ticklabels(['No', 'Yes'])
    # add_north_arrow(ax)

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

    plt.savefig(f"{outdir}{seabed}_{title}_{spacing}m.png", bbox_inches='tight', dpi=600)
    plt.interactive(False)
    plt.show()
    # plt.close()