from typing import Union, Any, Dict, Optional, Callable
from pathlib import Path
import numpy as np
import functools

from osgeo import gdal
gdal.UseExceptions()

def load_file(
        filename: Union[str, Path],
        folder: Optional[str] = None,
        verbose: bool = True) -> Dict[str, Any]:

    """
    Reads bathymetry data from a TIFF file.

    This function uses the GDAL library to read a TIFF file
    containing bathymetric data. It extracts the primary data matrix,
    the "no-data" value (NDV), and the spatial resolution of the image.

    Parameters
    ----------
    filename : str, Path
        The base filename or Path of the TIFF file (e.g., "bathymetry.tif").
    folder : str, optional
        An optional path to the directory containing the TIFF file.
    verbose : bool
              Optional flag to print some information on the bathymetry file

    Returns
    -------
    dict
        A dictionary containing the parsed bathymetric data:
        * 'depth' (numpy.ndarray of float): A 2D NumPy array containing the
            bathymetric depth values in meters. Pixels with no data are
            represented by the 'ndv' value.
        * 'ndv' (float or int or None): The value representing "no-data" pixels
            in the `depth` array. This can be `None`, `np.nan`, or a specific
            numeric value as defined in the TIFF file's metadata.
        * 'resolution' (float): The spatial resolution of the bathymetric data
            in meters per pixel (derived from the GeoTransform).
        * 'dimensions' (tuple) : Dimensions of the 2D NumPy array


    Raises
    ------
    TypeError
        If `filename` is not a string or Path.
        (Changed from ValueError for type-checking)
    FileNotFoundError
        If the specified TIFF file does not exist at the given path.
    RuntimeError
        If GDAL fails to open the file (e.g., corrupt file, unsupported format)
        or if the primary raster band cannot be retrieved.
    ValueError
        If spatial resolution is less than or equal to zero

    See Also
    --------
    osgeo.gdal.Open : GDAL's function for opening raster datasets.
    os.path.join : For robust path construction across operating systems.

    Notes
    -----
    This function relies on the `osgeo.gdal` library, which must be installed
    and correctly configured in your environment.

    Examples
    --------
    >>> # Assuming 'data/my_bathymetry.tif' exists
    >>> import numpy as np
    >>> # Example 1: File in current directory
    >>> data = load_file("my_bathymetry.tif")
    >>> print(data['resolution'])
    1.0
    >>> # Example 2: File in a specific folder
    >>> data_folder = "data"
    >>> data = load_file("another_map.tif", folder=data_folder)
    >>> print(data['depth'].shape)
    (100, 200)
    >>> # Example 3: Handling no-data values
    >>> ndv_value = data['ndv']
    >>> if ndv_value is not None:
    >>>     print(f"No-data value: {ndv_value}")
    >>>     # Count no-data pixels
    >>>     print(f"No-data pixels: {np.sum(data['depth'] == ndv_value)}")

    """

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

    with (gdal.Open(str(file_path)) as ds):
        if not ds:
            raise RuntimeError(
                f"GDAL failed to open TIFF file: '{file_path}'")

        band_count = ds.RasterCount

        if band_count < 1:
            raise RuntimeError(
                f"No raster bands found in {file_path}.")

        bands = []
        ndvs = []

        for i in range(1, band_count + 1):
            band = ds.GetRasterBand(i)

            if not band:
                raise RuntimeError(
                    f"Error retrieving raster band {i} from {file_path}.")

            bands.append(band.ReadAsArray())
            ndvs.append(band.GetNoDataValue())

        # Preserve existing behavior:
        depth = bands[0]
        ndv = ndvs[0]
        tvu = bands[1]
        survey_id = bands[2]
        # survey_id = rat_to_raster(ds, 3, "source_survey_id")
        depth_gt = ds.GetGeoTransform()
        resolution = depth_gt[1]

        # Get the data extent
        ulx, xres, xskew, uly, yskew, yres = depth_gt

        xsize = ds.RasterXSize
        ysize = ds.RasterYSize

        # Compute coordinates of all four corners
        corners_x = [
            ulx,
            ulx + xsize * xres,
            ulx + ysize * xskew,
            ulx + xsize * xres + ysize * xskew
        ]

        corners_y = [
            uly,
            uly + xsize * yskew,
            uly + ysize * yres,
            uly + xsize * yskew + ysize * yres
        ]

        extent = [
            min(corners_x),
            max(corners_x),
            min(corners_y),
            max(corners_y)
        ]

        if resolution < 1:
            raise ValueError(
                f"Spatial resolution should be at least 1"
                f"Resolution value from file: {resolution}"
            )

    if verbose:
        # Print some statistics of the bathymetry data
        spatial_coverage_length = int(depth.shape[0] * resolution)
        spatial_coverage_width = int(depth.shape[1] * resolution)
        print(
            f"""
    Input filename: {str(file_path)}
    Data dimensions: {depth.shape}
    Min/Max: {np.nanmin(depth), np.nanmax(depth)}
    Survey Coverage: {spatial_coverage_length}m x {spatial_coverage_width}m
    Spatial Resolution: {resolution}
            """
        )

    # Compile bathymetric data in a dictionary and return directly
    return {
        "depth": depth,
        "ndv": ndv,
        "tvu": tvu,
        "survey_id": survey_id,
        "resolution": resolution,
        "dimensions": depth.shape,
        "geotransform": depth_gt,
        "extent": extent,
        "xsize": xsize,
        "ysize": ysize,
    }


def remove_edge_Nans(
    depth: np.ndarray,
    ndv: Union[float, int, None, str],
    max_iterations: Optional[int] = None
) -> np.ndarray:
    """
    Iteratively removes edge rows and columns containing no-data values.

    This function crops the input 2D array by iteratively removing outermost
    rows and columns containing "no-data" value (NDV).
    The process continues until all edges are free of NDV pixels
    or a maximum iteration limit is reached.

    Parameters
    ----------
    depth : numpy.ndarray
        A 2D NumPy array representing surface elevation or bathymetry data.
        Expected to be numeric (e.g., float, int).
    ndv : float or int or None
        The value representing "no-data" pixels within the `depth` array.
        This could be a specific number (e.g., -9999), `np.nan`, or `None`.
    max_iteration : int, optional
        maximum number of iterations to be done
        Default is half the largest dimension

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped data

    Raises
    ------
    ValueError
        If the `depth` parameter is `None` or an empty array
        (e.g., has a zero dimension).
    TypeError
        If `depth` is not a `numpy.ndarray`.

    Examples
    --------
    >>> import numpy as np
    >>> # Example 1: Basic cropping
    >>> data = np.array([
    ...     [99, 99, 99, 99],
    ...     [99,  1,  2, 99],
    ...     [99,  3,  4, 99],
    ...     [99, 99, 99, 99]
    ... ])
    >>> cropped_data = remove_edge_Nans(data, 99)
    >>> print(cropped_data)
    [[1 2]
     [3 4]]

    >>> # Example 2: No cropping needed
    >>> data = np.array([[1, 2], [3, 4]])
    >>> cropped_data = remove_edge_Nans(data, 99)
    >>> print(cropped_data)
    [[1 2]
     [3 4]]

    >>> # Example 3: Different NDV (e.g., np.nan)
    >>> data_nan = np.array([
    ...     [np.nan, np.nan, 1.0, np.nan],
    ...     [np.nan, 2.0, 3.0, np.nan],
    ...     [np.nan, np.nan, np.nan, np.nan]
    ... ])
    >>> cropped_data_nan = remove_edge_Nans(data_nan, np.nan)
    >>> print(cropped_data_nan)
    [[1. 2.]
     [3. 4.]]

    """

    # Type check for depth
    if not isinstance(depth, np.ndarray):
        raise TypeError("Input 'depth' must be a NumPy array (np.ndarray).")

    # Handle initial empty array or None input
    if depth is None or depth.size == 0:
        raise ValueError("Input 'depth' array cannot be None or empty.")

    # Create a working copy to avoid modifying the original array passed in
    elev = depth.copy()
    original_shape = depth.shape

    # Set up value for max_iteration if none declared
    if max_iterations is None:
        max_dimensions = np.max(original_shape)
        max_iterations = int(np.max(max_dimensions) / 2)

    if ndv == np.nan:
        def is_ndv(data_array):
            return np.any(np.isnan(data_array))
    else:

        def is_ndv(data_array):
            return np.any(data_array == ndv)

    shrink_idx = 0
    have_ndv = True
    # remove edges that have NaN elements
    # continue until all edges are NaN free or exceeded 100 iterations
    # assumes that all inner elements are non NaN
    while have_ndv:
        tmp = elev[0, :]
        if is_ndv(tmp):
            elev = elev[1:, :]
        tmp = elev[:, 0]
        if is_ndv(tmp):
            elev = elev[:, 1:]
        tmp = elev[-1, :]
        if is_ndv(tmp):
            elev = elev[:-1, :]
        tmp = elev[:, -1]
        if is_ndv(tmp):
            elev = elev[:, :-1]
        shrink_idx += 1
        if not np.any(is_ndv(elev)):
            have_ndv = False
        if shrink_idx > max_iterations:
            break


    return elev


def remove_edge_Nans_bands(
        depth: np.ndarray,
        ndv: Union[float, int, None, str],
        max_iterations: Optional[int] = None):
    """
    Iteratively removes edge rows and columns containing no-data values.

    This function crops the input 2D array by iteratively removing outermost
    rows and columns containing "no-data" value (NDV).
    The process continues until all edges are free of NDV pixels
    or a maximum iteration limit is reached.

    Parameters
    ----------
    depth : numpy.ndarray
        A 2D NumPy array representing surface elevation or bathymetry data.
        Expected to be numeric (e.g., float, int).
    ndv : float or int or None
        The value representing "no-data" pixels within the `depth` array.
        This could be a specific number (e.g., -9999), `np.nan`, or `None`.
    max_iteration : int, optional
        maximum number of iterations to be done
        Default is half the largest dimension

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the cropped data

    Raises
    ------
    ValueError
        If the `depth` parameter is `None` or an empty array
        (e.g., has a zero dimension).
    TypeError
        If `depth` is not a `numpy.ndarray`.

    Examples
    --------
    >>> import numpy as np
    >>> # Example 1: Basic cropping
    >>> data = np.array([
    ...     [99, 99, 99, 99],
    ...     [99,  1,  2, 99],
    ...     [99,  3,  4, 99],
    ...     [99, 99, 99, 99]
    ... ])
    >>> cropped_data = remove_edge_Nans(data, 99)
    >>> print(cropped_data)
    [[1 2]
     [3 4]]

    >>> # Example 2: No cropping needed
    >>> data = np.array([[1, 2], [3, 4]])
    >>> cropped_data = remove_edge_Nans(data, 99)
    >>> print(cropped_data)
    [[1 2]
     [3 4]]

    >>> # Example 3: Different NDV (e.g., np.nan)
    >>> data_nan = np.array([
    ...     [np.nan, np.nan, 1.0, np.nan],
    ...     [np.nan, 2.0, 3.0, np.nan],
    ...     [np.nan, np.nan, np.nan, np.nan]
    ... ])
    >>> cropped_data_nan = remove_edge_Nans(data_nan, np.nan)
    >>> print(cropped_data_nan)
    [[1. 2.]
     [3. 4.]]

    """

    # Type check for depth
    if not isinstance(depth, np.ndarray):
        raise TypeError("Input 'depth' must be a NumPy array (np.ndarray).")

    # Handle initial empty array or None input
    if depth is None or depth.size == 0:
        raise ValueError("Input 'depth' array cannot be None or empty.")

    # Create a working copy to avoid modifying the original array passed in
    elev = depth.copy()
    original_shape = depth.shape
    top = 0
    bottom = original_shape[0]
    left = 0
    right = original_shape[1]

    # Set up value for max_iteration if none declared
    if max_iterations is None:
        max_dimensions = np.max(original_shape)
        max_iterations = int(np.max(max_dimensions) / 2)

    if ndv is None or np.isnan(ndv):
        def is_ndv(data_array):
            return np.any(np.isnan(data_array))
    else:
        def is_ndv(data_array):
            return np.any(data_array == ndv)

    shrink_idx = 0
    have_ndv = True
    # remove edges that have NaN elements
    # continue until all edges are NaN free or exceeded 100 iterations
    # assumes that all inner elements are non NaN
    while have_ndv:
        tmp = elev[0, :]
        if is_ndv(tmp):
            elev = elev[1:, :]
            top += 1
        tmp = elev[:, 0]
        if is_ndv(tmp):
            elev = elev[:, 1:]
            left += 1
        tmp = elev[-1, :]
        if is_ndv(tmp):
            elev = elev[:-1, :]
            bottom -= 1
        tmp = elev[:, -1]
        if is_ndv(tmp):
            elev = elev[:, :-1]
            right -= 1
        shrink_idx += 1
        if not np.any(is_ndv(elev)):
            have_ndv = False
        if shrink_idx > max_iterations:
            break

    rs = slice(top, bottom)  # row slice
    cs = slice(left, right)  # col_slice

    return elev, rs, cs
