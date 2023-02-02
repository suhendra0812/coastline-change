import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString
from skimage import filters, measure, morphology

logger = logging.getLogger(__name__)


def db_scale(img: np.ndarray) -> np.ndarray:
    db_output = 10 * np.log10(img, where=img > 0)

    return db_output


def lee_filter(img: np.ndarray, size: float) -> np.ndarray:
    img_mean = ndimage.uniform_filter(img, size)
    img_sqr_mean = ndimage.uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = ndimage.variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output


def segmentation(img: np.ndarray, threshold: float = None) -> np.ndarray:
    img = np.where(~np.isnan(img), img, 0)
    # jika ada nilai NaN di dalam piksel citra, maka diisi angka 0. karena akan error pada perhitungan threshold menggunakan otsu
    # jika nilai threshold tidak disediakan, maka dilakukan perhitungan nilai threshold otomatis menggunakan Otsu
    if not threshold:
        threshold = filters.threshold_otsu(img)
    logger.info(f"Threshold: {threshold}")

    # proses thresholding citra berdasarkan nilai threshold, kemudian diubah ke dalam angka 0 dan 1
    binary = (img > threshold).astype(np.uint8)
    # pengisian lubang-lubang di dalam area piksel 1 (putih)
    binary = ndimage.binary_fill_holes(binary)
    # penghapusan objek-objek kecil (piksel 1/putih)
    binary = morphology.remove_small_objects(binary).astype(int)
    # untuk smoothing area garis pantai
    binary = morphology.closing(binary, morphology.disk(5))

    return binary.astype(np.uint8)


def dem_masking(
    scene_data: xr.DataArray, dem_data: xr.DataArray, dem_threshold: float
) -> xr.DataArray:
    dem_regrid = dem_data.interp_like(scene_data.isel(time=-1))
    dem_regrid = dem_regrid > dem_threshold

    masked_data = scene_data.groupby("time").apply(
        lambda x: x.where(~dem_regrid, other=1)
    )

    return masked_data


def binary_extraction(
    scene_data: xr.DataArray,
    dem_data: Optional[xr.DataArray] = None,
    dem_threshold: float = 30.0,
    filter_size: int = 5,
    binary_threshold: Optional[Union[Iterable[float], float]] = None,
    return_db_data: bool = False,
) -> Iterable[xr.DataArray]:
    if not isinstance(binary_threshold, Iterable):
        binary_threshold = [binary_threshold for _ in range(len(scene_data.time))]

    logger.info("Convert to dB...")
    db_data = scene_data.groupby("time").apply(db_scale)

    logger.info("Speckle filter...")
    filter_data = db_data.groupby("time").apply(
        lambda img: xr.apply_ufunc(lee_filter, img, kwargs={"size": filter_size})
    )

    logger.info("Segmentation...")
    filter_data["threshold"] = xr.DataArray(
        binary_threshold, coords=[filter_data.time], dims=["time"]
    )
    binary_data = filter_data.groupby("time").apply(
        lambda img: xr.apply_ufunc(
            segmentation, kwargs={"img": img, "threshold": img.threshold.values}
        )
    )

    results = []

    if dem_data is not None:
        logger.info(f"Mask pixel with elevation: {dem_threshold} m")
        binary_data = dem_masking(binary_data, dem_data, dem_threshold)
        results.append(binary_data)

    if return_db_data:
        results.append(db_data)

    return results


def subpixel_contours(
    da: xr.DataArray,
    z_values: Union[Iterable[float], float] = [0.0],
    crs: Union[CRS, str] = None,
    affine: Optional[Union[Affine, Iterable[float]]] = None,
    attribute_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[Path, str]] = None,
    min_vertices: int = 2,
    dim: str = "time",
    errors: str = "ignore",
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    def contours_to_multiline(
        da_i: xr.DataArray, z_value: float, min_vertices: int = 2
    ) -> MultiLineString:
        """
        Helper function to apply marching squares contour extraction
        to an array and return a data as a shapely MultiLineString.
        The `min_vertices` parameter allows you to drop small contours
        with less than X vertices.
        """

        # Extracts contours from array, and converts each discrete
        # contour into a Shapely LineString feature. If the function
        # returns a KeyError, this may be due to an unresolved issue in
        # scikit-image: https://github.com/scikit-image/scikit-image/issues/4830
        line_features = [
            LineString(i[:, [1, 0]])
            for i in measure.find_contours(da_i.data, z_value)
            if i.shape[0] > min_vertices
        ]

        # Output resulting lines into a single combined MultiLineString
        return MultiLineString(line_features)

    # Check if CRS is provided as a xarray.DataArray attribute.
    # If not, require supplied CRS
    try:
        crs = da.crs
    except:
        if crs is None:
            raise ValueError(
                "Please add a `crs` attribute to the "
                "xarray.DataArray, or provide a CRS using the "
                "function's `crs` parameter (e.g. 'EPSG:3577')"
            )

    # Check if Affine transform is provided as a xarray.DataArray method.
    # If not, require supplied Affine
    try:
        affine = da.geobox.transform
    except KeyError:
        affine = da.transform
    except:
        if affine is None:
            raise TypeError(
                "Please provide an Affine object using the "
                "`affine` parameter (e.g. `from affine import "
                "Affine; Affine(30.0, 0.0, 548040.0, 0.0, -30.0, "
                "6886890.0)`"
            )

    # If z_values is supplied is not a list, convert to list:
    z_values = (
        z_values
        if (isinstance(z_values, list) or isinstance(z_values, np.ndarray))
        else [z_values]
    )

    # Test number of dimensions in supplied data array
    if len(da.shape) == 2:
        if verbose:
            logger.info(f"Operating in multiple z-value, single array mode")
        dim = "z_value"
        contour_arrays = {
            str(i)[0:10]: contours_to_multiline(da, i, min_vertices) for i in z_values
        }

    else:

        # Test if only a single z-value is given when operating in
        # single z-value, multiple arrays mode
        if verbose:
            logger.info(f"Operating in single z-value, multiple arrays mode")
        if len(z_values) > 1:
            raise ValueError(
                "Please provide a single z-value when operating "
                "in single z-value, multiple arrays mode"
            )

        contour_arrays = {
            str(i)[0:10]: contours_to_multiline(da_i, z_values[0], min_vertices)
            for i, da_i in da.groupby(dim)
        }

    # If attributes are provided, add the contour keys to that dataframe
    if attribute_df is not None:

        try:
            attribute_df.insert(0, dim, contour_arrays.keys())
        except ValueError:

            raise ValueError(
                "One of the following issues occured:\n\n"
                "1) `attribute_df` contains a different number of "
                "rows than the number of supplied `z_values` ("
                "'multiple z-value, single array mode')\n"
                "2) `attribute_df` contains a different number of "
                "rows than the number of arrays along the `dim` "
                "dimension ('single z-value, multiple arrays mode')"
            )

    # Otherwise, use the contour keys as the only main attributes
    else:
        attribute_df = list(contour_arrays.keys())

    # Convert output contours to a geopandas.GeoDataFrame
    contours_gdf = gpd.GeoDataFrame(
        data=attribute_df, geometry=list(contour_arrays.values()), crs=crs
    )

    # Define affine and use to convert array coords to geographic coords.
    # We need to add 0.5 x pixel size to the x and y to obtain the centre
    # point of our pixels, rather than the top-left corner
    shapely_affine = [
        affine.a,
        affine.b,
        affine.d,
        affine.e,
        affine.xoff + affine.a / 2.0,
        affine.yoff + affine.e / 2.0,
    ]
    contours_gdf["geometry"] = contours_gdf.affine_transform(shapely_affine)

    # Rename the data column to match the dimension
    contours_gdf = contours_gdf.rename({0: dim}, axis=1)

    # Drop empty timesteps
    empty_contours = contours_gdf.geometry.is_empty
    failed = ", ".join(map(str, contours_gdf[empty_contours][dim].to_list()))
    contours_gdf = contours_gdf[~empty_contours]

    # Raise exception if no data is returned, or if any contours fail
    # when `errors='raise'. Otherwise, logger.info failed contours
    if empty_contours.all() and errors == "raise":
        raise RuntimeError(
            "Failed to generate any valid contours; verify that "
            "values passed to `z_values` are valid and present "
            "in `da`"
        )
    elif empty_contours.all() and errors == "ignore":
        if verbose:
            logger.info(
                "Failed to generate any valid contours; verify that "
                "values passed to `z_values` are valid and present "
                "in `da`"
            )
    elif empty_contours.any() and errors == "raise":
        raise Exception(f"Failed to generate contours: {failed}")
    elif empty_contours.any() and errors == "ignore":
        if verbose:
            logger.info(f"Failed to generate contours: {failed}")

    # If asked to write out file, test if geojson or shapefile
    if output_path and output_path.endswith(".geojson"):
        if verbose:
            logger.info(f"Writing contours to {output_path}")
        contours_gdf.to_crs("EPSG:4326").to_file(filename=output_path, driver="GeoJSON")

    if output_path and output_path.endswith(".shp"):
        if verbose:
            logger.info(f"Writing contours to {output_path}")
        contours_gdf.to_file(filename=output_path)

    return contours_gdf


def smooth_linestring(linestring: LineString, smooth_sigma: int) -> LineString:
    """
    Uses a gauss filter to smooth out the LineString coordinates.
    """
    smooth_x = np.array(ndimage.gaussian_filter1d(linestring.xy[0], smooth_sigma))
    smooth_y = np.array(ndimage.gaussian_filter1d(linestring.xy[1], smooth_sigma))
    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)
    linestring_smoothed = LineString(smoothed_coords)

    return linestring_smoothed


def coastline_extraction(
    scene_data: xr.DataArray,
    crs: Union[CRS, str],
    transform: Union[Affine, str],
    min_vertices: int = 100,
    smooth_size: int = 5,
) -> gpd.GeoDataFrame:
    logger.info("Extract coastline...")

    coastline_gdf = subpixel_contours(
        scene_data,
        min_vertices=min_vertices,
        crs=crs,
        affine=transform,
    )

    new_lines = []
    for i, row in coastline_gdf.iterrows():
        line = row.geometry
        if line.geom_type == "MultiLineString":
            new_line = MultiLineString(
                [smooth_linestring(l, smooth_size) for l in line.geoms]
            )
        else:
            new_line = smooth_linestring(line, smooth_size)
        new_lines.append(new_line)

    coastline_gdf.geometry = new_lines

    return coastline_gdf
