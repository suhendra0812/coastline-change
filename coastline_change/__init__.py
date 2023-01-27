import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as xr
import xarray as xr
from coastline_change.analysis import create_transects, transect_analysis
from coastline_change.datasets import DEMSTACDataset, S1STACDataset
from coastline_change.detection import binary_extraction, coastline_extraction
from dask.distributed import Client
from pyproj import CRS

logger = logging.getLogger(__name__)


def file_path(path: str) -> Optional[Path]:
    path = Path(path)
    if path.is_file():
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a file.")


def rescale_img(
    img: np.ndarray, target_type_min: float, target_type_max: float, target_type: type
) -> np.ndarray:
    img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))

    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax

    new_img = (a * img + b).astype(target_type)

    return new_img


def save_data(
    data: xr.DataArray,
    band_name: str,
    crs: Union[CRS, str],
    output_dir: Union[Path, str],
    prefix: str,
) -> None:
    for time, d in data.rename(band_name).groupby("time"):
        year = pd.to_datetime(time).year
        raster_path = output_dir / f"{prefix}_{band_name}_{year}.tif"
        d.rio.to_raster(raster_path, crs=crs, compress="lzw")
        logger.info(f"Saved to {raster_path}")


def s1_coastline_change(
    region_id: Any,
    region_gdf: gpd.GeoDataFrame,
    point_gdf: gpd.GeoDataFrame,
    tide_types: Iterable[str],
    collection: str,
    start_date: Union[datetime, str],
    end_date: Union[datetime, str],
    min_area: float = 0.5,
    threshold: Optional[float] = None,
    output_dir: Union[Path, str] = Path("."),
) -> None:
    xmin, ymin, xmax, ymax = region_gdf.total_bounds.tolist()

    dem_stac = DEMSTACDataset(
        collection=["cop-dem-glo-30"],
        area_of_interest=[xmin, ymin, xmax, ymax],
    )
    dem_data = dem_stac.get_data()

    x = point_gdf.unary_union.centroid.x
    y = point_gdf.unary_union.centroid.y

    for tide_type in tide_types:
        logger.info(f"Tide type: {tide_type}")
        suboutput_dir = output_dir / tide_type
        suboutput_dir.mkdir(parents=True, exist_ok=True)

        s1_stac = S1STACDataset(
            collection=collection,
            start_date=start_date,
            stop_date=end_date,
            area_of_interest=[xmin, ymin, xmax, ymax],
            minimum_area_cover=min_area,
            tide_filter=tide_type,
            tide_xy=(x, y),
        )

        s1_data = s1_stac.get_data()
        logger.info(f"S1 Found: {len(s1_data)} datasets")

        logger.info("Load data from dask client...")
        vh_data = s1_data.load()

        binary_data, db_data = binary_extraction(
            scene_data=vh_data,
            dem_data=dem_data,
            binary_threshold=threshold,
            return_db_data=True,
        )

        crs = s1_data.crs

        rescaled_db_data = db_data.groupby("time").apply(
            rescale_img,
            target_type_min=1,
            target_type_max=255,
            target_type=np.uint8,
        )

        save_data(
            data=rescaled_db_data,
            band_name="vh_db",
            crs=crs,
            output_dir=suboutput_dir,
            prefix=f"{region_id:04d}_s1",
        )

        save_data(
            data=binary_data,
            band_name="vh_binary",
            crs=crs,
            output_dir=suboutput_dir,
            prefix=f"{region_id:04d}_s1",
        )

        transform = s1_data.transform
        coastline_gdf = coastline_extraction(
            scene_data=binary_data,
            crs=crs,
            transform=transform,
        )

        baseline = coastline_gdf.geometry.iloc[0]
        transect_gdf = create_transects(
            line=baseline,
            space=500,
            length=100,
            crs=crs,
        )

        transect_analysis_gdf = transect_analysis(
            line_gdf=coastline_gdf,
            transect_gdf=transect_gdf,
            time_column="time",
            reverse=True,
        )

        if transect_analysis_gdf is None:
            continue

        coastline_path = suboutput_dir / f"{region_id:04d}_s1_coastlines.geojson"
        transect_path = suboutput_dir / f"{region_id:04d}_s1_transects.geojson"
        transect_analysis_path = (
            suboutput_dir / f"{region_id:04d}_s1_transect_analysis.geojson"
        )

        coastline_gdf.to_file(coastline_path, driver="GeoJSON")
        transect_gdf.to_file(transect_path, driver="GeoJSON")
        transect_analysis_gdf.to_file(transect_analysis_path, driver="GeoJSON")

    logger.info("Completed.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--collection",
        type=str,
        help="STAC Collection that is available in Microsoft Planetary Computer Catalog.",
    )
    parser.add_argument(
        "--region-file",
        type=file_path,
        help="Polygon file in geospatial format (.shp, .geojson, etc.)",
    )
    parser.add_argument(
        "--point-file",
        type=file_path,
        help="Point file in geospatial format (.shp, .geojson, etc.)",
    )
    parser.add_argument(
        "--region-ids",
        nargs="+",
        type=int,
        help="Provide list of region id which is separated by space. Example: 1 2 3",
    )
    parser.add_argument(
        "--province",
        type=str,
        help="Provide name of province in Indonesia. If this is provided, it will override the list of region id that cover in this area. Example: Jawa Barat",
    )
    parser.add_argument(
        "-t",
        "--tide-types",
        choices=["low", "mean", "high"],
        default="mean",
        const="mean",
        nargs="?",
        help="List of tide type which is used to filter out the dataset.",
    )
    parser.add_argument(
        "-s",
        "--start-date",
        type=lambda d: datetime.strptime(d, r"%Y-%m-%d"),
        help="Set a start date in YYYY-mm-dd format. Example: 2015-01-01",
    )

    parser.add_argument(
        "-e",
        "--end-date",
        type=lambda d: datetime.strptime(d, r"%Y-%m-%d"),
        help="Set a end date in YYYY-mm-dd format. Example: 2015-01-01",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.5,
        help="Provide the minimum of area of scene that cover the region of interest (in 0-1 ratio).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Provide the pixel value threshold of image to get binary image. If not provided, it will use Otsu algorithm to get the threshold automatically.",
    )

    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        help="The output directory which collects the output.",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        type=bool,
        default=False,
        help="Parallelized the execution of the script using Dask distributed.",
    )

    parser.add_argument(
        "-f",
        "--config-file",
        type=file_path,
        help="The config file which provide all the parameters.",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()

    collection = args.collection
    region_file = args.region_file
    point_file = args.point_file
    region_ids = args.region_ids
    province = args.province
    tide_types = args.tide_types
    start_date = args.start_date
    end_date = args.end_date
    min_area = args.min_area
    threshold = args.threshold
    base_output_dir = args.output_dir
    parallel = args.parallel
    config_file = args.config_file

    if config_file:
        with open(config_file) as f:
            config = json.load(f)

        collection = config.get("collection")
        region_file = config.get("region_file")
        point_file = config.get("point_file")
        region_ids = config.get("region_ids")
        province = config.get("province")
        tide_types = config.get("tide_types")
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        min_area = config.get("min_area")
        threshold = config.get("threshold")
        base_output_dir = config.get("output_dir")
        parallel = config.get("parallel")

    region_gdf = gpd.read_file(region_file)
    point_gdf = gpd.read_file(point_file)

    if province:
        region_ids = [
            i + 1 for i in region_gdf.query("province == @PROVINCE").index.tolist()
        ]

    if not base_output_dir:
        base_output_dir = Path(__file__).parent / "output"

    if parallel:
        with Client() as dask_client:
            logger.info(f"Parallelized using Dask.")
            logger.info(f"Dask client dashboard link: {dask_client.dashboard_link}")

            for i, region_id in enumerate(region_ids):
                logger.info(f"({i+1}/{len(region_ids)}) Region ID: {region_id}")

                output_dir = Path(base_output_dir) / f"{region_id:04d}"
                output_dir.mkdir(parents=True, exist_ok=True)

                selected_region_gdf = region_gdf.loc[[region_id - 1]]
                selected_point_gdf = point_gdf.loc[[region_id - 1]]

                s1_coastline_change(
                    region_id,
                    selected_region_gdf,
                    selected_point_gdf,
                    tide_types,
                    collection,
                    start_date,
                    end_date,
                    min_area,
                    threshold,
                    output_dir,
                )
    else:
        for i, region_id in enumerate(region_ids):
            logger.info(f"({i+1}/{len(region_ids)}) Region ID: {region_id}")

            output_dir = Path(base_output_dir) / f"{region_id:04d}"
            output_dir.mkdir(parents=True, exist_ok=True)

            selected_region_gdf = region_gdf.loc[[region_id - 1]]
            selected_point_gdf = point_gdf.loc[[region_id - 1]]

            s1_coastline_change(
                region_id,
                selected_region_gdf,
                selected_point_gdf,
                tide_types,
                collection,
                start_date,
                end_date,
                min_area,
                threshold,
                output_dir,
            )
