import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import geopandas as gpd
import pandas as pd
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from dateutil.parser import parse as dateparse
from dateutil.relativedelta import relativedelta
from pystac import Item, ItemCollection
from rioxarray.merge import merge_arrays
from shapely.geometry import box, mapping, shape
from tqdm import tqdm

from coastline_change.tides import (
    TideType,
    get_interpolated_tide,
    tide_type_calculation,
)

logger = logging.getLogger(__name__)


def intersection_percent(item: Item, aoi: Dict[str, Any]) -> float:
    """The percentage that the Item's geometry intersects the AOI. An Item that
    completely covers the AOI has a value of 1.
    """
    geom_item = shape(item.geometry)
    geom_aoi = shape(aoi)

    intersected_geom = geom_aoi.intersection(geom_item)

    intersection_percent = intersected_geom.area / geom_aoi.area

    return intersection_percent


def filter_items_by_tide(
    item_list: Union[ItemCollection, Iterable[Item]],
    tide_xy: Iterable[float],
    tide_filter: Union[TideType, str, float],
) -> ItemCollection:

    datetime_list = [item.datetime for item in item_list]
    tide_df = get_interpolated_tide(tide_xy[0], tide_xy[1], datetime_list)
    tide_list = tide_df["level"].tolist()

    if isinstance(tide_filter, (TideType, str)):
        tide_filter = tide_type_calculation(tide_list, tide_filter)

    tide_item_list = []
    for item, tide in zip(item_list, tide_list):
        item.properties["tide"] = tide
        tide_item_list.append(item)

    item_group = itertools.groupby(tide_item_list, lambda item: item.datetime.year)
    item_list = [
        sorted(group, key=lambda x: abs(x.properties["tide"] - tide_filter))[0]
        for _, group in item_group
    ]

    return ItemCollection(item_list)


def filter_data_by_tide(
    scene_data: xr.DataArray,
    tide_xy: Iterable[float],
    tide_filter: Union[TideType, str, float],
) -> xr.DataArray:

    datetime_list = pd.to_datetime(scene_data.time.values).to_pydatetime()
    tide_df = get_interpolated_tide(tide_xy[0], tide_xy[1], datetime_list)
    tide_list = tide_df["level"].tolist()

    if isinstance(tide_filter, (TideType, str)):
        tide_filter = tide_type_calculation(tide_list, tide_filter)

    logger.info(f"Filter scene data by tide value: {tide_filter} m...")

    tide_list = [
        xr.concat(
            sorted(group, key=lambda x: abs(x.tide - tide_filter))[:1],
            dim="time",
        )
        for _, group in scene_data.groupby("time.year")
    ]
    tide_scene_data = xr.concat(tide_list, dim="time")

    return tide_scene_data


class STACDataset(ABC):
    @abstractmethod
    def get_items(self) -> ItemCollection:
        pass

    @abstractmethod
    def get_data(self) -> xr.DataArray:
        pass


@dataclass
class S1STACDataset(STACDataset):
    collection: str
    start_date: Union[datetime, str]
    stop_date: Union[datetime, str]
    area_of_interest: Union[Path, str, Iterable[float]]
    minimum_area_cover: float = 0.5
    tide_filter: Optional[Union[TideType, float]] = None
    tide_xy: Optional[Iterable[float]] = None

    def __post_init__(self) -> None:
        self.client_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = pystac_client.Client.open(self.client_url)

        if isinstance(self.start_date, str):
            self.start_date = dateparse(self.start_date)
        if isinstance(self.stop_date, str):
            self.stop_date = dateparse(self.stop_date)
        if isinstance(self.area_of_interest, (Path, str)):
            if isinstance(self.area_of_interest, str):
                self.area_of_interest = Path(self.area_of_interest)
            self.area_of_interest = gpd.read_file(
                self.area_of_interest
            ).total_bounds.tolist()

        if isinstance(self.tide_filter, (TideType, str)):
            if isinstance(self.tide_filter, str):
                self.tide_filter = TideType(self.tide_filter)

            if not self.tide_xy:
                warnings.warn(
                    """'tide_xy' is not provided.
                    Use centroid of 'area_of_interest' instead."""
                )
                self.tide_xy = [
                    box(*self.area_of_interest).centroid.x,
                    box(*self.area_of_interest).centroid.y,
                ]

    def get_items(self) -> ItemCollection:
        item_list = []

        date_range = pd.date_range(self.start_date, self.stop_date, freq="YS")
        date_range = date_range.to_pydatetime()
        for start_date in tqdm(date_range, desc="Get STAC Items"):
            stop_date = start_date + relativedelta(years=1)

            search = self.catalog.search(
                collections=[self.collection],
                datetime=[start_date, stop_date],
                bbox=self.area_of_interest,
                query={"sar:polarizations": {"eq": ["VV", "VH"]}},
            )
            items = search.item_collection()
            for item in items:
                area = intersection_percent(item, mapping(box(*self.area_of_interest)))
                if area >= self.minimum_area_cover:
                    item_list.append(item)

        item_list = sorted(item_list, key=lambda x: x.datetime)

        if self.tide_filter:
            item_list = filter_items_by_tide(item_list, self.tide_xy, self.tide_filter)

        items = ItemCollection(item_list)

        return items

    def get_data(self) -> xr.DataArray:
        signed_items = [pc.sign(item).to_dict() for item in self.get_items()]

        data = (
            stackstac.stack(
                signed_items,
                bounds_latlon=self.area_of_interest,
                epsg=3857,
                resolution=10,
            )
            # .where(lambda x: x > 0, other=np.nan)
            .sel(band="vh").rio.write_nodata(0)
        )

        return data


@dataclass
class S2STACDataset(STACDataset):
    collection: str
    start_date: Union[datetime, str]
    stop_date: Union[datetime, str]
    area_of_interest: Union[Path, str, Iterable[float]]
    minimum_area_cover: float = 0.5
    maximum_cloud_cover: float = 0.1

    def __post_init__(self) -> None:
        self.client_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = pystac_client.Client.open(self.client_url)

        if isinstance(self.area_of_interest, (Path, str)):
            if isinstance(self.area_of_interest, str):
                self.area_of_interest = Path(self.area_of_interest)
            self.area_of_interest = gpd.read_file(
                self.area_of_interest
            ).total_bounds.tolist()

    def get_items(self) -> ItemCollection:
        item_list = []

        date_range = pd.date_range(self.start_date, self.stop_date, freq="YS")
        date_range = date_range.to_pydatetime()
        for start_date in tqdm(date_range, desc="Get STAC Items"):
            stop_date = start_date + relativedelta(years=1)

            search = self.catalog.search(
                collections=[self.collection],
                datetime=[start_date, stop_date],
                bbox=self.area_of_interest,
                query={"eo:cloud_cover": {"lt": self.maximum_cloud_cover * 100}},
            )
            items = search.item_collection()
            for item in items:
                area = intersection_percent(item, mapping(box(*self.area_of_interest)))
                if area >= self.minimum_area_cover:
                    item_list.append(item)

        item_list = sorted(item_list, key=lambda x: x.datetime)
        items = ItemCollection(item_list)

        return items

    def get_data(self) -> xr.DataArray:
        return super().get_data()


@dataclass
class DEMSTACDataset(STACDataset):
    collection: str
    area_of_interest: Union[Path, str, Iterable[float]]

    def __post_init__(self) -> None:
        self.client_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.catalog = pystac_client.Client.open(self.client_url)

        if isinstance(self.area_of_interest, (Path, str)):
            if isinstance(self.area_of_interest, str):
                self.area_of_interest = Path(self.area_of_interest)
            self.area_of_interest = gpd.read_file(
                self.area_of_interest
            ).total_bounds.tolist()

    def get_items(self) -> ItemCollection:
        search = self.catalog.search(
            collections=[self.collection], bbox=self.area_of_interest
        )

        items = search.item_collection()

        return items

    def get_data(self) -> xr.DataArray:
        signed_items = [pc.sign(item).to_dict() for item in self.get_items()]

        data = (
            stackstac.stack(
                signed_items, bounds_latlon=self.area_of_interest, epsg=3857
            )
            # .where(lambda x: x > 0, other=np.nan)
            .sel(band="data").rio.write_nodata(0)
        )

        merged_data = merge_arrays([dem for dem in data.load()])

        return merged_data
