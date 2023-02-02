import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Callable, Dict, Iterable, Union

import httpx
import numpy as np
import pandas as pd
from dateutil.parser import parse as dateparse

logger = logging.getLogger(__name__)


class TideType(Enum):
    LOW = "low"
    HIGH = "high"
    MEAN = "mean"


TIDE_CALCULATION: Dict[TideType, Callable] = {
    TideType.HIGH: np.max,
    TideType.LOW: np.min,
    TideType.MEAN: np.mean,
}


def tide_interpolation(
    tide_df: pd.DataFrame, datetime_list: Iterable[datetime]
) -> pd.DataFrame:
    df = tide_df.copy()
    df.set_index("datetime", inplace=True)
    index_list = pd.DatetimeIndex(
        df.index.tolist() + pd.to_datetime(datetime_list, utc=True).tolist()
    )
    interp_df = df.copy()
    interp_df = interp_df.reindex(index_list)
    interp_df = interp_df[["level"]].interpolate(method="time")
    interp_df = interp_df.loc[pd.to_datetime(datetime_list, utc=True)]
    interp_df.sort_index(inplace=True)
    interp_df["lat"] = np.repeat(df["lat"].unique(), len(interp_df))
    interp_df["lon"] = np.repeat(df["lon"].unique(), len(interp_df))
    interp_df.reset_index(inplace=True)
    interp_df.rename(columns={"index": "datetime"}, inplace=True)
    logger.info("Tide interpolated")

    return interp_df


def tide_type_calculation(
    tide_list: Union[np.ndarray, Iterable[float]], tide_type: Union[TideType, str]
) -> float:
    if not isinstance(tide_type, TideType):
        tide_type = TideType(tide_type)
    return TIDE_CALCULATION[tide_type](tide_list)


@dataclass
class BIGTide:
    longitude: float
    latitude: float
    start_date: Union[datetime, str]
    stop_date: Union[datetime, str]
    url: str = field(
        default="https://srgi.big.go.id/tides_data/prediction-v2",
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.start_date, datetime):
            self.start_date = dateparse(self.start_date)
        if not isinstance(self.stop_date, datetime):
            self.stop_date = dateparse(self.stop_date)

    def get_data(self) -> pd.DataFrame:
        logger.info("Tide predicting...")
        params = {
            "coords": f"{self.longitude},{self.latitude}",
            "awal": f"{self.start_date.date().isoformat()}",
            "akhir": f"{self.stop_date.date().isoformat()}",
        }

        with httpx.Client(timeout=30, verify=False) as client:
            r = client.get(self.url, params=params)

            results = r.json()["results"]
            predictions = {
                key: value
                for key, value in results["predictions"].items()
                if len(value) == 5
            }
            values = [v for v in predictions.values()]
            error_list = ["Site", "is", "out", "of", "model", "grid", "OR", "land"]
            if set(error_list).issubset(values[0]):
                raise ValueError(" ".join(error_list))
            df = pd.DataFrame(data=values)
            df.columns = ["lat", "lon", "date", "time", "level"]
            df["lat"] = df["lat"].astype(float)
            df["lon"] = df["lon"].astype(float)
            df["level"] = df["level"].astype(float)
            df["datetime"] = pd.to_datetime(
                df["date"].str.cat(df["time"], sep="T"), utc=True
            )
            df = df.sort_values(by="datetime")

        return df


def get_interpolated_tide(
    longitude: float, latitude: float, datetime_list: Iterable[datetime]
) -> pd.DataFrame:
    tide = BIGTide(
        longitude=longitude,
        latitude=latitude,
        start_date=datetime_list[0],
        stop_date=datetime_list[-1],
    )

    tide_df = tide.get_data()
    interp_tide_df = tide_interpolation(tide_df, datetime_list)

    return interp_tide_df


def main() -> None:
    datetime_list = [
        "2023-01-01T01:30:00",
        "2023-01-15T12:00:30",
        "2023-01-21T06:45:05",
    ]
    tide = BIGTide(
        longitude=107.17644745549725,
        latitude=-5.874220082346136,
        start_date=datetime_list[0],
        stop_date=datetime_list[-1],
    )
    tide_df = tide.get_data()
    interp_tide_df = tide_interpolation(tide_df, datetime_list)
    logger.info(f"Tide data: {len(interp_tide_df)} found.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()
