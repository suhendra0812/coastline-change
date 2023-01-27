import logging
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


def create_transects(
    line: LineString, space: float, length: float, crs: Union[CRS, str]
) -> gpd.GeoDataFrame:
    logger.info("Create transects...")

    # Profile spacing. The distance at which to space the perpendicular profiles
    # In the same units as the original shapefile (e.g. metres)
    space = space

    # Length of cross-sections to calculate either side of central line
    # i.e. the total length will be twice the value entered here.
    # In the same co-ordinates as the original shapefile
    length = length

    # Define a schema for the output features. Add a new field called 'Dist'
    # to uniquely identify each profile

    transect_list = []

    # Calculate the number of profiles to generate
    n_prof = int(line.length / space)

    # Start iterating along the line
    for prof in range(1, n_prof + 1):
        # Get the start, mid and end points for this segment
        seg_st = line.interpolate((prof - 1) * space)
        seg_mid = line.interpolate((prof - 0.5) * space)
        seg_end = line.interpolate(prof * space)

        # Get a displacement vector for this segment
        vec = np.array(
            [
                [
                    seg_end.x - seg_st.x,
                ],
                [
                    seg_end.y - seg_st.y,
                ],
            ]
        )

        # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
        rot_anti = np.array([[0, -1], [1, 0]])
        rot_clock = np.array([[0, 1], [-1, 0]])
        vec_anti = np.dot(rot_anti, vec)
        vec_clock = np.dot(rot_clock, vec)

        # Normalise the perpendicular vectors
        len_anti = ((vec_anti**2).sum()) ** 0.5
        vec_anti = vec_anti / len_anti
        len_clock = ((vec_clock**2).sum()) ** 0.5
        vec_clock = vec_clock / len_clock

        # Scale them up to the profile length
        vec_anti = vec_anti * length
        vec_clock = vec_clock * length

        # Calculate displacements from midpoint
        prof_st = (seg_mid.x + float(vec_clock[0]), seg_mid.y + float(vec_clock[1]))
        prof_end = (seg_mid.x + float(vec_anti[0]), seg_mid.y + float(vec_anti[1]))

        distance = (prof - 0.5) * space
        transect = LineString([prof_end, prof_st])

        gdf = gpd.GeoDataFrame({"distance": [distance]}, geometry=[transect])

        transect_list.append(gdf)

    transect_gdf = pd.concat(transect_list, ignore_index=True)
    transect_gdf.crs = crs

    return transect_gdf


def transect_analysis(
    line_gdf: gpd.GeoDataFrame,
    transect_gdf: gpd.GeoDataFrame,
    time_column: str,
    reverse: bool = False,
) -> gpd.GeoDataFrame:
    logger.info("Transect analysis...")

    line_gdf = line_gdf.copy()
    transect_gdf = transect_gdf.copy()

    # memastikan formatnya menjadi objek datetime python
    line_gdf[time_column] = pd.to_datetime(line_gdf[time_column])
    # line_gdf["time_idx"], _ = pd.factorize(line_gdf[time_column])

    # diurutkan berdasarkan waktu
    line_gdf.sort_values(by=time_column, inplace=True, ignore_index=True)
    transect_gdf.reset_index(drop=True, inplace=True)

    # list hasil analisis transek
    analysis_list = []

    # looping row GeoDataFrame transek
    for i, transect in transect_gdf.iterrows():
        # mengambil informasi titik awal dan akhir dari satu transek
        start, end = transect.geometry.boundary.geoms
        # jika reverse titiknya ditukar
        if reverse:
            start, end = end, start
        # jika antara garis pantai dan transek ada yang bersentuhan, maka lakukan proses di bawah ini
        if any(line_gdf.geometry.intersects(transect.geometry)):
            intersect_gdf = line_gdf.copy()
            # mengambil titik intersection-nya
            intersect_gdf.geometry = intersect_gdf.geometry.intersection(
                transect.geometry
            )
            # list jenis geometry
            geom_types = [geom.geom_type for geom in intersect_gdf.geometry]
            # jika jumlah jenis geometry 'Point' sama dengan jumlah garis pantai, amak lanjutkan ke proses di bawah ini
            if geom_types.count("Point") == len(intersect_gdf):
                # mengumpulkan infromasi hasil analisis, diawali dengan nama yang disi oleh urutan trasek
                analysis_data = {"name": [i]}

                # di bawah ini akan dilakukan perhitungan perubahan garis pantai setiap dua tahun sekali
                # contoh:
                # jika tahun di garis pantai = [2015, 2016, 2017, 2018], maka perubahan garis pantai akan dihitung sbb:
                # 1. Garis pantai 2015 dikurangi garis pantai 2016
                # 2. Garis pantai 2016 dikurangi garis pantai 2015
                # 3. Garis pantai 2017 dikurangi garis pantai 2016
                # 4. Garis pantai 2018 dikurangi garis pantai 2017

                # ini penjelasan kodingannya
                # looping jumlah garis pantai dan urutannya
                # jika jumlah garis pantai ada 3 berarti nilai j = [0, 1, 2] untuk digunakan sebagai indeks garis pantai lamam
                # nilai k = j + 1, maka hasilnya k = [1, 2, 3] untuk digunakan sebagai indeks garis pantai baru
                # kemudian dilakukan pengecekan, jika k (indeks garis pantai baru) terakhir sama dengan jumlah garis pantai maka proses berhenti (break)
                for j in range(len(intersect_gdf)):
                    k = j + 1
                    if k == len(intersect_gdf):
                        break
                    # garis pantai tanggal lama
                    oldest_intersect = intersect_gdf.iloc[j]
                    # tanggal garis pantai tanggal lama
                    oldest_date = oldest_intersect[time_column]
                    # geometry garis pantai tanggal lama
                    oldest_geom = oldest_intersect.geometry
                    # jarak garis pantai tanggal lama dengan titik start dari transek
                    oldest_distance = oldest_geom.distance(start)
                    # garis pantai tanggal baru
                    latest_intersect = intersect_gdf.iloc[k]
                    # garis pantai tanggal baru
                    latest_date = latest_intersect[time_column]
                    # geometry garis pantai tanggal baru
                    latest_geom = latest_intersect.geometry
                    # jarak garis pantai tanggal baru dengan titik start dari transek
                    latest_distance = latest_geom.distance(start)

                    # ini hanya untuk penamaan kolom, setiap hasil analisis di kolom tabel akan ditambahkan tanggal garis pantainya
                    date_str = oldest_date.strftime("%Y%m%d")
                    # date_idx = oldest_intersect["time_idx"]

                    # pengecekan jika j > 0 maka lakukan perhitungan perubahan di bawah ini
                    if j > 0:
                        change = latest_distance - oldest_distance
                        rate = change / ((latest_date - oldest_date).days / 365)
                    # j = 0 maka belum ada perubahan
                    else:
                        change = 0
                        rate = 0

                    # masukkan hasil analisis berdasarkan tanggal garis pantai
                    analysis_data[f"distance_{date_str}"] = [oldest_distance]
                    analysis_data[f"change_{date_str}"] = [change]
                    analysis_data[f"rate_{date_str}"] = [rate]

                # hasil analisis perubahan garis pantai disimpan dalam GeoDataFrame
                # kolom GeoDataFrame berisi:
                # - name (urutan transek)
                # - distance_[tanggal_garis_pantai] (jarak dari titk awal transek)
                # - change_[tanggal_garis_pantai] (perubahan jarak antara garis pantai lama dan baru)
                # - rate_[tanggal_garis_pantai] (laju perubahan jarak antara garis pantai lama dan baru per satuan waktu)
                # - mean_distance (rata-rata perubahan dari semua distance per tanggal)
                # - mean_change (rata-rata perubahan dari semua change per tanggal)
                # - mean_rate (rata-rata perubahan dari semua rate per tanggal)

                # membuat objek geometry line/garis berdasarkan geometry dari titik-titik intersection
                analysis_geom = LineString(intersect_gdf.geometry)
                # membuat objek GeoDataFrame berdasarkan hasil analisis dan geometry garis intersection
                analysis_gdf = gpd.GeoDataFrame(analysis_data, geometry=[analysis_geom])

                # di bawah ini perhitungan rata-rata masing-masing kolom distance, change dan rate
                distance_columns = analysis_gdf.columns[
                    analysis_gdf.columns.str.contains("distance")
                ]
                analysis_gdf["mean_distance"] = analysis_gdf[distance_columns].mean(
                    axis=1
                )

                change_columns = analysis_gdf.columns[
                    analysis_gdf.columns.str.contains("change")
                ]
                analysis_gdf["mean_change"] = analysis_gdf[change_columns].mean(axis=1)

                rate_columns = analysis_gdf.columns[
                    analysis_gdf.columns.str.contains("rate")
                ]
                analysis_gdf["mean_rate"] = analysis_gdf[rate_columns].mean(axis=1)

                # hasil analisis setiap transek digabungkan ke dalam list
                analysis_list.append(analysis_gdf)

    if not analysis_list:
        logger.warning("No analysis resulted")
        return

    # penggabungan list hasil analisis menjadi satu objek GeoDataFrame
    transect_analysis_gdf = pd.concat(analysis_list, ignore_index=True)
    # sistem proyeksi disamakan dengan garis pantai
    transect_analysis_gdf.crs = line_gdf.crs

    return transect_analysis_gdf
