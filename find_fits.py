import polars as pl
from shapely.geometry import Polygon, Point
from typing import List, Tuple

'''
BORE_LAT and BORE_LON are approximately the center points of the rectangular bound of a fits file.
So if any fits file's BORE_LAT and BORE_LON are lying in a given area then it must cover >= 25% of that area.
'''

# given a dataframe return a list of dataframes
class FitsSurrounding:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def check_overlap(self, border: List[Tuple[float, float]]) -> pl.DataFrame:
        area = Polygon(border)
        mask = (
            pl.struct(["BORE_LAT", "BORE_LON"])
            .map_elements(lambda row: area.contains(Point(row["BORE_LON"], row["BORE_LAT"])), return_dtype= pl.Boolean)
        )
        return self.df.filter(mask)

    def fits_overlap(self) -> List:
        results = []
        for row in self.df.iter_rows(named=True):
            border = [
                (row["V0_LON"], row["V0_LAT"]),
                (row["V1_LON"], row["V1_LAT"]),
                (row["V2_LON"], row["V2_LAT"]),
                (row["V3_LON"], row["V3_LAT"])
            ]
            matching_rows = self.check_overlap(border)
            results.append(matching_rows)
        return results


# given a fits file and the dataframe, return the dataframe that corresponds to the given fits file
class FitsCorrespond:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def check_overlap(self, border: List[Tuple[float, float]]) -> pl.DataFrame:
        area = Polygon(border)
        mask = (
            pl.struct(["BORE_LAT", "BORE_LON"])
            .map_elements(
                lambda row: area.contains(Point(row["BORE_LON"], row["BORE_LAT"])),
                return_dtype=pl.Boolean
            )
        )
        return self.df.filter(mask)

    def fits_overlap(self, fits_row: dict) -> pl.DataFrame:
        border = [
            (fits_row["V0_LON"], fits_row["V0_LAT"]),
            (fits_row["V1_LON"], fits_row["V1_LAT"]),
            (fits_row["V2_LON"], fits_row["V2_LAT"]),
            (fits_row["V3_LON"], fits_row["V3_LAT"])
        ]

        return self.check_overlap(border)


# given a boundary of points, return a dataframe with all fits files that fall in that area
class FitsTogether:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def check_overlap(self, border: List[Tuple[float, float]]) -> pl.DataFrame:
        area = Polygon(border)
        mask = (
            pl.struct(["BORE_LAT", "BORE_LON"])
            .map_elements(
                lambda row: area.contains(Point(row["BORE_LON"], row["BORE_LAT"])),
                return_dtype=pl.Boolean
            )
        )
        return self.df.filter(mask)
