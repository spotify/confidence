from typing import Tuple

import numpy as np
from pandas import DataFrame, Series

from spotify_confidence.analysis.constants import CI_LOWER, CI_UPPER, SFX1, SFX2


class BootstrapComputer(object):
    def __init__(self, bootstrap_samples_column, interval_size):
        self._bootstrap_samples = bootstrap_samples_column
        self._interval_size = interval_size

    def _point_estimate(self, row: Series) -> float:
        return row[self._bootstrap_samples].mean()

    def _variance(self, row: Series) -> float:
        variance = row[self._bootstrap_samples].var()

        if variance < 0:
            raise ValueError("Computed variance is negative. " "Please check your inputs.")
        return variance

    def _std_err(self, row: Series) -> float:
        return None

    def _add_point_estimate_ci(self, row: DataFrame) -> Series:

        row[CI_LOWER] = np.percentile(row[self._bootstrap_samples], 100 * (1 - self._interval_size) / 2)
        row[CI_UPPER] = np.percentile(row[self._bootstrap_samples], 100 * (1 - (1 - self._interval_size) / 2))
        return row

    def _p_value(self, row) -> float:
        return -1

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        differences = row[self._bootstrap_samples + SFX2] - row[self._bootstrap_samples + SFX1]
        lower = np.percentile(differences, 100 * row[alpha_column] / 2)
        upper = np.percentile(differences, 100 * (1 - row[alpha_column] / 2))
        return lower, upper

    def _achieved_power(self, df: DataFrame, mde: float, alpha: float) -> DataFrame:
        return None
