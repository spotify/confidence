from typing import Tuple

import numpy as np
from pandas import DataFrame, Series

from spotify_confidence.analysis.constants import CI_LOWER


class BootstrapComputer(object):
    def __init__(self, bootstrap_samples_column, interval_size):
        self._bootstrap_samples = bootstrap_samples_column

    def _point_estimate(self, row: Series) -> float:
        return row[self._bootstrap_samples].mean()

    def _variance(self, row: Series) -> float:
        variance = row[self._bootstrap_samples].variance()

        if variance < 0:
            raise ValueError('Computed variance is negative. '
                             'Please check your inputs.')
        return variance

    def _add_point_estimate_ci(self, row: DataFrame) -> Series:

        row[CI_LOWER] = np.percentile(row[self._bootstrap_samples], (1 - self._interval_size) / 2)
        row[CI_LOWER] = np.percentile(row[self._bootstrap_samples], 1 - (1-self._interval_size) / 2)
        return row

    def _p_value(self, row) -> float:
        p_value = 0.5
        return p_value

    def _ci(self, row, alpha_column: str) -> Tuple[float, float]:
        return 0.5

    def _achieved_power(self,
                        df: DataFrame,
                        mde: float,
                        alpha: float) -> DataFrame:
        return 0.5
