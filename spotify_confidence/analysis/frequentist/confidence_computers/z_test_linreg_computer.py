from functools import reduce

from pandas import DataFrame, Series
import numpy as np
from spotify_confidence.analysis.constants import REGRESSION_PARAM, POINT_ESTIMATE, CI_LOWER, CI_UPPER, VARIANCE

from spotify_confidence.analysis.confidence_utils import unlist, dfmatmul, get_all_group_columns

from spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer import ZTestComputer


class ZTestLinregComputer(ZTestComputer):
    def __init__(
        self,
        numerator,
        numerator_sumsq,
        denominator,
        ordinal_group_column,
        interval_size,
        feature_column,
        feature_sum_squares_column,
        feature_cross_sum_column,
        method_column,
    ):
        self._numerator = numerator
        self._numerator_sumsq = numerator_sumsq
        self._denominator = denominator
        self._ordinal_group_column = ordinal_group_column
        self._interval_size = interval_size
        self._feature = feature_column
        self._feature_ssq = feature_sum_squares_column
        self._feature_cross = feature_cross_sum_column
        self._method_column = method_column

    def _estimate_slope(self, df) -> DataFrame:
        def col_sum(x):
            out = reduce(lambda x, y: x + y, x)
            return out

        def dimension(x):
            return x.shape[0] if isinstance(x, np.ndarray) and x.size > 1 else 1

        k = df[self._feature_ssq].apply(dimension).iloc[0]

        XX0 = np.zeros((k + 1, k + 1))
        XX0[1 : (k + 1), 1 : (k + 1)] = col_sum(df[self._feature_ssq])
        XX0[0, 0] = col_sum(df[self._denominator])
        XX0[0, 1 : (k + 1)] = col_sum(df[self._feature])
        XX0[1 : (k + 1), 0] = col_sum(df[self._feature])

        Xy0 = np.zeros((k + 1, 1))
        Xy0[
            0,
        ] = col_sum(df[self._numerator])
        Xy0[1 : (k + 1),] = np.atleast_2d(
            col_sum(df[self._feature_cross])
        ).reshape(-1, 1)

        b = np.matmul(np.linalg.inv(XX0), Xy0)
        out = b[1 : (k + 1)]
        if out.size == 1:
            out = out.item()

        outseries = Series(index=df.index, dtype=df[self._feature].dtype)
        df[REGRESSION_PARAM] = outseries.apply(lambda x: out)
        return df

    def _point_estimate(self, row: Series) -> float:

        if row[self._denominator] == 0:
            raise ValueError(
                """Can't compute point estimate:
                                denominator is 0"""
            )
        out1 = row[self._numerator] / row[self._denominator]
        out2 = dfmatmul(row[REGRESSION_PARAM], row[self._feature], outer=False)
        out3 = out2 / row[self._denominator]
        return out1 - out3

    def _variance(self, row: DataFrame) -> Series:
        XX = unlist(row[self._feature_ssq])
        X = unlist(row[self._feature])
        Xy = unlist(row[self._feature_cross])
        y = row[self._numerator]
        yy = row[self._numerator_sumsq]
        n = row[self._denominator]

        sample_var = XX / n - dfmatmul(X / n, X / n)
        sample_cov = Xy / n - dfmatmul(X / n, y / n)
        variance1 = yy / n - (y / n) ** 2
        b = np.atleast_2d(row[REGRESSION_PARAM])
        variance2 = np.matmul(np.transpose(b), np.matmul(sample_var, b)).item()
        variance3 = -2 * np.matmul(np.transpose(b), sample_cov).item()

        return variance1 + variance2 + variance3

    def _sufficient_statistics(self) -> DataFrame:
        if self._sufficient is None:
            self._sufficient = (
                self._df.assign(**{REGRESSION_PARAM: self._estimate_slope})
                .assign(**{POINT_ESTIMATE: self._adjusted_point_estimate})
                .assign(**{VARIANCE: self._adjusted_variance})
                .pipe(self._add_point_estimate_ci)
            )
        return self._sufficient
