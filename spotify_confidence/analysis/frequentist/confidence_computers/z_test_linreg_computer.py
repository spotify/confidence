from functools import reduce
from typing import Union, Dict

import numpy as np
from pandas import DataFrame, Series

from spotify_confidence.analysis.confidence_utils import unlist, dfmatmul
from spotify_confidence.analysis.constants import (
    REGRESSION_PARAM,
    FEATURE,
    FEATURE_SUMSQ,
    FEATURE_CROSS,
    NUMERATOR,
    DENOMINATOR,
)
from spotify_confidence.analysis.frequentist.confidence_computers import z_test_computer


def estimate_slope(df, **kwargs: Dict) -> DataFrame:
    if kwargs[FEATURE] not in df:
        return df

    def col_sum(x):
        return reduce(lambda x, y: x + y, x)

    def dimension(x):
        return x.shape[0] if isinstance(x, np.ndarray) and x.size > 1 else 1

    k = df[kwargs[FEATURE_SUMSQ]].apply(dimension).iloc[0]

    XX0 = np.zeros((k + 1, k + 1))
    XX0[1 : (k + 1), 1 : (k + 1)] = col_sum(df[kwargs[FEATURE_SUMSQ]])
    XX0[0, 0] = col_sum(df[kwargs[DENOMINATOR]])
    XX0[0, 1 : (k + 1)] = col_sum(df[kwargs[FEATURE]])
    XX0[1 : (k + 1), 0] = col_sum(df[kwargs[FEATURE]])

    Xy0 = np.zeros((k + 1, 1))
    Xy0[0,] = col_sum(df[kwargs[NUMERATOR]])
    Xy0[1 : (k + 1),] = np.atleast_2d(col_sum(df[kwargs[FEATURE_CROSS]])).reshape(-1, 1)

    try:
        b = np.matmul(np.linalg.inv(XX0), Xy0)
    except np.linalg.LinAlgError:
        b = np.zeros((k + 1, 1))
    out = b[1 : (k + 1)]
    if out.size == 1:
        out = out.item()

    outseries = Series(index=df.index, dtype=df[kwargs[FEATURE]].dtype)
    df[REGRESSION_PARAM] = outseries.apply(lambda x: out)
    return df


def point_estimate(df: Series, **kwargs) -> float:
    df = estimate_slope(df, **kwargs)
    point_estimate = df[kwargs[NUMERATOR]] / df[kwargs[DENOMINATOR]]

    if REGRESSION_PARAM in df:
        feature_mean = df[kwargs[FEATURE]].sum() / df[kwargs[DENOMINATOR]].sum()

        def lin_reg_point_estimate_delta(row: Series, feature_mean: float, **kwargs: Dict) -> Series:
            return dfmatmul(
                row[REGRESSION_PARAM], row[kwargs[FEATURE]] - feature_mean * row[kwargs[DENOMINATOR]], outer=False
            )

        return (
            point_estimate
            - df.apply(lin_reg_point_estimate_delta, feature_mean=feature_mean, axis=1, **kwargs)
            / df[kwargs[DENOMINATOR]]
        )

    return point_estimate


def lin_reg_variance_delta(row, **kwargs):
    y = row[kwargs[NUMERATOR]]
    n = row[kwargs[DENOMINATOR]]

    XX = unlist(row[kwargs[FEATURE_SUMSQ]])
    X = unlist(row[kwargs[FEATURE]])
    Xy = unlist(row[kwargs[FEATURE_CROSS]])

    sample_var = XX / n - dfmatmul(X / n, X / n)
    sample_cov = Xy / n - dfmatmul(X / n, y / n)
    b = np.atleast_2d(row[REGRESSION_PARAM])
    variance2 = np.matmul(np.transpose(b), np.matmul(sample_var, b)).item()
    variance3 = -2 * np.matmul(np.transpose(b), sample_cov).item()

    return variance2 + variance3


def variance(df: DataFrame, **kwargs) -> Series:
    variance1 = z_test_computer.variance(df, **kwargs)
    if kwargs[FEATURE] in df:
        computed_variances = variance1 + df.apply(lin_reg_variance_delta, axis=1, **kwargs)
        if (computed_variances < 0).any():
            raise ValueError("Computed variance is negative, please check sufficient " "statistics.")
        return computed_variances
    else:
        return variance1


def add_point_estimate_ci(df: DataFrame, **kwargs: Dict) -> DataFrame:
    return z_test_computer.add_point_estimate_ci(df, **kwargs)


def std_err(df: DataFrame, **kwargs: Dict) -> DataFrame:
    return z_test_computer.std_err(df, **kwargs)


def p_value(df: DataFrame, **kwargs: Dict) -> DataFrame:
    return z_test_computer.p_value(df, **kwargs)


def ci(df: DataFrame, alpha_column: str, **kwargs: Dict) -> DataFrame:
    return z_test_computer.ci(df, alpha_column, **kwargs)


def powered_effect(
    df: DataFrame,
    z_alpha: float,
    z_power: float,
    binary: bool,
    non_inferiority: bool,
    avg_column: float,
    var_column: float,
) -> Series:
    return z_test_computer.powered_effect(df, z_alpha, z_power, binary, non_inferiority, avg_column, var_column)


def required_sample_size(
    binary: Union[Series, bool],
    non_inferiority: Union[Series, bool],
    hypothetical_effect: Union[Series, float],
    control_avg: Union[Series, float],
    control_var: Union[Series, float],
    z_alpha: float = None,
    kappa: float = None,
    proportion_of_total: Union[Series, float] = None,
    z_power: float = None,
) -> Union[Series, float]:
    return z_test_computer.required_sample_size(
        binary,
        non_inferiority,
        hypothetical_effect,
        control_avg,
        control_var,
        z_alpha,
        kappa,
        proportion_of_total,
        z_power,
    )
