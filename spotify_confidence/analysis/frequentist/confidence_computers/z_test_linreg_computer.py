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
    ORIGINAL_POINT_ESTIMATE,
    ORIGINAL_VARIANCE,
)
from spotify_confidence.analysis.frequentist.confidence_computers import z_test_computer


def estimate_slope(df, arg_dict: Dict) -> DataFrame:
    if arg_dict[FEATURE] not in df:
        return df

    def col_sum(x):
        return reduce(lambda x, y: x + y, x)

    def dimension(x):
        return x.shape[0] if isinstance(x, np.ndarray) and x.size > 1 else 1

    k = df[arg_dict[FEATURE_SUMSQ]].apply(dimension).iloc[0]

    XX0 = np.zeros((k + 1, k + 1))
    XX0[1 : (k + 1), 1 : (k + 1)] = col_sum(df[arg_dict[FEATURE_SUMSQ]])
    XX0[0, 0] = col_sum(df[arg_dict[DENOMINATOR]])
    XX0[0, 1 : (k + 1)] = col_sum(df[arg_dict[FEATURE]])
    XX0[1 : (k + 1), 0] = col_sum(df[arg_dict[FEATURE]])

    Xy0 = np.zeros((k + 1, 1))
    Xy0[
        0,
    ] = col_sum(df[arg_dict[NUMERATOR]])
    Xy0[1 : (k + 1),] = np.atleast_2d(
        col_sum(df[arg_dict[FEATURE_CROSS]])
    ).reshape(-1, 1)

    b = np.matmul(np.linalg.inv(XX0), Xy0)
    out = b[1 : (k + 1)]
    if out.size == 1:
        out = out.item()

    outseries = Series(index=df.index, dtype=df[arg_dict[FEATURE]].dtype)
    df[REGRESSION_PARAM] = outseries.apply(lambda x: out)
    return df


def point_estimate(df: Series, arg_dict) -> float:
    df = estimate_slope(df, arg_dict)
    point_estimate = df[arg_dict[NUMERATOR]] / df[arg_dict[DENOMINATOR]]

    if REGRESSION_PARAM in df:

        def lin_reg_point_estimate_delta(row: Series, arg_dict: Dict) -> Series:
            return dfmatmul(row[REGRESSION_PARAM], row[arg_dict[FEATURE]], outer=False)

        return (
            point_estimate
            - df.apply(lin_reg_point_estimate_delta, arg_dict=arg_dict, axis=1) / df[arg_dict[DENOMINATOR]]
        )

    return point_estimate


def lin_reg_variance_delta(row, arg_dict):
    y = row[arg_dict[NUMERATOR]]
    n = row[arg_dict[DENOMINATOR]]

    XX = unlist(row[arg_dict[FEATURE_SUMSQ]])
    X = unlist(row[arg_dict[FEATURE]])
    Xy = unlist(row[arg_dict[FEATURE_CROSS]])

    sample_var = XX / n - dfmatmul(X / n, X / n)
    sample_cov = Xy / n - dfmatmul(X / n, y / n)
    b = np.atleast_2d(row[REGRESSION_PARAM])
    variance2 = np.matmul(np.transpose(b), np.matmul(sample_var, b)).item()
    variance3 = -2 * np.matmul(np.transpose(b), sample_cov).item()

    return variance2 + variance3


def variance(df: DataFrame, arg_dict) -> Series:
    variance1 = z_test_computer.variance(df, arg_dict)

    if arg_dict[FEATURE] in df:
        return variance1 + df.apply(lin_reg_variance_delta, arg_dict=arg_dict, axis=1)
    else:
        return variance1


def add_point_estimate_ci(df: DataFrame, arg_dict: Dict) -> DataFrame:
    df = df.assign(**{ORIGINAL_POINT_ESTIMATE: z_test_computer.point_estimate(df, arg_dict)}).assign(
        **{ORIGINAL_VARIANCE: z_test_computer.variance(df, arg_dict)}
    )

    return z_test_computer.add_point_estimate_ci(df, arg_dict)


def std_err(df: DataFrame, arg_dict: Dict) -> DataFrame:
    return z_test_computer.std_err(df, arg_dict)


def p_value(df: DataFrame, arg_dict: Dict) -> DataFrame:
    return z_test_computer.p_value(df, arg_dict)


def ci(df: DataFrame, alpha_column: str, arg_dict: Dict) -> DataFrame:
    return z_test_computer.ci(df, alpha_column, arg_dict)


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
