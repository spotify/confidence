from typing import Any, Optional, Tuple

import numpy as np
from pandas import DataFrame, Series

from spotify_confidence.analysis.constants import BOOTSTRAPS, CI_LOWER, CI_UPPER, INTERVAL_SIZE, SFX1, SFX2


def point_estimate(df: DataFrame, **kwargs: Any) -> float:
    bootstrap_samples = kwargs[BOOTSTRAPS]
    return df[bootstrap_samples].map(lambda a: a.mean())


def variance(df: Series, **kwargs: Any) -> float:
    bootstrap_samples = kwargs[BOOTSTRAPS]
    variance = df[bootstrap_samples].map(lambda a: a.var())

    if (variance < 0).any():
        raise ValueError("Computed variance is negative. Please check your inputs.")
    return variance


def std_err(row: Series, **kwargs: Any) -> Optional[float]:
    return None


def add_point_estimate_ci(df: DataFrame, **kwargs: Any) -> DataFrame:
    bootstrap_samples = kwargs[BOOTSTRAPS]
    interval_size = kwargs[INTERVAL_SIZE]
    df[CI_LOWER] = df[bootstrap_samples].map(lambda a: np.percentile(a, 100 * (1 - interval_size) / 2))
    df[CI_UPPER] = df[bootstrap_samples].map(lambda a: np.percentile(a, 100 * (1 - (1 - interval_size) / 2)))
    return df


def p_value(row, **kwargs: Any) -> float:
    return -1


def ci(df, alpha_column: str, **kwargs: Any) -> Tuple[Series, Series]:
    bootstrap_samples = kwargs[BOOTSTRAPS]
    lower = df.apply(
        lambda row: np.percentile(
            row[bootstrap_samples + SFX2] - row[bootstrap_samples + SFX1], 100 * row[alpha_column] / 2
        ),
        axis=1,
    )
    upper = df.apply(
        lambda row: np.percentile(
            row[bootstrap_samples + SFX2] - row[bootstrap_samples + SFX1], 100 * (1 - row[alpha_column] / 2)
        ),
        axis=1,
    )
    return lower, upper


def achieved_power(df: DataFrame, mde: float, alpha: float) -> Optional[DataFrame]:
    return None
