from typing import Tuple, Union, Dict, Iterable

import numpy as np
from pandas import DataFrame, Series
from scipy import optimize
from scipy import stats as st

from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import (
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    DENOMINATOR,
    INTERVAL_SIZE,
    FINAL_EXPECTED_SAMPLE_SIZE,
    ORDINAL_GROUP_COLUMN,
    POINT_ESTIMATE,
    CI_LOWER,
    CI_UPPER,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    VARIANCE,
    NUMBER_OF_COMPARISONS,
    TWO_SIDED,
    SFX2,
    SFX1,
    STD_ERR,
    PREFERENCE_TEST,
    NULL_HYPOTHESIS,
    DIFFERENCE,
    ALPHA,
    IS_SIGNIFICANT,
    HOLM,
    SPOT_1_HOLM,
    HOMMEL,
    SIMES_HOCHBERG,
    SPOT_1_HOMMEL,
    SPOT_1_SIMES_HOCHBERG,
    NIM,
    ADJUSTED_ALPHA,
    NUMBER_OF_COMPARISONS_VALIDATION,
    ADJUSTED_ALPHA_VALIDATION,
    PREFERRED_DIRECTION_COLUMN_DEFAULT,
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    PREFERENCE_DICT,
)
from spotify_confidence.analysis.frequentist.sequential_bound_solver import bounds


def sequential_bounds(t: np.array, alpha: float, sides: int, state: DataFrame = None):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000, state=state)


def sample_ratio_test(df: DataFrame, arg_dict: dict) -> Tuple[float, DataFrame]:
    n_tot = df[arg_dict[DENOMINATOR]].sum()
    expected_proportions = df[arg_dict[NUMERATOR]]
    observed_proportions = df[arg_dict[DENOMINATOR]] / n_tot
    sq_diff = np.power(observed_proportions - expected_proportions, 2)

    chi2_stat = n_tot * sq_diff.divide(expected_proportions).sum()
    deg_freedom = df.shape[0] - 1
    p_value = 1 - st.chi2.cdf(chi2_stat, deg_freedom)

    return p_value


def p_value(df: DataFrame, arg_dict: Dict[str, str], validation: bool = True) -> Series:
    return float('nan')


def ci(df: DataFrame, alpha_column: str, arg_dict: Dict[str, str]) -> Tuple[Series, Series]:
    return (float('nan'), float('nan'))


def point_estimate(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    return float('nan')


def variance(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    return float('nan')


def add_point_estimate_ci(df: Series, arg_dict: Dict[str, str]) -> Series:
    df[CI_LOWER] = float('nan')
    df[CI_UPPER] = float('nan')
    return df


def std_err(df: Series, arg_dict: Dict[str, str]) -> float:
    return float('nan')


def compute_sequential_adjusted_alpha(df: DataFrame, arg_dict: Dict[str, str], validation: bool):
    denominator = arg_dict[DENOMINATOR]
    final_expected_sample_size_column = arg_dict[FINAL_EXPECTED_SAMPLE_SIZE]
    ordinal_group_column = arg_dict[ORDINAL_GROUP_COLUMN]
    n_comparisons = arg_dict[NUMBER_OF_COMPARISONS if not validation else NUMBER_OF_COMPARISONS_VALIDATION]

    def adjusted_alphas_for_group(grp: DataFrame, validation: bool) -> Series:
        return (
            sequential_bounds(
                t=grp["sample_size_proportions"].values,
                alpha=grp[ALPHA].values[0] / n_comparisons,
                sides=2 if (grp[PREFERENCE_TEST] == TWO_SIDED).all() and not validation else 1,
            )
            .df.set_index(grp.index)
            .assign(
                **{
                    ADJUSTED_ALPHA: lambda df: df.apply(
                        lambda row: 2 * (1 - st.norm.cdf(row["zb"]))
                        if not validation and (grp[PREFERENCE_TEST] == TWO_SIDED).all()
                        else 1 - st.norm.cdf(row["zb"]),
                        axis=1,
                    )
                }
            )
        )[["zb", ADJUSTED_ALPHA]]

    groups_except_ordinal = [column for column in df.index.names if column != ordinal_group_column]
    max_sample_size_by_group = (
        (
            df[["current_total_" + denominator, final_expected_sample_size_column]]
            .groupby(groups_except_ordinal, sort=False)
            .max()
            .max(axis=1)
        )
        if len(groups_except_ordinal) > 0
        else (df[["current_total_" + denominator, final_expected_sample_size_column]].max().max())
    )
    sample_size_proportions = Series(
        data=df.groupby(df.index.names, sort=False)["current_total_" + denominator].first() / max_sample_size_by_group,
        name="sample_size_proportions",
    )

    return Series(
        data=df.groupby(df.index.names, sort=False)[[ALPHA, PREFERENCE_TEST]]
        .first()
        .merge(sample_size_proportions, left_index=True, right_index=True)
        .assign(_sequential_dummy_index_=1)
        .groupby(groups_except_ordinal + ["_sequential_dummy_index_"], sort=False)[
            ["sample_size_proportions", PREFERENCE_TEST, ALPHA]
        ]
        .apply(adjusted_alphas_for_group, validation=validation)[ADJUSTED_ALPHA],
        name=ADJUSTED_ALPHA if not validation else ADJUSTED_ALPHA_VALIDATION,
    )
