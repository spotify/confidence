from _warnings import warn
from typing import Iterable, Dict

from pandas import DataFrame, Series
from statsmodels.stats.multitest import multipletests

from spotify_confidence.analysis.constants import (
    BONFERRONI,
    BONFERRONI_ONLY_COUNT_TWOSIDED,
    PREFERENCE_TEST,
    TWO_SIDED,
    HOLM,
    HOMMEL,
    SIMES_HOCHBERG,
    SIDAK,
    HOLM_SIDAK,
    FDR_BH,
    FDR_BY,
    FDR_TSBH,
    FDR_TSBKY,
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    SPOT_1,
    SPOT_1_HOLM,
    SPOT_1_HOMMEL,
    SPOT_1_SIMES_HOCHBERG,
    SPOT_1_SIDAK,
    SPOT_1_HOLM_SIDAK,
    SPOT_1_FDR_BH,
    SPOT_1_FDR_BY,
    SPOT_1_FDR_TSBH,
    SPOT_1_FDR_TSBKY,
    NIM,
    NUMBER_OF_COMPARISONS,
    FINAL_EXPECTED_SAMPLE_SIZE,
    CORRECTION_METHOD,
    METHOD,
    IS_SIGNIFICANT,
    P_VALUE,
    ADJUSTED_ALPHA,
    ADJUSTED_P,
    ALPHA,
    INTERVAL_SIZE,
    ZTEST,
    ZTESTLINREG,
    CI_LOWER,
    CI_UPPER,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    PREFERENCE,
    ADJUSTED_ALPHA_POWER_SAMPLE_SIZE,
    CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO,
    ADJUSTED_POWER,
    POWER,
)
from spotify_confidence.analysis.frequentist.confidence_computers import confidence_computers


def get_num_comparisons(
    df: DataFrame,
    correction_method: str,
    number_of_level_comparisons: int,
    groupby: Iterable,
    metric_column: str,
    treatment_column: str,
    single_metric: bool,
    segments: Iterable,
) -> int:
    if correction_method == BONFERRONI:
        return max(
            1,
            number_of_level_comparisons * df.assign(_dummy_=1).groupby(groupby + ["_dummy_"], sort=False).ngroups,
        )
    elif correction_method == BONFERRONI_ONLY_COUNT_TWOSIDED:
        return max(
            number_of_level_comparisons
            * df.query(f'{PREFERENCE_TEST} == "{TWO_SIDED}"')
            .assign(_dummy_=1)
            .groupby(groupby + ["_dummy_"], sort=False)
            .ngroups,
            1,
        )
    elif correction_method in [
        HOLM,
        HOMMEL,
        SIMES_HOCHBERG,
        SIDAK,
        HOLM_SIDAK,
        FDR_BH,
        FDR_BY,
        FDR_TSBH,
        FDR_TSBKY,
    ]:
        return 1
    elif correction_method in [
        BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
        SPOT_1,
        SPOT_1_HOLM,
        SPOT_1_HOMMEL,
        SPOT_1_SIMES_HOCHBERG,
        SPOT_1_SIDAK,
        SPOT_1_HOLM_SIDAK,
        SPOT_1_FDR_BH,
        SPOT_1_FDR_BY,
        SPOT_1_FDR_TSBH,
        SPOT_1_FDR_TSBKY,
    ]:
        if metric_column is None or treatment_column is None:
            return max(
                1,
                number_of_level_comparisons
                * df[df[NIM].isnull()].assign(_dummy_=1).groupby(groupby + ["_dummy_"], sort=False).ngroups,
            )
        else:
            if single_metric:
                if df[df[NIM].isnull()].shape[0] > 0:
                    number_success_metrics = 1
                else:
                    number_success_metrics = 0
            else:
                number_success_metrics = df[df[NIM].isnull()].groupby(metric_column, sort=False).ngroups

            number_segments = (
                1
                if len(segments) == 0 or not all(item in df.index.names for item in segments)
                else df.groupby(segments, sort=False).ngroups
            )

            return max(1, number_of_level_comparisons * max(1, number_success_metrics) * number_segments)
    else:
        raise ValueError(f"Unsupported correction method: {correction_method}.")


def add_adjusted_p_and_is_significant(df: DataFrame, **kwargs: Dict) -> DataFrame:
    n_comparisons = kwargs[NUMBER_OF_COMPARISONS]
    if kwargs[FINAL_EXPECTED_SAMPLE_SIZE] is not None:
        if kwargs[CORRECTION_METHOD] not in [
            BONFERRONI,
            BONFERRONI_ONLY_COUNT_TWOSIDED,
            BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
            SPOT_1,
        ]:
            raise ValueError(
                f"{kwargs[CORRECTION_METHOD]} not supported for sequential tests. Use one of"
                f"{BONFERRONI}, {BONFERRONI_ONLY_COUNT_TWOSIDED}, "
                f"{BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY}, {SPOT_1}"
            )
        adjusted_alpha = compute_sequential_adjusted_alpha(df, **kwargs)
        df = df.merge(adjusted_alpha, left_index=True, right_index=True)
        df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
        df[P_VALUE] = None
        df[ADJUSTED_P] = None
    elif kwargs[CORRECTION_METHOD] in [
        HOLM,
        HOMMEL,
        SIMES_HOCHBERG,
        SIDAK,
        HOLM_SIDAK,
        FDR_BH,
        FDR_BY,
        FDR_TSBH,
        FDR_TSBKY,
        SPOT_1_HOLM,
        SPOT_1_HOMMEL,
        SPOT_1_SIMES_HOCHBERG,
        SPOT_1_SIDAK,
        SPOT_1_HOLM_SIDAK,
        SPOT_1_FDR_BH,
        SPOT_1_FDR_BY,
        SPOT_1_FDR_TSBH,
        SPOT_1_FDR_TSBKY,
    ]:
        if kwargs[CORRECTION_METHOD].startswith("spot-"):
            correction_method = kwargs[CORRECTION_METHOD][7:]
        else:
            correction_method = kwargs[CORRECTION_METHOD]
        df[ADJUSTED_ALPHA] = df[ALPHA] / n_comparisons
        is_significant, adjusted_p, _, _ = multipletests(
            pvals=df[P_VALUE], alpha=1 - kwargs[INTERVAL_SIZE], method=correction_method
        )
        df[ADJUSTED_P] = adjusted_p
        df[IS_SIGNIFICANT] = is_significant
    elif kwargs[CORRECTION_METHOD] in [
        BONFERRONI,
        BONFERRONI_ONLY_COUNT_TWOSIDED,
        BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
        SPOT_1,
    ]:
        df[ADJUSTED_ALPHA] = df[ALPHA] / n_comparisons
        df[ADJUSTED_P] = df[P_VALUE].map(lambda p: min(p * n_comparisons, 1))
        df[IS_SIGNIFICANT] = df[P_VALUE] < df[ADJUSTED_ALPHA]
    else:
        raise ValueError("Can't figure out which correction method to use :(")

    return df


def compute_sequential_adjusted_alpha(df: DataFrame, **kwargs: Dict) -> Series:
    if df[kwargs[METHOD]].isin([ZTEST, ZTESTLINREG]).all():
        return confidence_computers[ZTEST].compute_sequential_adjusted_alpha(df, **kwargs)
    else:
        raise NotImplementedError("Sequential testing is only supported for z-test and z-testlinreg")


def add_ci(df: DataFrame, **kwargs: Dict) -> DataFrame:
    lower, upper = confidence_computers[df[kwargs[METHOD]].values[0]].ci(df, ALPHA, **kwargs)

    if kwargs[CORRECTION_METHOD] in [
        HOLM,
        HOMMEL,
        SIMES_HOCHBERG,
        SPOT_1_HOLM,
        SPOT_1_HOMMEL,
        SPOT_1_SIMES_HOCHBERG,
    ] and all(df[PREFERENCE_TEST] != TWO_SIDED):
        if all(df[kwargs[METHOD]] == "z-test"):
            adjusted_lower, adjusted_upper = confidence_computers["z-test"].ci_for_multiple_comparison_methods(
                df, kwargs[CORRECTION_METHOD], alpha=1 - kwargs[INTERVAL_SIZE]
            )
        else:
            raise NotImplementedError(f"{kwargs[CORRECTION_METHOD]} is only supported for ZTests")
    elif kwargs[CORRECTION_METHOD] in [
        BONFERRONI,
        BONFERRONI_ONLY_COUNT_TWOSIDED,
        BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
        SPOT_1,
        SPOT_1_HOLM,
        SPOT_1_HOMMEL,
        SPOT_1_SIMES_HOCHBERG,
        SPOT_1_SIDAK,
        SPOT_1_HOLM_SIDAK,
        SPOT_1_FDR_BH,
        SPOT_1_FDR_BY,
        SPOT_1_FDR_TSBH,
        SPOT_1_FDR_TSBKY,
    ]:
        adjusted_lower, adjusted_upper = confidence_computers[df[kwargs[METHOD]].values[0]].ci(
            df, ADJUSTED_ALPHA, **kwargs
        )
    else:
        warn(f"Confidence intervals not supported for {kwargs[CORRECTION_METHOD]}")
        adjusted_lower = None
        adjusted_upper = None

    return (
        df.assign(**{CI_LOWER: lower})
        .assign(**{CI_UPPER: upper})
        .assign(**{ADJUSTED_LOWER: adjusted_lower})
        .assign(**{ADJUSTED_UPPER: adjusted_upper})
    )


def set_alpha_and_adjust_preference(df: DataFrame, **kwargs: Dict) -> DataFrame:
    alpha_0 = 1 - kwargs[INTERVAL_SIZE]
    return df.assign(
        **{
            ALPHA: df.apply(
                lambda row: 2 * alpha_0
                if kwargs[CORRECTION_METHOD] == SPOT_1 and row[PREFERENCE] != TWO_SIDED
                else alpha_0,
                axis=1,
            )
        }
    ).assign(**{ADJUSTED_ALPHA_POWER_SAMPLE_SIZE: lambda df: df[ALPHA] / kwargs[NUMBER_OF_COMPARISONS]})


def get_preference(df: DataFrame, correction_method: str):
    return TWO_SIDED if correction_method == SPOT_1 else df[PREFERENCE]


def add_adjusted_power(df: DataFrame, correction_method: str, metric_column: str, single_metric: bool) -> DataFrame:
    if correction_method in CORRECTION_METHODS_THAT_REQUIRE_METRIC_INFO:
        if metric_column is None:
            return df.assign(**{ADJUSTED_POWER: None})
        else:
            number_total_metrics = 1 if single_metric else df.groupby(metric_column, sort=False).ngroups
            if single_metric:
                if df[df[NIM].isnull()].shape[0] > 0:
                    number_success_metrics = 1
                else:
                    number_success_metrics = 0
            else:
                number_success_metrics = df[df[NIM].isnull()].groupby(metric_column, sort=False).ngroups

            number_guardrail_metrics = number_total_metrics - number_success_metrics
            power_correction = (
                number_guardrail_metrics if number_success_metrics == 0 else number_guardrail_metrics + 1
            )
            return df.assign(**{ADJUSTED_POWER: 1 - (1 - df[POWER]) / power_correction})
    else:
        return df.assign(**{ADJUSTED_POWER: df[POWER]})
