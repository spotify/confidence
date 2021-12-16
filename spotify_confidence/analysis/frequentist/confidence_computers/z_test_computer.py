from typing import Tuple, Union, Dict

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
)
from spotify_confidence.analysis.frequentist.sequential_bound_solver import bounds


def sequential_bounds(t: np.array, alpha: float, sides: int, state: DataFrame = None):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000, state=state)


def point_estimate(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    numerator = arg_dict[NUMERATOR]
    denominator = arg_dict[DENOMINATOR]
    if (df[denominator] == 0).any():
        raise ValueError("""Can't compute point estimate: denominator is 0""")
    return df[numerator] / df[denominator]


def variance(df: DataFrame, arg_dict: Dict[str, str]) -> float:
    numerator = arg_dict[NUMERATOR]
    denominator = arg_dict[DENOMINATOR]
    numerator_sumsq = arg_dict[NUMERATOR_SUM_OF_SQUARES]
    binary = df[numerator_sumsq] == df[numerator]
    if binary.all():
        # This equals row[POINT_ESTIMATE]*(1-row[POINT_ESTIMATE]) when the data is binary,
        # and also gives a robust fallback in case it's not
        variance = df[numerator_sumsq] / df[denominator] - df[POINT_ESTIMATE] ** 2
    else:
        variance = (df[numerator_sumsq] - np.power(df[numerator], 2) / df[denominator]) / (df[denominator] - 1)
    if (variance < 0).any():
        raise ValueError("Computed variance is negative. " "Please check your inputs.")
    return variance


def std_err(df: Series, arg_dict: Dict[str, str]) -> float:
    denominator = arg_dict[DENOMINATOR]
    return np.sqrt(df[VARIANCE + SFX1] / df[denominator + SFX1] + df[VARIANCE + SFX2] / df[denominator + SFX2])


def add_point_estimate_ci(df: Series, arg_dict: Dict[str, str]) -> Series:
    denominator = arg_dict[DENOMINATOR]
    interval_size = arg_dict[INTERVAL_SIZE]
    df[CI_LOWER], df[CI_UPPER] = _zconfint_generic(
        mean=df[POINT_ESTIMATE],
        std_mean=np.sqrt(df[VARIANCE] / df[denominator]),
        alpha=1 - interval_size,
        alternative=TWO_SIDED,
    )
    return df


def p_value(df: DataFrame, arg_dict: Dict[str, str]) -> Series:
    _, p_value = _zstat_generic(
        value1=df[POINT_ESTIMATE + SFX2],
        value2=df[POINT_ESTIMATE + SFX1],
        std_diff=df[STD_ERR],
        alternative=df[PREFERENCE_TEST].values[0],
        diff=df[NULL_HYPOTHESIS],
    )
    return p_value


def ci(df: DataFrame, alpha_column: str, arg_dict: Dict[str, str]) -> Tuple[Series, Series]:
    return _zconfint_generic(
        mean=df[DIFFERENCE], std_mean=df[STD_ERR], alpha=df[alpha_column], alternative=df[PREFERENCE_TEST].values[0]
    )


def achieved_power(df: DataFrame, mde: float, alpha: float, arg_dict: Dict[str, str]) -> DataFrame:
    denominator = arg_dict[DENOMINATOR]
    v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
    n1, n2 = df[denominator + SFX1], df[denominator + SFX2]

    var_pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

    return power_calculation(mde, var_pooled, alpha, n1, n2)


def compute_sequential_adjusted_alpha(df: DataFrame, arg_dict: Dict[str, str]):
    denominator = arg_dict[DENOMINATOR]
    final_expected_sample_size_column = arg_dict[FINAL_EXPECTED_SAMPLE_SIZE]
    ordinal_group_column = arg_dict[ORDINAL_GROUP_COLUMN]
    n_comparisons = arg_dict[NUMBER_OF_COMPARISONS]

    def adjusted_alphas_for_group(grp: DataFrame) -> Series:
        return (
            sequential_bounds(
                t=grp["sample_size_proportions"].values,
                alpha=grp[ALPHA].values[0] / n_comparisons,
                sides=2 if (grp[PREFERENCE_TEST] == TWO_SIDED).all() else 1,
            )
            .df.set_index(grp.index)
            .assign(
                **{
                    ADJUSTED_ALPHA: lambda df: df.apply(
                        lambda row: 2 * (1 - st.norm.cdf(row["zb"]))
                        if (grp[PREFERENCE_TEST] == TWO_SIDED).all()
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
        .apply(adjusted_alphas_for_group)[ADJUSTED_ALPHA],
        name=ADJUSTED_ALPHA,
    )


def ci_for_multiple_comparison_methods(
    df: DataFrame,
    correction_method: str,
    alpha: float,
    w: float = 1.0,
) -> Tuple[Union[Series, float], Union[Series, float]]:
    if TWO_SIDED in df[PREFERENCE_TEST]:
        raise ValueError(
            "CIs can only be produced for one-sided tests when other multiple test corrections "
            "methods than bonferroni are applied"
        )
    m_scal = len(df)
    num_significant = sum(df[IS_SIGNIFICANT])
    r = m_scal - num_significant

    def _aw(W: float, alpha: float, m_scal: float, r: int):
        return alpha * (1 - (1 - W) * (m_scal - r) / m_scal)

    def _bw(W: float, alpha: float, m_scal: float, r: int):
        return 1 - (1 - alpha) / np.power((1 - (1 - W) * (1 - np.power((1 - alpha), (1 / m_scal)))), (m_scal - r))

    if correction_method in [HOLM, SPOT_1_HOLM]:
        adjusted_alpha_rej_equal_m = 1 - alpha / m_scal
        adjusted_alpha_rej_less_m = 1 - (1 - w) * (alpha / m_scal)
        adjusted_alpha_accept = 1 - _aw(w, alpha, m_scal, r) / r if r != 0 else 0
    elif correction_method in [HOMMEL, SIMES_HOCHBERG, SPOT_1_HOMMEL, SPOT_1_SIMES_HOCHBERG]:
        adjusted_alpha_rej_equal_m = np.power((1 - alpha), (1 / m_scal))
        adjusted_alpha_rej_less_m = 1 - (1 - w) * (1 - np.power((1 - alpha), (1 / m_scal)))
        adjusted_alpha_accept = 1 - _bw(w, alpha, m_scal, r) / r if r != 0 else 0
    else:
        raise ValueError(
            "CIs not supported for correction method. "
            f"Supported methods: {HOMMEL}, {HOLM}, {SIMES_HOCHBERG},"
            f"{SPOT_1_HOLM}, {SPOT_1_HOMMEL} and {SPOT_1_SIMES_HOCHBERG}"
        )

    def _compute_ci_for_row(row: Series) -> Tuple[float, float]:
        if row[IS_SIGNIFICANT] and num_significant == m_scal:
            alpha_adj = adjusted_alpha_rej_equal_m
        elif row[IS_SIGNIFICANT] and num_significant < m_scal:
            alpha_adj = adjusted_alpha_rej_less_m
        else:
            alpha_adj = adjusted_alpha_accept

        ci_sign = -1 if row[PREFERENCE_TEST] == "larger" else 1
        bound1 = row[DIFFERENCE] + ci_sign * st.norm.ppf(alpha_adj) * row[STD_ERR]
        if ci_sign == -1:
            bound2 = max(row[NULL_HYPOTHESIS], bound1)
        else:
            bound2 = min(row[NULL_HYPOTHESIS], bound1)

        bound = bound2 if row[IS_SIGNIFICANT] else bound1

        lower = bound if row[PREFERENCE_TEST] == "larger" else -np.inf
        upper = bound if row[PREFERENCE_TEST] == "smaller" else np.inf

        row[ADJUSTED_LOWER] = lower
        row[ADJUSTED_UPPER] = upper

        return row

    ci_df = df.apply(_compute_ci_for_row, axis=1)[[ADJUSTED_LOWER, ADJUSTED_UPPER]]

    return ci_df[ADJUSTED_LOWER], ci_df[ADJUSTED_UPPER]


def ci_width(
    z_alpha, binary, non_inferiority, hypothetical_effect, control_avg, control_var, control_count, treatment_count
) -> Union[Series, float]:
    treatment_var = _get_hypothetical_treatment_var(
        binary, non_inferiority, control_avg, control_var, hypothetical_effect
    )
    _, std_err = st.stats._unequal_var_ttest_denom(control_var, control_count, treatment_var, treatment_count)
    return 2 * z_alpha * std_err


def powered_effect(
    df: DataFrame,
    z_alpha: float,
    z_power: float,
    binary: bool,
    non_inferiority: bool,
    avg_column: float,
    var_column: float,
) -> Series:

    if binary and not non_inferiority:
        effect = df.apply(
            lambda row: _search_MDE_binary_local_search(
                control_avg=row[avg_column],
                control_var=row[var_column],
                non_inferiority=False,
                kappa=row["kappa"],
                proportion_of_total=row["proportion_of_total"],
                current_number_of_units=row["current_number_of_units"],
                z_alpha=z_alpha,
                z_power=z_power,
            )[0],
            axis=1,
        )
    else:
        treatment_var = _get_hypothetical_treatment_var(
            binary_metric=binary,
            non_inferiority=df[NIM] is not None,
            control_avg=df[avg_column],
            control_var=df[var_column],
            hypothetical_effect=0,
        )
        n2_partial = np.power((z_alpha + z_power), 2) * (df[var_column] / df["kappa"] + treatment_var)
        effect = np.sqrt(
            (1 / (df["current_number_of_units"] * df["proportion_of_total"])) * (n2_partial + df["kappa"] * n2_partial)
        )

    return effect


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

    if kappa is None:
        raise ValueError("kappa is None, must be postive float")
    if proportion_of_total is None:
        raise ValueError("proportion_of_total is None, must be between 0 and 1")

    treatment_var = np.vectorize(_get_hypothetical_treatment_var)(
        binary, non_inferiority, control_avg, control_var, hypothetical_effect
    )

    n2 = _treatment_group_sample_size(
        z_alpha=z_alpha,
        z_power=z_power,
        hypothetical_effect=hypothetical_effect,
        control_var=control_var,
        treatment_var=treatment_var,
        kappa=kappa,
    )
    required_sample_size = np.ceil((n2 + n2 * kappa) / proportion_of_total)
    return required_sample_size


def _search_MDE_binary_local_search(
    control_avg: float,
    control_var: float,
    non_inferiority: bool,
    kappa: float,
    proportion_of_total: float,
    current_number_of_units: float,
    z_alpha: float = None,
    z_power: float = None,
):
    def f(x):
        return _find_current_powered_effect(
            hypothetical_effect=x,
            control_avg=control_avg,
            control_var=control_var,
            binary=True,
            non_inferiority=non_inferiority,
            kappa=kappa,
            proportion_of_total=proportion_of_total,
            current_number_of_units=current_number_of_units,
            z_alpha=z_alpha,
            z_power=z_power,
        )

    max_val = 1 - control_avg
    min_val = min(10e-9, max_val)

    if min_val == max_val:
        # corner case that crashes the optimizer
        return min_val, f(min_val)

    max_iter = 100  # max number of iterations before falling back to slow grid search

    # we stop immediately if a solution was found that is "good enough". A threshold of
    # 1 indicates that
    # the approximated number of units (based on the current effect candidate) is off by
    # at most 1.0
    goodness_threshold = 1.0

    curr_iter = 0
    best_x = None
    best_fun = float("inf")

    bounds_queue = [(min_val, max_val)]

    while curr_iter < max_iter and best_fun > goodness_threshold:

        # take next value from queue
        interval = bounds_queue.pop(0)

        # conduct a bounded local search, using a very small tol value improved
        # performance during tests
        # result = optimize.minimize_scalar(f, bounds=(interval[0], interval[1]),
        # method='bounded', tol=10e-14)
        result = optimize.minimize_scalar(
            f, bounds=(interval[0], interval[1]), method="bounded", options={"xatol": 10e-14, "maxiter": 50}
        )

        if result.fun < best_fun:
            best_x = result.x
            best_fun = result.fun

        curr_iter += 1

        # add new bounds to the queue
        interval_split = (interval[0] + interval[1]) / 2
        bounds_queue.append((interval[0], interval_split))
        bounds_queue.append((interval_split, interval[1]))

    if best_fun <= goodness_threshold:
        return best_x, best_fun
    else:  # check if grid search finds a better solution
        alt_result_x, alt_result_fun = _search_MDE_binary(
            control_avg,
            control_var,
            non_inferiority,
            kappa,
            proportion_of_total,
            current_number_of_units,
            z_alpha,
            z_power,
            return_cost_val=True,
        )

        return (alt_result_x, alt_result_fun) if alt_result_fun < best_fun else (best_x, best_fun)


def _search_MDE_binary(
    control_avg: float,
    control_var: float,
    non_inferiority: bool,
    kappa: float,
    proportion_of_total: float,
    current_number_of_units: float,
    z_alpha: float = None,
    z_power: float = None,
    return_cost_val=False,
):
    candidate_effects = np.linspace(10e-9, 1 - control_avg, num=2000)
    for i in range(2):
        test = []
        for effect in candidate_effects:
            test.append(
                _find_current_powered_effect(
                    hypothetical_effect=effect,
                    control_avg=control_avg,
                    control_var=control_var,
                    binary=True,
                    non_inferiority=non_inferiority,
                    kappa=kappa,
                    proportion_of_total=proportion_of_total,
                    current_number_of_units=current_number_of_units,
                    z_alpha=z_alpha,
                    z_power=z_power,
                )
            )

        test = np.array(test)
        index = [idx for idx, element in enumerate(test) if element == test.min()]
        if len(index) != 1:
            index = [index[int(np.ceil(len(index) / 2))]]
        if i == 0:
            if index[0] == 9999:
                return np.inf
            lower_effect_bound = 10e-9 if index[0] == 0 else candidate_effects[index[0] - 1]
            candidate_effects = np.linspace(lower_effect_bound, candidate_effects[index[0]], num=10000)

    index = [idx for idx, element in enumerate(test) if element == test.min()]

    return candidate_effects[index[0]], test[index[0]] if return_cost_val else candidate_effects[index[0]]


def _treatment_group_sample_size(
    z_alpha: float,
    z_power: float,
    hypothetical_effect: float,
    control_var: float,
    treatment_var: float,
    kappa: float,
) -> float:
    return np.ceil(np.power((z_alpha + z_power) / abs(hypothetical_effect), 2) * (control_var / kappa + treatment_var))


def _find_current_powered_effect(
    hypothetical_effect: float,
    control_avg: float,
    control_var: float,
    binary: bool,
    non_inferiority: bool,
    kappa: float,
    proportion_of_total: float,
    current_number_of_units: float,
    z_power: float = None,
    z_alpha: float = None,
) -> float:

    treatment_var = _get_hypothetical_treatment_var(
        binary_metric=binary,
        non_inferiority=non_inferiority,
        control_avg=control_avg,
        control_var=control_var,
        hypothetical_effect=hypothetical_effect,
    )
    n2 = _treatment_group_sample_size(
        z_alpha,
        z_power,
        hypothetical_effect,
        control_var,
        treatment_var,
        kappa,
    )

    return np.power(current_number_of_units - ((n2 + n2 * kappa) / proportion_of_total), 2)


def _get_hypothetical_treatment_var(
    binary_metric: bool,
    non_inferiority: bool,
    control_avg: float,
    control_var: float,
    hypothetical_effect: float,
) -> float:
    if binary_metric and not non_inferiority:
        # For binary metrics, the variance can be derived from the average. However,
        # we do *not* do this for
        # non-inferiority tests because for non-inferiority tests, the basic assumption
        # is that the
        # mean of the control group and treatment group are identical.
        return (control_avg + hypothetical_effect) * (1 - (control_avg + hypothetical_effect))
    else:
        return control_var
