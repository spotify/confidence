from typing import Tuple, Union

import numpy as np
from pandas import DataFrame, Series
from scipy import optimize
from scipy import stats as st
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic

from spotify_confidence.analysis.confidence_utils import power_calculation
from spotify_confidence.analysis.constants import (
    POINT_ESTIMATE,
    CI_LOWER,
    CI_UPPER,
    VARIANCE,
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
    ADJUSTED_ALPHA_POWER_SAMPLE_SIZE,
    ADJUSTED_POWER,
    ALTERNATIVE_HYPOTHESIS,
    POWERED_EFFECT,
    POWERED_EFFECT_METRIC,
    REQUIRED_SAMPLE_SIZE,
    REQUIRED_SAMPLE_SIZE_METRIC,
)
from spotify_confidence.analysis.frequentist.sequential_bound_solver import bounds


def sequential_bounds(t: np.array, alpha: float, sides: int):
    return bounds(t, alpha, rho=2, ztrun=8, sides=sides, max_nints=1000)


class ZTestComputer(object):
    def __init__(self, numerator, numerator_sumsq, denominator, ordinal_group_column, interval_size):
        self._numerator = numerator
        self._numerator_sumsq = numerator_sumsq
        self._denominator = denominator
        self._ordinal_group_column = ordinal_group_column
        self._interval_size = interval_size

    def _point_estimate(self, row: Series) -> float:
        if row[self._denominator] == 0:
            raise ValueError("""Can't compute point estimate: denominator is 0""")
        return row[self._numerator] / row[self._denominator]

    def _variance(self, row: Series) -> float:
        variance = row[self._numerator_sumsq] / row[self._denominator] - row[POINT_ESTIMATE] ** 2
        if variance < 0:
            raise ValueError("Computed variance is negative. " "Please check your inputs.")
        return variance

    def _std_err(self, row: Series) -> float:
        return np.sqrt(
            row[VARIANCE + SFX1] / row[self._denominator + SFX1] + row[VARIANCE + SFX2] / row[self._denominator + SFX2]
        )

    def _add_point_estimate_ci(self, row: Series) -> Series:
        row[CI_LOWER], row[CI_UPPER] = _zconfint_generic(
            mean=row[POINT_ESTIMATE],
            std_mean=np.sqrt(row[VARIANCE] / row[self._denominator]),
            alpha=1 - self._interval_size,
            alternative=TWO_SIDED,
        )
        return row

    def _p_value(self, row: Series) -> float:
        _, p_value = _zstat_generic(
            value1=row[POINT_ESTIMATE + SFX2],
            value2=row[POINT_ESTIMATE + SFX1],
            std_diff=row[STD_ERR],
            alternative=row[PREFERENCE_TEST],
            diff=row[NULL_HYPOTHESIS],
        )
        return p_value

    def _ci(self, row: Series, alpha_column: str) -> Tuple[float, float]:
        return _zconfint_generic(
            mean=row[DIFFERENCE], std_mean=row[STD_ERR], alpha=row[alpha_column], alternative=row[PREFERENCE_TEST]
        )

    def _achieved_power(self, df: DataFrame, mde: float, alpha: float) -> DataFrame:
        v1, v2 = df[VARIANCE + SFX1], df[VARIANCE + SFX2]
        n1, n2 = df[self._denominator + SFX1], df[self._denominator + SFX2]

        var_pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)

        return power_calculation(mde, var_pooled, alpha, n1, n2)

    def _compute_sequential_adjusted_alpha(
        self,
        df: DataFrame,
        final_expected_sample_size_column: str,
        filtered_sufficient_statistics: DataFrame,
        num_comparisons: int,
    ):
        total_sample_size = (
            filtered_sufficient_statistics.groupby(df.index.names)
            .agg({self._denominator: sum, final_expected_sample_size_column: np.mean})
            .rename(columns={self._denominator: f"total_{self._denominator}"})
        )
        groups_except_ordinal = [column for column in df.index.names if column != self._ordinal_group_column]
        max_sample_size_by_group = (
            total_sample_size[f"total_{self._denominator}"].max()
            if len(groups_except_ordinal) == 0
            else total_sample_size.groupby(groups_except_ordinal)[f"total_{self._denominator}"].max()
        )

        if type(max_sample_size_by_group) is not Series:
            total_sample_size = total_sample_size.assign(
                **{f"total_{self._denominator}_max": max_sample_size_by_group}
            )
        else:
            total_sample_size = total_sample_size.merge(
                right=max_sample_size_by_group, left_index=True, right_index=True, suffixes=("", "_max")
            )

        total_sample_size = total_sample_size.assign(
            final_expected_sample_size=lambda df: df[
                [f"total_{self._denominator}_max", final_expected_sample_size_column]
            ].max(axis=1)
        ).assign(
            sample_size_proportions=lambda df: df["total_" + self._denominator] / df["final_expected_sample_size"]
        )

        def adjusted_alphas_for_group(grp: DataFrame) -> Series:
            return (
                sequential_bounds(
                    t=grp["sample_size_proportions"].values,
                    alpha=grp[ALPHA].values[0] / num_comparisons,
                    sides=2 if (grp[PREFERENCE_TEST] == TWO_SIDED).all() else 1,
                )
                .df.set_index(grp.index)
                .assign(
                    adjusted_alpha=lambda df: df.apply(
                        lambda row: 2 * (1 - st.norm.cdf(row["zb"]))
                        if (grp[PREFERENCE_TEST] == TWO_SIDED).all()
                        else 1 - st.norm.cdf(row["zb"]),
                        axis=1,
                    )
                )
            )[["zb", "adjusted_alpha"]]

        return (
            df.merge(total_sample_size, left_index=True, right_index=True)
            .groupby(groups_except_ordinal + ["level_1", "level_2"])[
                ["sample_size_proportions", PREFERENCE_TEST, ALPHA]
            ]
            .apply(adjusted_alphas_for_group)
            .reset_index()
            .set_index(df.index.names)
        )["adjusted_alpha"]

    def _ci_for_multiple_comparison_methods(
        self,
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

            return lower, upper

        return df.apply(_compute_ci_for_row, axis=1)

    def _powered_effect(
        self,
        df: Series,
        kappa: float,
        proportion_of_total: float,
        z_alpha: float,
        z_power: float,
        binary: bool,
        current_number_of_units: int,
        non_inferiority: bool,
    ) -> Series:

        if binary and not non_inferiority:
            effect = self._search_MDE_binary_local_search(
                control_avg=df[POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                non_inferiority=False,
                kappa=kappa,
                proportion_of_total=proportion_of_total,
                current_number_of_units=current_number_of_units,
                z_alpha=z_alpha,
                z_power=z_power,
            )[0]
        else:
            treatment_var = self._get_hypothetical_treatment_var(
                binary_metric=binary,
                non_inferiority=df[NIM] is not None,
                control_avg=df[POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                hypothetical_effect=0,
            )
            n2_partial = np.power((z_alpha + z_power), 2) * (df[VARIANCE + SFX1] / kappa + treatment_var)
            effect = np.sqrt((1 / (current_number_of_units * proportion_of_total)) * (n2_partial + kappa * n2_partial))

        return effect

    def _required_sample_size(
        self,
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

        treatment_var = np.vectorize(self._get_hypothetical_treatment_var)(
            binary, non_inferiority, control_avg, control_var, hypothetical_effect
        )

        n2 = self._treatment_group_sample_size(
            z_alpha=z_alpha,
            z_power=z_power,
            hypothetical_effect=hypothetical_effect,
            control_var=control_var,
            treatment_var=treatment_var,
            kappa=kappa,
        )
        required_sample_size = np.ceil((n2 + n2 * kappa) / proportion_of_total)
        return required_sample_size

    def _powered_effect_and_required_sample_size(
        self,
        row: Series,
    ) -> Series:
        z_alpha = st.norm.ppf(
            1 - row[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE] / (2 if row[PREFERENCE_TEST] == TWO_SIDED else 1)
        )
        z_power = st.norm.ppf(row[ADJUSTED_POWER])
        n1, n2 = row[self._denominator + SFX1], row[self._denominator + SFX2]
        kappa = n1 / n2
        binary = row[self._numerator_sumsq + SFX1] == row[self._numerator + SFX1]
        current_number_of_units = n1 + n2
        proportion_of_total = current_number_of_units / row[f"current_total_{self._denominator}"]

        if isinstance(row[NIM], float):
            non_inferiority = not np.isnan(row[NIM])
        elif row[NIM] is None:
            non_inferiority = row[NIM] is not None
        else:
            raise ValueError("NIM has to be type float or None.")

        row[POWERED_EFFECT] = self._powered_effect(
            df=row,
            kappa=kappa,
            proportion_of_total=1,
            z_alpha=z_alpha,
            z_power=z_power,
            binary=binary,
            current_number_of_units=current_number_of_units,
            non_inferiority=non_inferiority,
        )

        row[POWERED_EFFECT_METRIC] = self._powered_effect(
            df=row,
            kappa=kappa,
            proportion_of_total=proportion_of_total,
            z_alpha=z_alpha,
            z_power=z_power,
            binary=binary,
            current_number_of_units=current_number_of_units,
            non_inferiority=non_inferiority,
        )

        if ALTERNATIVE_HYPOTHESIS in row and NULL_HYPOTHESIS in row and row[ALTERNATIVE_HYPOTHESIS] is not None:
            row[REQUIRED_SAMPLE_SIZE] = self._required_sample_size(
                proportion_of_total=1,
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                hypothetical_effect=row[ALTERNATIVE_HYPOTHESIS] - row[NULL_HYPOTHESIS],
                control_avg=row[POINT_ESTIMATE + SFX1],
                control_var=row[VARIANCE + SFX1],
                kappa=kappa,
            )
            row[REQUIRED_SAMPLE_SIZE_METRIC] = self._required_sample_size(
                proportion_of_total=proportion_of_total,
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                hypothetical_effect=row[ALTERNATIVE_HYPOTHESIS] - row[NULL_HYPOTHESIS],
                control_avg=row[POINT_ESTIMATE + SFX1],
                control_var=row[VARIANCE + SFX1],
                kappa=kappa,
            )
        else:
            row[REQUIRED_SAMPLE_SIZE] = None

        return row

    def _currently_powered_effect(
        self,
        control_avg: float,
        control_var: float,
        binary_metric: bool,
        non_inferiority: bool = False,
        power: float = None,
        alpha: float = None,
        kappa: float = None,
        proportion_of_total: float = None,
        current_number_of_units: float = None,
    ):
        z_alpha = st.norm.ppf(1 - alpha)
        z_power = st.norm.ppf(power)

        if binary_metric and not non_inferiority:
            effect = self._search_MDE_binary_local_search(
                control_avg=control_avg,
                control_var=control_var,
                non_inferiority=non_inferiority,
                kappa=kappa,
                proportion_of_total=proportion_of_total,
                current_number_of_units=current_number_of_units,
                z_alpha=z_alpha,
                z_power=z_power,
            )[0]
        else:
            treatment_var = self._get_hypothetical_treatment_var(
                binary_metric, non_inferiority, control_avg, control_var, hypothetical_effect=0
            )
            n2_partial = np.power((z_alpha + z_power), 2) * (control_var / kappa + treatment_var)
            effect = np.sqrt((1 / (current_number_of_units * proportion_of_total)) * (n2_partial + kappa * n2_partial))

        return effect

    def _search_MDE_binary_local_search(
        self,
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
            return self.find_current_powered_effect(
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
            alt_result_x, alt_result_fun = self.search_MDE_binary(
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

    def search_MDE_binary(
        self,
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
                    self.find_current_powered_effect(
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
        self,
        z_alpha: float,
        z_power: float,
        hypothetical_effect: float,
        control_var: float,
        treatment_var: float,
        kappa: float,
    ) -> float:
        return np.ceil(
            np.power((z_alpha + z_power) / abs(hypothetical_effect), 2) * (control_var / kappa + treatment_var)
        )

    def find_current_powered_effect(
        self,
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

        treatment_var = self._get_hypothetical_treatment_var(
            binary_metric=binary,
            non_inferiority=non_inferiority,
            control_avg=control_avg,
            control_var=control_var,
            hypothetical_effect=hypothetical_effect,
        )
        n2 = self._treatment_group_sample_size(
            z_alpha,
            z_power,
            hypothetical_effect,
            control_var,
            treatment_var,
            kappa,
        )

        return np.power(current_number_of_units - ((n2 + n2 * kappa) / proportion_of_total), 2)

    def _get_hypothetical_treatment_var(
        self,
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
