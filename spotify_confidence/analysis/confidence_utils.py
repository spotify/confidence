# Copyright 2017-2020 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import (Union, Iterable, Tuple, List)
from pandas import (DataFrame, concat, Series)
import numpy as np
from scipy import optimize
from scipy.stats import norm

from spotify_confidence.analysis.constants import (
    INCREASE_PREFFERED, DECREASE_PREFFERED, TWO_SIDED,
    NIM_TYPE, NIM_INPUT_COLUMN_NAME, PREFERRED_DIRECTION_INPUT_NAME,
    NIM, NULL_HYPOTHESIS, PREFERENCE,
    SFX1, SFX2, POINT_ESTIMATE)


def get_all_group_columns(categorical_columns: Iterable,
                          ordinal_column: str) -> Iterable:
    all_columns = categorical_columns + [ordinal_column]
    all_columns = [col for col in all_columns if col is not None]
    return all_columns


def validate_categorical_columns(
        categorical_group_columns: Union[str, Iterable]) -> Iterable:
    if isinstance(categorical_group_columns, str):
        pass
    elif isinstance(categorical_group_columns, Iterable):
        pass
    else:
        raise TypeError("""categorical_group_columns must be string or
                           iterable (list of columns) and you must
                           provide at least one""")


def listify(column_s: Union[str, Iterable]) -> List:
    if isinstance(column_s, str):
        return [column_s]
    elif isinstance(column_s, Iterable):
        return list(column_s)
    elif column_s is None:
        return []


def get_remaning_groups(all_groups: Iterable,
                        some_groups: Iterable) -> Iterable:
    if some_groups is None:
        remaining_groups = all_groups
    else:
        remaining_groups = [
            group for group in all_groups
            if group not in some_groups and group is not None
        ]
    return remaining_groups


def validate_levels(df: DataFrame,
                    level_columns: Union[str, Iterable],
                    levels: Iterable):
    for level in levels:
        try:
            df.groupby(level_columns).get_group(level)
        except (KeyError, ValueError):
            raise ValueError("""
                    Invalid level: '{}'
                    Must supply a level within the ungrouped dimensions: {}
                    Valid levels:
                    {}
                    """.format(
                level, level_columns,
                list(df.groupby(level_columns).groups.keys())))


def add_nim_columns(df: DataFrame, nims: NIM_TYPE) -> DataFrame:
    def _nim_2_signed_nim(nim: Tuple[float, str]) -> Tuple[float, float, str]:
        nim_value = 0 if nim[0] is None or (type(nim[0]) is float and np.isnan(nim[0])) else nim[0]
        if nim[1] is None or (type(nim[1]) is float and np.isnan(nim[1])):
            return (nim[0], nim_value, TWO_SIDED)
        elif nim[1].lower() == INCREASE_PREFFERED:
            return (nim[0], -nim_value, 'larger')
        elif nim[1].lower() == DECREASE_PREFFERED:
            return (nim[0], nim_value, 'smaller')
        else:
            raise ValueError(f'{nim[1].lower()} not in '
                             f'{[INCREASE_PREFFERED, DECREASE_PREFFERED]}')

    if nims is None:
        return (
            df.assign(**{NIM: None})
              .assign(**{NULL_HYPOTHESIS: 0})
              .assign(**{PREFERENCE: TWO_SIDED})
        )
    elif type(nims) is tuple:
        return (
            df.assign(**{NIM: _nim_2_signed_nim((nims[0], nims[1]))[0]})
              .assign(**{NULL_HYPOTHESIS: df[POINT_ESTIMATE] * _nim_2_signed_nim((nims[0], nims[1]))[1]})
              .assign(**{PREFERENCE: _nim_2_signed_nim((nims[0], nims[1]))[2]})
        )
    elif type(nims) is dict:
        sgnd_nims = {group: _nim_2_signed_nim(nim) for group, nim in nims.items()}
        nim_df = (
            DataFrame(index=df.index,
                      columns=[NIM, NULL_HYPOTHESIS, PREFERENCE],
                      data=list(df.index.to_series().map(sgnd_nims)))
        )
        return (
            df.assign(**{NIM: nim_df[NIM]})
              .assign(**{NULL_HYPOTHESIS: df[POINT_ESTIMATE] * nim_df[NULL_HYPOTHESIS]})
              .assign(**{PREFERENCE: nim_df[PREFERENCE]})
        )
    elif type(nims) is bool:
        return (
            df.assign(**{NIM: lambda df: df[NIM_INPUT_COLUMN_NAME]})
              .assign(**{NULL_HYPOTHESIS: lambda df: df.apply(
                lambda row: row[POINT_ESTIMATE] * _nim_2_signed_nim((row[NIM], row[PREFERRED_DIRECTION_INPUT_NAME]))[1],
                axis=1)})
              .assign(**{PREFERENCE: lambda df: df.apply(lambda row: _nim_2_signed_nim(
                (row[NIM], row[PREFERRED_DIRECTION_INPUT_NAME]))[2], axis=1)})
        )
    else:
        raise ValueError(f'non_inferiority_margins must be None, tuple, dict,'
                         f'or DataFrame, but is {type(nims)}.')


def equals_none_or_nan(x, y):
    return True if x == y or (x is None and y is None) \
                   or (type(x) is float and type(y) is float and np.isnan(x) and np.isnan(y)) else False


def validate_and_rename_nims(df: DataFrame) -> DataFrame:
    if (df.apply(lambda row: equals_none_or_nan(row[NIM + SFX1], row[NIM + SFX2]), axis=1).all() and
            df.apply(lambda row: equals_none_or_nan(row[PREFERENCE + SFX1], row[PREFERENCE + SFX2]), axis=1).all()):
        return (
            df.rename(columns={NIM + SFX1: NIM,
                               NULL_HYPOTHESIS + SFX1: NULL_HYPOTHESIS,
                               PREFERENCE + SFX1: PREFERENCE})
              .drop(columns=[NIM + SFX2,
                             NULL_HYPOTHESIS + SFX2,
                             PREFERENCE + SFX2])
        )

    raise ValueError("Non-inferiority margins do not agree across levels")


def validate_and_rename_final_expected_sample_sizes(df: DataFrame, column: str) -> DataFrame:
    if column is None:
        return df

    if df.apply(lambda row: equals_none_or_nan(row[column + SFX1], row[column + SFX2]), axis=1).all():
        return (
            df.rename(columns={column + SFX1: column})
              .drop(columns=[column + SFX2])
        )

    raise ValueError("Final expected sample sizes do not agree across levels")


def select_levels(df: DataFrame,
                  level_columns: Union[str, Iterable],
                  level_1: Union[str, Tuple],
                  level_2: Union[str, Tuple]) -> DataFrame:
    gdf = df.groupby(level_columns)
    return concat([gdf.get_group(level_1), gdf.get_group(level_2)])


def level2str(level: Union[str, Tuple]) -> str:
    if isinstance(level, str) or not isinstance(level, Iterable):
        return str(level)
    else:
        return ', '.join([str(sub_level) for sub_level in level])


def validate_data(df: DataFrame,
                  numerator: str,
                  numerator_sumsq: str,
                  denominator: str,
                  group_columns: Iterable,
                  ordinal_group_column: str):
    """Integrity check input dataframe.
    """
    _validate_column(df, numerator)
    if numerator_sumsq is not None:
        _validate_column(df, numerator_sumsq)
    _validate_column(df, denominator)

    if not group_columns:
        raise ValueError("""At least one of `categorical_group_columns`
                            or `ordinal_group_column` must be specified."""
                         )

    for col in group_columns:
        _validate_column(df, col)

    # Ensure there's at most 1 observation per grouping.
    max_one_row_per_grouping = all(
        df.groupby(group_columns).size() <= 1)
    if not max_one_row_per_grouping:
        raise ValueError(
            """Each grouping should have at most 1 observation.""")

    if ordinal_group_column:
        ordinal_column_type = df[
            ordinal_group_column].dtype.type
        if not np.issubdtype(ordinal_column_type, np.number) \
                and not issubclass(ordinal_column_type, np.datetime64):
            raise TypeError("""`ordinal_group_column` is type `{}`.
        Must be number or datetime type.""".format(ordinal_column_type))


def _validate_column(df: DataFrame, col: str):
    if col not in df.columns:
        raise ValueError(f"""Column {col} is not in dataframe""")


def _get_finite_bounds(numbers: Series) -> Tuple[float, float]:
    finite_numbers = numbers[numbers.abs() != float("inf")]
    return finite_numbers.min(), finite_numbers.max()


def axis_format_precision(numbers: Series,
                          absolute: bool,
                          extra_zeros: int = 0) -> Tuple[str, float, float]:
    min_value, max_value = _get_finite_bounds(numbers)

    if max_value == min_value:
        return "0.00", min_value, max_value

    extra_zeros += 2 if absolute else 0
    precision = -int(np.log10(abs(max_value - min_value))) + extra_zeros
    zeros = ''.join(['0'] * precision)
    return "0.{}{}".format(zeros, '' if absolute else '%'), min_value, max_value


def to_finite(s: Series, limit: float) -> Series:
    return s.clip(-100*abs(limit), 100*abs(limit))


def add_color_column(df: DataFrame, cols: Iterable) -> DataFrame:
    return df.assign(color=df[cols].agg(level2str, axis='columns'))


def power_calculation(mde: float,
                      baseline_var: float,
                      alpha: float,
                      n1: int,
                      n2: int) -> float:

    z_alpha = norm.ppf(1 - alpha / 2)
    a = abs(mde) / np.sqrt(baseline_var)
    b = np.sqrt(n1 * n2 / (n1 + n2))
    z_stat = a * b

    return norm.cdf(z_stat - z_alpha) + norm.cdf(-z_stat - z_alpha)


###################################################################################################
#################### current powered effect



def _currently_powered_effect(
    control_avg: float,
    control_var: float,
    metric_type: str,
    non_inferiority: bool = False,
    power: float = None,
    alpha: float = None,
    z_power: float = None,
    z_alpha: float = None,
    kappa: float = None,
    proportion_of_total: float = None,
    current_number_of_units: float = None,
):
    z_alpha = norm.ppf(1 - alpha) if z_alpha is None else z_alpha
    z_power = norm.ppf(power) if z_power is None else z_power

    if metric_type == BINARY and not non_inferiority:
        effect = _search_MDE_binary_local_search(
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
        treatment_var = _get_hypothetical_treatment_var(
            metric_type, non_inferiority, control_avg, control_var, hypothetical_effect=0
        )
        n2_partial = np.power((z_alpha + z_power), 2) * (control_var / kappa + treatment_var)
        effect = np.sqrt((1 / (current_number_of_units * proportion_of_total)) * (n2_partial + kappa * n2_partial))

    return effect


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
        return find_current_powered_effect(
            hypothetical_effect=x,
            control_avg=control_avg,
            control_var=control_var,
            metric_type=BINARY,
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

    # we stop immediately if a solution was found that is "good enough". A threshold of 1 indicates that
    # the approximated number of units (based on the current effect candidate) is off by at most 1.0
    goodness_threshold = 1.0

    curr_iter = 0
    best_x = None
    best_fun = float("inf")

    bounds_queue = [(min_val, max_val)]

    while curr_iter < max_iter and best_fun > goodness_threshold:

        # take next value from queue
        interval = bounds_queue.pop(0)

        # conduct a bounded local search, using a very small tol value improved performance during tests
        # result = optimize.minimize_scalar(f, bounds=(interval[0], interval[1]), method='bounded', tol=10e-14)
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
        alt_result_x, alt_result_fun = search_MDE_binary(
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
                find_current_powered_effect(
                    hypothetical_effect=effect,
                    control_avg=control_avg,
                    control_var=control_var,
                    metric_type=BINARY,
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

    def find_current_powered_effect(
            hypothetical_effect: float,
            control_avg: float,
            control_var: float,
            metric_type: str,
            non_inferiority: bool,
            kappa: float,
            proportion_of_total: float,
            current_number_of_units: float,

            power: float = None,
            alpha: float = None,
    ) -> float:

        z_alpha = norm.ppf(1 - alpha)
        z_power = norm.ppf(power)

        treatment_var = _get_hypothetical_treatment_var(
            metric_type, non_inferiority, control_avg, control_var, hypothetical_effect
        )
        n2 = _treatment_group_sample_size(z_alpha, z_power, hypothetical_effect, control_var,
                                          treatment_var, kappa, )

        return np.power(current_number_of_units - ((n2 + n2 * kappa) / proportion_of_total), 2)

def _get_hypothetical_treatment_var(
    metric_type: str, non_inferiority: bool, control_avg: float, control_var: float, hypothetical_effect: float,
) -> float:
    if metric_type == BINARY and not non_inferiority:
        # For binary metrics, the variance can be derived from the average. However, we do *not* do this for
        # non-inferiority tests because for non-inferiority tests, the basic assumption is that the
        # mean of the control group and treatment group are identical.
        return _binary_variance(control_avg + hypothetical_effect)
    else:
        return control_var


def _treatment_group_sample_size(
    z_alpha: float, z_power: float, hypothetical_effect: float, control_var: float, treatment_var: float, kappa: float,
) -> float:
    return np.ceil(np.power((z_alpha + z_power) / abs(hypothetical_effect), 2) * (control_var / kappa + treatment_var))
