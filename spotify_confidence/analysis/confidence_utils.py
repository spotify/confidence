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

from typing import (Union, Iterable, Tuple, List, Dict)
from pandas import (DataFrame, concat, Series)
import numpy as np
from scipy.stats import norm

from spotify_confidence.analysis.constants import (
    INCREASE_PREFFERED, DECREASE_PREFFERED, TWO_SIDED)


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


def validate_nims(df: DataFrame,
                  groupby: List,
                  nims: Union[Tuple, Dict[str, Tuple[float, str]]]):
    assert (nims is None or
            type(nims) is tuple or
            (type(nims) is dict and
             set(df.groupby(groupby).groups.keys()) == set(nims.keys())))


def signed_nims(nims: Union[Tuple, Dict[str, Tuple[float, str]]]
                ) -> Union[Tuple[float, float, str],
                           Dict[str, Tuple[float, float, str]]]:
    def _nim_2_signed_nim(nim: Tuple[float, str]) -> Tuple[float, float, str]:
        nim_value = 0 if nim[0] is None or np.isnan(nim[0]) else nim[0]
        if nim[1] is None:
            return (nim[0], nim_value, TWO_SIDED)
        elif nim[1].lower() == INCREASE_PREFFERED:
            return (nim[0], -nim_value, 'larger')
        elif nim[1].lower() == DECREASE_PREFFERED:
            return (nim[0], nim_value, 'smaller')
        else:
            raise ValueError(f'{nim[1].lower()} not in '
                             f'{[INCREASE_PREFFERED, DECREASE_PREFFERED]}')
    if nims is None:
        return (None, 0.0, TWO_SIDED)
    elif type(nims) is tuple:
        return _nim_2_signed_nim(nims)
    elif type(nims) is dict:
        return {group: _nim_2_signed_nim(nim) for group, nim in nims.items()}
    else:
        raise ValueError(f'Non inferiority margins need to be either tuple '
                         f'or dictionary')


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
