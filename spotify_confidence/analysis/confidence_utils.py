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

from collections import OrderedDict
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, Iterable, Tuple, List

import numpy as np
from pandas import DataFrame, concat, Series
from scipy.stats import norm

from spotify_confidence.analysis.constants import (
    SFX1,
    SFX2,
)


def groupbyApplyParallel(dfGrouped, func_to_apply):
    with ThreadPoolExecutor(max_workers=32, thread_name_prefix="groupbyApplyParallel") as p:
        ret_list = p.map(
            func_to_apply,
            [group for name, group in dfGrouped],
        )
    return concat(ret_list)


def applyParallel(df, func_to_apply, splits=32):
    with ThreadPoolExecutor(max_workers=splits, thread_name_prefix="applyParallel") as p:
        ret_list = p.map(
            func_to_apply,
            np.array_split(df, min(splits, len(df))),
        )
    return concat(ret_list)


def get_all_group_columns(categorical_columns: Iterable, additional_column: str) -> Iterable:
    all_columns = listify(categorical_columns) + listify(additional_column)
    return list(OrderedDict.fromkeys(all_columns))


def remove_group_columns(categorical_columns: Iterable, additional_column: str) -> Iterable:
    od = OrderedDict.fromkeys(categorical_columns)
    if additional_column is not None:
        del od[additional_column]
    return list(od)


def validate_categorical_columns(categorical_group_columns: Union[str, Iterable]) -> Iterable:
    if isinstance(categorical_group_columns, str):
        pass
    elif isinstance(categorical_group_columns, Iterable):
        pass
    else:
        raise TypeError(
            """categorical_group_columns must be string or
                           iterable (list of columns) and you must
                           provide at least one"""
        )


def listify(column_s: Union[str, Iterable]) -> List:
    if isinstance(column_s, str):
        return [column_s]
    elif isinstance(column_s, Iterable):
        return list(column_s)
    elif column_s is None:
        return []


def get_remaning_groups(all_groups: Iterable, some_groups: Iterable) -> Iterable:
    if some_groups is None:
        remaining_groups = all_groups
    else:
        remaining_groups = [group for group in all_groups if group not in some_groups and group is not None]
    return remaining_groups


def get_all_categorical_group_columns(
    categorical_columns: Union[str, Iterable, None],
    metric_column: Union[str, None],
    treatment_column: Union[str, None],
) -> Iterable:
    all_columns = listify(treatment_column) + listify(categorical_columns) + listify(metric_column)
    return list(OrderedDict.fromkeys(all_columns))


def validate_levels(df: DataFrame, level_columns: Union[str, Iterable], levels: Iterable):
    for level in levels:
        try:
            df.groupby(level_columns).get_group(level)
        except (KeyError, ValueError):
            raise ValueError(
                """
                    Invalid level: '{}'
                    Must supply a level within the ungrouped dimensions: {}
                    Valid levels:
                    {}
                    """.format(
                    level, level_columns, list(df.groupby(level_columns).groups.keys())
                )
            )


def validate_and_rename_columns(df: DataFrame, columns: Iterable[str]) -> DataFrame:
    for column in columns:
        if column is None or column + SFX1 not in df.columns or column + SFX2 not in df.columns:
            continue

        if (df[column + SFX1].isna() == df[column + SFX1].isna()).all() and (
            df[column + SFX1][df[column + SFX1].notna()] == df[column + SFX1][df[column + SFX1].notna()]
        ).all():
            df = df.rename(columns={column + SFX1: column}).drop(columns=[column + SFX2])
        else:
            raise ValueError(f"Values of {column} do not agree across levels: {df[[column + SFX1, column + SFX2]]}")
    return df


def drop_and_rename_columns(df: DataFrame, columns: Iterable[str]) -> DataFrame:
    columns_dict = {col + SFX1: col for col in columns}
    return df.rename(columns=columns_dict).drop(columns=[col + SFX2 for col in columns])


def level2str(level: Union[str, Tuple]) -> str:
    if isinstance(level, str) or not isinstance(level, Iterable):
        return str(level)
    else:
        return ", ".join([str(sub_level) for sub_level in level])


def validate_data(df: DataFrame, columns_that_must_exist, group_columns: Iterable, ordinal_group_column: str):
    """Integrity check input dataframe."""
    for col in columns_that_must_exist:
        _validate_column(df, col)

    if not group_columns:
        raise ValueError(
            """At least one of `categorical_group_columns`
                            or `ordinal_group_column` must be specified."""
        )

    for col in group_columns:
        _validate_column(df, col)

    # Ensure there's at most 1 observation per grouping.
    max_one_row_per_grouping = all(df.groupby(group_columns, sort=False).size() <= 1)
    if not max_one_row_per_grouping:
        raise ValueError("""Each grouping should have at most 1 observation.""")

    if ordinal_group_column:
        ordinal_column_type = df[ordinal_group_column].dtype.type
        if not np.issubdtype(ordinal_column_type, np.number) and not issubclass(ordinal_column_type, np.datetime64):
            raise TypeError(
                """`ordinal_group_column` is type `{}`.
        Must be number or datetime type.""".format(
                    ordinal_column_type
                )
            )


def _validate_column(df: DataFrame, col: str):
    if col not in df.columns:
        raise ValueError(f"""Column {col} is not in dataframe""")


def is_non_inferiority(nim) -> bool:
    if isinstance(nim, float):
        return not np.isnan(nim)
    elif nim is None:
        return nim is not None


def reset_named_indices(df):
    named_indices = [name for name in df.index.names if name is not None]
    if len(named_indices) > 0:
        return df.reset_index(named_indices, drop=True).sort_index()
    else:
        return df


def _get_finite_bounds(numbers: Series) -> Tuple[float, float]:
    finite_numbers = numbers[numbers.abs() != float("inf")]
    return finite_numbers.min(), finite_numbers.max()


def axis_format_precision(numbers: Series, absolute: bool, extra_zeros: int = 0) -> Tuple[str, float, float]:
    min_value, max_value = _get_finite_bounds(numbers)

    if max_value == min_value:
        return "0.00", min_value, max_value

    extra_zeros += 2 if absolute else 0
    precision = -int(np.log10(abs(max_value - min_value))) + extra_zeros
    zeros = "".join(["0"] * precision)
    return "0.{}{}".format(zeros, "" if absolute else "%"), min_value, max_value


def to_finite(s: Series, limit: float) -> Series:
    return s.clip(-100 * abs(limit), 100 * abs(limit))


def add_color_column(df: DataFrame, cols: Iterable) -> DataFrame:
    return df.assign(color=df[cols].agg(level2str, axis="columns"))


def power_calculation(mde: float, baseline_var: float, alpha: float, n1: int, n2: int) -> float:

    z_alpha = norm.ppf(1 - alpha / 2)
    a = abs(mde) / np.sqrt(baseline_var)
    b = np.sqrt(n1 * n2 / (n1 + n2))
    z_stat = a * b

    return norm.cdf(z_stat - z_alpha) + norm.cdf(-z_stat - z_alpha)


def unlist(x):
    x0 = x[0] if isinstance(x, list) else x
    x1 = np.atleast_2d(x0)
    if x1.shape[0] < x1.shape[1]:
        x1 = x1.transpose()
    return x1


def dfmatmul(x, y, outer=True):

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.shape[0] < x.shape[1]:
        x = x.transpose()
    if y.shape[0] < y.shape[1]:
        y = y.transpose()

    if outer:
        out = np.matmul(x, np.transpose(y))
    else:
        out = np.matmul(np.transpose(x), y)

    if out.size == 1:
        out = out.item()
    return out
