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
from typing import Union, Iterable, Tuple, List

import numpy as np
from pandas import DataFrame, concat, Series
from scipy.stats import norm

from spotify_confidence.analysis.constants import (
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    TWO_SIDED,
    NIM_TYPE,
    NIM_INPUT_COLUMN_NAME,
    PREFERRED_DIRECTION_INPUT_NAME,
    NIM,
    NULL_HYPOTHESIS,
    PREFERENCE,
    ALTERNATIVE_HYPOTHESIS,
    SFX1,
    SFX2,
    POINT_ESTIMATE,
)


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


def add_mde_columns(df: DataFrame, mde_column: str) -> DataFrame:
    def _mde_2_signed_mde(mde: Tuple[float, str]) -> Tuple[float, float, str]:
        mde_value = None if (type(mde[0]) is float and np.isnan(mde[0])) else mde[0]
        if mde[1] is None or (type(mde[1]) is float and np.isnan(mde[1])):
            return (mde[0], mde_value, TWO_SIDED)
        elif mde[1].lower() == INCREASE_PREFFERED:
            return (mde[0], None if mde_value is None else -mde_value, "larger")
        elif mde[1].lower() == DECREASE_PREFFERED:
            return (mde[0], mde_value, "smaller")

    if mde_column is not None:
        return (
            df.assign(
                tmp_alt_hyp_dir=df.apply(
                    lambda row: _mde_2_signed_mde(
                        (
                            row[mde_column],
                            (row[PREFERRED_DIRECTION_INPUT_NAME] if PREFERRED_DIRECTION_INPUT_NAME in row else np.nan),
                        )
                    ),
                    axis=1,
                )
            )
            .assign(
                **{
                    ALTERNATIVE_HYPOTHESIS: lambda df: df.apply(
                        lambda row: None
                        if row["tmp_alt_hyp_dir"][1] is None
                        else row[POINT_ESTIMATE] * row["tmp_alt_hyp_dir"][1],
                        axis=1,
                    )
                }
            )
            .assign(**{PREFERENCE: lambda df: df.apply(lambda row: row["tmp_alt_hyp_dir"][2], axis=1)})
            .drop(columns=[mde_column, "tmp_alt_hyp_dir"])
            .assign(**{NULL_HYPOTHESIS: 0})
        )
    else:
        return df


def add_nim_columns(df: DataFrame, nims: NIM_TYPE) -> DataFrame:
    def _nim_2_signed_nim(nim: Tuple[float, str]) -> Tuple[float, float, str]:
        nim_value = 0 if nim[0] is None or (type(nim[0]) is float and np.isnan(nim[0])) else nim[0]
        if nim[1] is None or (type(nim[1]) is float and np.isnan(nim[1])):
            return (nim[0], nim_value, TWO_SIDED)
        elif nim[1].lower() == INCREASE_PREFFERED:
            return (nim[0], -nim_value, "larger")
        elif nim[1].lower() == DECREASE_PREFFERED:
            return (nim[0], nim_value, "smaller")
        else:
            raise ValueError(f"{nim[1].lower()} not in " f"{[INCREASE_PREFFERED, DECREASE_PREFFERED]}")

    if nims is None:
        return (
            df.assign(**{NIM: None})
            .assign(**{NULL_HYPOTHESIS: 0})
            .assign(**{PREFERENCE: TWO_SIDED})
            .assign(**{ALTERNATIVE_HYPOTHESIS: None})
        )
    elif type(nims) is tuple:
        return (
            df.assign(**{NIM: _nim_2_signed_nim((nims[0], nims[1]))[0]})
            .assign(**{NULL_HYPOTHESIS: df[POINT_ESTIMATE] * _nim_2_signed_nim((nims[0], nims[1]))[1]})
            .assign(**{PREFERENCE: _nim_2_signed_nim((nims[0], nims[1]))[2]})
            .assign(**{ALTERNATIVE_HYPOTHESIS: 0})
        )
    elif type(nims) is dict:
        sgnd_nims = {group: _nim_2_signed_nim(nim) for group, nim in nims.items()}
        nim_df = DataFrame(
            index=df.index, columns=[NIM, NULL_HYPOTHESIS, PREFERENCE], data=list(df.index.to_series().map(sgnd_nims))
        )
        return (
            df.assign(**{NIM: nim_df[NIM]})
            .assign(**{NULL_HYPOTHESIS: df[POINT_ESTIMATE] * nim_df[NULL_HYPOTHESIS]})
            .assign(**{PREFERENCE: nim_df[PREFERENCE]})
            .assign(**{ALTERNATIVE_HYPOTHESIS: 0})
        )
    elif type(nims) is bool:
        return (
            df.assign(**{NIM: lambda df: df[NIM_INPUT_COLUMN_NAME]})
            .assign(
                **{
                    NULL_HYPOTHESIS: lambda df: df.apply(
                        lambda row: row[POINT_ESTIMATE]
                        * _nim_2_signed_nim((row[NIM], row[PREFERRED_DIRECTION_INPUT_NAME]))[1],
                        axis=1,
                    )
                }
            )
            .assign(
                **{
                    PREFERENCE: lambda df: df.apply(
                        lambda row: _nim_2_signed_nim((row[NIM], row[PREFERRED_DIRECTION_INPUT_NAME]))[2], axis=1
                    )
                }
            )
            .assign(**{ALTERNATIVE_HYPOTHESIS: 0})
        )
    else:
        raise ValueError(f"non_inferiority_margins must be None, tuple, dict," f"or DataFrame, but is {type(nims)}.")


def equals_none_or_nan(x, y):
    return (
        True
        if x == y
        or (x is None and y is None)
        or (type(x) is float and type(y) is float and np.isnan(x) and np.isnan(y))
        else False
    )


def validate_and_rename_columns(df: DataFrame, column: str) -> DataFrame:
    if column is None or column + SFX1 not in df.columns or column + SFX2 not in df.columns:
        return df

    if df.apply(lambda row: equals_none_or_nan(row[column + SFX1], row[column + SFX2]), axis=1).all():
        return df.rename(columns={column + SFX1: column}).drop(columns=[column + SFX2])

    raise ValueError(f"Values of {column} do not agree across levels: {df[[column + SFX1, column + SFX2]]}")


def select_levels(
    df: DataFrame, level_columns: Union[str, Iterable], level_1: Union[str, Tuple], level_2: Union[str, Tuple]
) -> DataFrame:
    gdf = df.groupby(level_columns)
    return concat([gdf.get_group(level_1), gdf.get_group(level_2)])


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
    max_one_row_per_grouping = all(df.groupby(group_columns).size() <= 1)
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
