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

from abc import ABCMeta, abstractmethod
from functools import wraps
import types
import warnings

import chartify
import numpy as np
import pandas as pd

from spotify_confidence.options import options
from spotify_confidence.chartgrid import ChartGrid

# warnings.simplefilter("once")

INITIAL_RANDOMIZATION_SEED = np.random.get_state()[1][0]


def axis_format_precision(max_value, min_value, absolute):
    extra_zeros = 2 if absolute else 0
    precision = -int(np.log10(abs(max_value - min_value))) + extra_zeros
    zeros = "".join(["0"] * precision)
    return "0.{}{}".format(zeros, "" if absolute else "%")


def add_color_column(df, cols):
    for i, column in enumerate(cols):
        if i == 0:
            df["color"] = df[column]
        else:
            df["color"] = df["color"] + " " + df[column]
    return df


def randomization_warning_decorator(f):
    """Set numpy randomization seed and warn users if not fixed.

    Note to developers:
    Do not compare random variables that have been
    sampled from the same seed. It will lead to incorrect results.
    To avoid this situation it's best to apply this decorator to
    public methods that involve randomization.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):

        option_seed = options.get_option("randomization_seed")
        np_seed = INITIAL_RANDOMIZATION_SEED
        if option_seed != np_seed and option_seed is None:
            randomization_warning_message = """
    Your analysis will not be reproducible!
    Using a method that involves randomization without setting a seed.
    Please run the following and add it to the top of your script or
    notebook after you import confidence:

    confidence.options.set_option('randomization_seed', {})

    """.format(
                INITIAL_RANDOMIZATION_SEED
            )
            warnings.warn(randomization_warning_message)
            option_seed = np_seed
        np.random.seed(option_seed)
        return f(*args, **kwargs)

    return wrapper


class BaseTest(object, metaclass=ABCMeta):
    """Base test class that provides abstract methods
    to ensure consistency across test classes."""

    def __init__(
        self,
        data_frame,
        categorical_group_columns,
        ordinal_group_column,
        numerator_column,
        denominator_column,
        interval_size,
    ):

        self._data_frame = data_frame
        self._numerator_column = numerator_column
        self._denominator_column = denominator_column
        self._interval_size = interval_size

        categorical_string_or_none = isinstance(categorical_group_columns, str) or categorical_group_columns is None
        self._categorical_group_columns = (
            [categorical_group_columns] if categorical_string_or_none else categorical_group_columns
        )
        self._ordinal_group_column = ordinal_group_column

        self._all_group_columns = self._categorical_group_columns + [self._ordinal_group_column]
        self._all_group_columns = [column for column in self._all_group_columns if column is not None]
        self._validate_data()

    def _validate_data(self):
        """Integrity check input dataframe."""
        if not self._all_group_columns:
            raise ValueError(
                """At least one of `categorical_group_columns`
                                or `ordinal_group_column` must be specified."""
            )

        # Ensure there's at most 1 observation per grouping.
        max_one_row_per_grouping = all(self._data_frame.groupby(self._all_group_columns).size() <= 1)
        if not max_one_row_per_grouping:
            raise ValueError("""Each grouping should have at most 1 observation.""")

        if self._ordinal_group_column:
            ordinal_column_type = self._data_frame[self._ordinal_group_column].dtype.type
            if not np.issubdtype(ordinal_column_type, np.number) and not issubclass(
                ordinal_column_type, np.datetime64
            ):
                raise TypeError(
                    """`ordinal_group_column` is type `{}`.
    Must be number or datetime type.""".format(
                        ordinal_column_type
                    )
                )

    @classmethod
    def as_cumulative(
        cls, data_frame, numerator_column, denominator_column, ordinal_group_column, categorical_group_columns=None
    ):
        """
        Instantiate the class with a cumulative representation of the dataframe.
        Sorts by the ordinal variable and calculates the cumulative sum
        May be used for to visualize the difference between groups as a
        time series.

        Args:
           data_frame (pd.DataFrame): DataFrame
           numerator_column (str): Column name for numerator column.
           denominator_column (str): Column name for denominator column.
           ordinal_group_column (str): Column name for ordinal grouping
               (e.g. numeric or date values).
           categorical_group_columns (str or list),
               Optional: Column names for categorical groupings.

        """

        sorted_df = data_frame.sort_values(by=ordinal_group_column)
        cumsum_cols = [numerator_column, denominator_column]
        if categorical_group_columns:
            sorted_df[cumsum_cols] = sorted_df.groupby(by=categorical_group_columns)[cumsum_cols].cumsum()
        else:
            sorted_df[cumsum_cols] = sorted_df[cumsum_cols].cumsum()

        return cls(sorted_df, numerator_column, denominator_column, categorical_group_columns, ordinal_group_column)

    def summary(self):
        """Return Pandas DataFrame with summary statistics."""
        return self._summary(self._data_frame, self._interval)

    def _summary(self, data_frame, ci_function):
        """Return the input dataframe with added columns:
        - Lower & upper bounds of
            Bayesian: credible interval
            Frequentist: confidence interval
        - Additional summary stats
            (e.g. probability in the case of Binomial data)
        """
        summary_df = data_frame[self._all_group_columns + [self._numerator_column, self._denominator_column]].copy()

        summary_df["point_estimate"] = summary_df[self._numerator_column] * 1.0 / summary_df[self._denominator_column]
        summary_df[["ci_lower", "ci_upper"]] = data_frame.apply(ci_function, axis=1, result_type="expand")

        return summary_df

    def summary_plot(self, groupby=None):
        """Plot for each group in the data_frame:

        if ordinal level exists:
            Frequentist: line graph with area to represent confidence interval
            Bayesian: line graph with area to represent credible interval
        if categorical levels:
            Bayesian: KDE plot of posterior distributions by group
            Frequentist: Interval plots of confidence intervals by group

        Args:
            groupby (str): Name of column.
                If specified, will plot a separate chart for each level of the
                grouping.

        Returns:
            ChartGrid object.
        """
        chart_grid = self._iterate_groupby_to_chartgrid(self._summary_plot, groupby=groupby)
        return chart_grid

    def _summary_plot(self, level_name, level_df, remaining_groups, groupby):

        if self._ordinal_group_column is not None and self._ordinal_group_column in remaining_groups:

            ch = self._ordinal_summary_plot(level_name, level_df, remaining_groups, groupby)
        else:
            ch = self._categorical_summary_plot(level_name, level_df, remaining_groups, groupby)
        return ch

    def _ordinal_summary_plot(self, level_name, level_df, remaining_groups, groupby):
        remaining_groups = self._remaining_categorical_groups(remaining_groups)
        df = self._summary(level_df, self._interval)
        title = "Estimate of {} / {}".format(self._numerator_column, self._denominator_column)
        y_axis_label = "{} / {}".format(self._numerator_column, self._denominator_column)
        return self._ordinal_plot(
            "point_estimate",
            df,
            groupby,
            level_name,
            remaining_groups,
            absolute=True,
            title=title,
            y_axis_label=y_axis_label,
        )

    def _ordinal_plot(self, center_name, df, groupby, level_name, remaining_groups, absolute, title, y_axis_label):
        df = add_color_column(df, remaining_groups)
        colors = "color" if remaining_groups else None
        ch = chartify.Chart(x_axis_type=self._ordinal_type())
        ch.plot.line(
            data_frame=df.sort_values(self._ordinal_group_column),
            x_column=self._ordinal_group_column,
            y_column=center_name,
            color_column=colors,
        )
        ch.style.color_palette.reset_palette_order()
        ch.plot.area(
            data_frame=df.sort_values(self._ordinal_group_column),
            x_column=self._ordinal_group_column,
            y_column="ci_lower",
            second_y_column="ci_upper",
            color_column=colors,
        )
        ch.axes.set_yaxis_label(y_axis_label)
        ch.axes.set_xaxis_label(self._ordinal_group_column)
        ch.set_source_label("")
        axis_format = axis_format_precision(df["ci_lower"].min(), df["ci_upper"].max(), absolute)
        ch.axes.set_yaxis_tick_format(axis_format)
        subtitle = "" if not groupby else "{}: {}".format(groupby, level_name)
        ch.set_subtitle(subtitle)
        ch.set_title(title)
        if colors:
            ch.set_legend_location("outside_bottom")
        return ch

    def _remaining_categorical_groups(self, remaining_groups):
        remaining_groups_list = [remaining_groups] if isinstance(remaining_groups, str) else remaining_groups

        remaining_categorical_groups = [
            group_name for group_name in remaining_groups_list if group_name != self._ordinal_group_column
        ]
        return remaining_categorical_groups

    def _ordinal_type(self):
        ordinal_column_type = self._data_frame[self._ordinal_group_column].dtype.type
        axis_type = "datetime" if issubclass(ordinal_column_type, np.datetime64) else "linear"
        return axis_type

    @abstractmethod
    def _categorical_summary_plot(self, level_name, level_df, remaining_groups, groupby):
        pass

    @abstractmethod
    def difference(self, level_1, level_2, absolute=True, groupby=None):
        """Return dataframe containing the difference in means between
            group 1 and 2 and the appropriate test statistics.
        Frequentist:
        - Calculate one of the following tests depending of the
            response variable type.
            - Binomial: Chisq / fisher exact test
            - Gaussian: t-test / z-test
            Return the p-value.
        Bayesian:
        - Calcuate the posterior distribution of the difference in means.
            Return the
            - probability that group 2 > group 1.
            - Expected loss
            - Expected change
            - Expected gain
            - 95% CI interval
        """
        pass

    def difference_plot(self, level_1, level_2, absolute=True, groupby=None):
        """Plot representing the difference between group 1 and 2.
        - Difference in means or proportions, depending
            on the response variable type.

        Frequentist:
        - Plot interval plot with confidence interval of the
            difference between groups

        Bayesian:
        - Plot KDE representing the posterior distribution of the difference.
            - Probability that group2 > group1
            - Mean difference
            - 95% interval.

        Args:
            level_1 (str, tuple of str): Name of first level.
            level_2 (str, tuple of str): Name of second level.
            absolute (bool): If True then return the absolute
                difference (level2 - level1)
                otherwise return the relative difference (level2 / level1 - 1)
            groupby (str): Name of column, or list of columns.
                If specified, will return an interval for each level
                of the grouped dimension, or a confidence band if the
                grouped dimension is ordinal

        Returns:
            GroupedChart object.
        """

        use_ordinal_axis = self._use_ordinal_axis(groupby)

        if use_ordinal_axis:
            ch = self._ordinal_difference_plot(level_1, level_2, absolute, groupby)
            chart_grid = ChartGrid()
            chart_grid.charts.append(ch)
        else:
            chart_grid = self._categorical_difference_plot(level_1, level_2, absolute, groupby)

        return chart_grid

    def _use_ordinal_axis(self, groupby):
        is_ordinal_difference_plot = (
            groupby is not None and self._ordinal_group_column is not None and self._ordinal_group_column in groupby
        )
        return is_ordinal_difference_plot

    def _ordinal_difference_plot(self, level_1, level_2, absolute, groupby):
        difference_df = self.difference(level_1, level_2, absolute, groupby)
        remaining_groups = self._remaining_categorical_groups(groupby)
        title = "Change from {} to {}".format(level_1, level_2)
        y_axis_label = self.get_difference_plot_label(absolute)
        ch = self._ordinal_plot(
            "difference",
            difference_df,
            groupby=None,
            level_name="",
            remaining_groups=remaining_groups,
            absolute=absolute,
            title=title,
            y_axis_label=y_axis_label,
        )
        ch.callout.line(0)

        return ch

    def get_difference_plot_label(self, absolute):
        change_type = "Absolute" if absolute else "Relative"
        return change_type + " change in {} / {}".format(self._numerator_column, self._denominator_column)

    @abstractmethod
    def _categorical_difference_plot(self, level_1, level_2, absolute, groupby):
        pass

    @abstractmethod
    def multiple_difference(self, level, absolute=True, groupby=None, level_as_reference=False):
        """The pairwise probability that the specific group
        is greater than all other groups.
        """
        pass

    def multiple_difference_plot(self, level, absolute=True, groupby=None, level_as_reference=False):
        """Compare level to all other groups or, if level_as_reference = True,
        all other groups to level.

        Args:
            level (str, tuple of str): Name of level.
            absolute (bool): If True then return the absolute
                difference (level2 - level1)
                otherwise return the relative difference (level2 / level1 - 1)
            groupby (str): Name of column, or list of columns.
                If specified, will return an interval for each level
                of the grouped dimension, or a confidence band if the
                grouped dimension is ordinal
            level_as_reference: If false (default), compare level to all other
             groups. If true, compare all other groups to level.
        """
        use_ordinal_axis = self._use_ordinal_axis(groupby)

        if use_ordinal_axis:
            ch = self._ordinal_multiple_difference_plot(level, absolute, groupby, level_as_reference)
            chart_grid = ChartGrid()
            chart_grid.charts.append(ch)
        else:
            chart_grid = self._categorical_multiple_difference_plot(level, absolute, groupby, level_as_reference)

        return chart_grid

    def _ordinal_multiple_difference_plot(self, level, absolute, groupby, level_as_reference):
        difference_df = self.multiple_difference(level, absolute, groupby, level_as_reference)
        remaining_groups = self._remaining_categorical_groups(groupby)
        groupby_columns = self._add_level_column(remaining_groups, level_as_reference)
        title = "Comparison to {}".format(level)
        y_axis_label = self.get_difference_plot_label(absolute)
        ch = self._ordinal_plot(
            "difference",
            difference_df,
            groupby=None,
            level_name="",
            remaining_groups=groupby_columns,
            absolute=absolute,
            title=title,
            y_axis_label=y_axis_label,
        )
        ch.callout.line(0)
        return ch

    def _add_level_column(self, groupby, level_as_reference):
        level_column = "level_2" if level_as_reference else "level_1"
        if groupby is None:
            groupby_columns = level_column
        else:
            if isinstance(groupby, str):
                groupby_columns = [groupby, level_column]
            else:
                groupby_columns = groupby + [level_column]
        return groupby_columns

    @abstractmethod
    def _categorical_multiple_difference_plot(self, level, absolute, groupby, level_as_reference):
        pass

    @staticmethod
    def _validate_levels(level_df, remaining_groups, level):
        try:
            level_df.groupby(remaining_groups).get_group(level)
        except (KeyError, ValueError):
            raise ValueError(
                """
                Invalid level: '{}'
                Must supply a level within the ungrouped dimensions: {}
                Valid levels:
                {}
                """.format(
                    level, remaining_groups, list(level_df.groupby(remaining_groups).groups.keys())
                )
            )

    def _groupby_iterator(self, input_function, groupby, **kwargs):
        groupby = [] if groupby is None else groupby
        # Will group over the whole dataframe if groupby is None
        level_groups = groupby if groupby else np.ones(len(self._data_frame))

        remaining_groups = [group for group in self._all_group_columns if group not in groupby and group is not None]

        for level_name, level_df in self._data_frame.groupby(level_groups):
            yield input_function(level_name, level_df, remaining_groups, groupby, **kwargs)

    def _iterate_groupby_to_chartgrid(self, input_function, groupby, **kwargs):
        """Iterate through groups in the test and apply the input function.

        Returns ChartGrid"""
        chart_grid = ChartGrid()

        chart_grid.charts = list(self._groupby_iterator(input_function, groupby, **kwargs))

        return chart_grid

    def _iterate_groupby_to_dataframe(self, input_function, groupby, **kwargs):
        """Iterate through groups in the test and apply the input function.

        Returns pd.DataFrame"""
        groupby_iterator = self._groupby_iterator(input_function, groupby, **kwargs)

        # Flatten any nested generators.
        groupby_iterator = list(groupby_iterator)
        if isinstance(groupby_iterator[0], types.GeneratorType):
            groupby_iterator = [group for generator in groupby_iterator for group in generator]

        results_data_frame = pd.concat(groupby_iterator, axis=0)

        results_data_frame = results_data_frame.reset_index(drop=True)

        return results_data_frame

    def _all_groups(self):
        """Return a list of all group keys.

        Returns: list"""
        groups = list(self._data_frame.groupby(self._all_group_columns).groups.keys())
        return groups

    def _add_group_by_columns(self, difference_df, groupby, level_name):
        if groupby:
            groupby = groupby[0] if len(groupby) == 1 else groupby
            if isinstance(groupby, str):
                difference_df.insert(0, column=groupby, value=level_name)
            else:
                for col, val in zip(groupby, level_name):
                    difference_df.insert(0, column=col, value=val)


# class BinomialResponse(BaseTest, metaclass=ABCMeta):
#     """Binomial Response Variable.
#     """

# class GaussianResponse(BaseTest, metaclass=ABCMeta):
#     """Base class for tests of normal response variables

#     E.g. Revenue per user
#     """

#     pass


# class PoissonResponse(BaseTest, metaclass=ABCMeta):
#     """Base class for tests of poisson response variables.

#     E.g. # of days active per user per month
#     """
#     pass


# class MultinomialResponse(BaseTest, metaclass=ABCMeta):
#     """Base class for tests of multinomial response variables.

#     E.g. single choice answer survey
#         self.
#     """

#     def __init__(self, data_frame, categorical_group_columns,
#                  ordinal_group_column, category_column, value_column):
#         self._category_column = category_column
#         self._value_column = value_column
#         super().__init__(data_frame, categorical_group_columns,
#                          ordinal_group_column)


# class CategoricalResponse(BaseTest, metaclass=ABCMeta):
#     """Base class for tests of categorical response variables.

#     E.g. multiple choice answer survey
#     """

#     def __init__(self, data_frame, categorical_group_columns,
#                  ordinal_group_column, category_column, value_column):
#         self._category_column = category_column
#         self._value_column = value_column
#         super().__init__(data_frame, categorical_group_columns,
#                          ordinal_group_column)

#     pass
