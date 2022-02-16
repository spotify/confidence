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

from typing import Union, Iterable, Tuple

import numpy as np
from bokeh.models import tools
from chartify import Chart
from pandas import DataFrame

from ..abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from ..confidence_utils import (
    axis_format_precision,
    add_color_column,
    get_remaning_groups,
    get_all_group_columns,
    listify,
    level2str,
    to_finite,
)
from ..constants import (
    POINT_ESTIMATE,
    DIFFERENCE,
    CI_LOWER,
    CI_UPPER,
    P_VALUE,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    ADJUSTED_P,
    NULL_HYPOTHESIS,
    NIM,
    NIM_TYPE,
)
from ...chartgrid import ChartGrid


class ChartifyGrapher(ConfidenceGrapherABC):
    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: str,
        denominator_column: str,
        categorical_group_columns: str,
        ordinal_group_column: str,
    ):

        self._df = data_frame
        self._numerator = numerator_column
        self._denominator = denominator_column
        self._categorical_group_columns = categorical_group_columns
        self._ordinal_group_column = ordinal_group_column
        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, self._ordinal_group_column)

    def plot_summary(self, summary_df: DataFrame, groupby: Union[str, Iterable]) -> ChartGrid:

        ch = ChartGrid()
        if groupby is None:
            ch.charts.append(self._summary_plot(level_name=None, level_df=summary_df, groupby=groupby))
        else:
            for level_name, level_df in summary_df.groupby(groupby):
                ch.charts.append(self._summary_plot(level_name=level_name, level_df=level_df, groupby=groupby))
        return ch

    def plot_difference(
        self, difference_df, absolute, groupby, nims: NIM_TYPE, use_adjusted_intervals: bool
    ) -> ChartGrid:
        if self._ordinal_group_column in listify(groupby):
            ch = self._ordinal_difference_plot(difference_df, absolute, groupby, use_adjusted_intervals)
            chart_grid = ChartGrid([ch])
        else:
            chart_grid = self._categorical_difference_plot(difference_df, absolute, groupby, use_adjusted_intervals)
        return chart_grid

    def plot_differences(
        self, difference_df, absolute, groupby, nims: NIM_TYPE, use_adjusted_intervals: bool
    ) -> ChartGrid:

        remaining_groups = get_remaning_groups(groupby, self._ordinal_group_column)
        groupby_columns = self._add_level_columns(remaining_groups)

        if self._ordinal_group_column in listify(groupby):
            ch = self._ordinal_difference_plot(difference_df, absolute, groupby_columns, use_adjusted_intervals)
            chart_grid = ChartGrid([ch])
        else:
            chart_grid = self._categorical_difference_plot(
                difference_df, absolute, groupby_columns, use_adjusted_intervals
            )
        return chart_grid

    def plot_multiple_difference(
        self,
        difference_df,
        absolute,
        groupby,
        level_as_reference,
        nims: NIM_TYPE,
        use_adjusted_intervals: bool,
    ) -> ChartGrid:
        if self._ordinal_group_column in listify(groupby):
            ch = self._ordinal_multiple_difference_plot(
                difference_df, absolute, groupby, level_as_reference, use_adjusted_intervals
            )
            chart_grid = ChartGrid([ch])
        else:
            chart_grid = self._categorical_multiple_difference_plot(
                difference_df, absolute, groupby, level_as_reference, use_adjusted_intervals
            )
        return chart_grid

    def _ordinal_difference_plot(
        self, difference_df: DataFrame, absolute: bool, groupby: Union[str, Iterable], use_adjusted_intervals: bool
    ) -> Chart:
        remaining_groups = get_remaning_groups(groupby, self._ordinal_group_column)

        if "level_1" in groupby and "level_2" in groupby:
            title = "Change from level_1 to level_2"
        else:
            title = "Change from {} to {}".format(
                difference_df["level_1"].values[0], difference_df["level_2"].values[0]
            )

        y_axis_label = self._get_difference_plot_label(absolute)
        ch = self._ordinal_plot(
            "difference",
            difference_df,
            groupby=None,
            level_name="",
            remaining_groups=remaining_groups,
            absolute=absolute,
            title=title,
            y_axis_label=y_axis_label,
            use_adjusted_intervals=use_adjusted_intervals,
        )
        ch.callout.line(0)

        return ch

    def _get_difference_plot_label(self, absolute):
        change_type = "Absolute" if absolute else "Relative"
        return change_type + " change in {} / {}".format(self._numerator, self._denominator)

    def _categorical_difference_plot(
        self, difference_df: DataFrame, absolute: bool, groupby: Union[str, Iterable], use_adjusted_intervals: bool
    ) -> ChartGrid:
        if groupby is None:
            groupby = "dummy_groupby"
            difference_df[groupby] = "Difference"

        if "level_1" in groupby and "level_2" in groupby:
            title = "Change from level_1 to level_2"
        else:
            title = "Change from {} to {}".format(
                difference_df["level_1"].values[0], difference_df["level_2"].values[0]
            )
        x_label = "" if groupby is None else "{}".format(groupby)

        chart_grid = self._categorical_difference_chart(
            absolute, difference_df, groupby, title, x_label, use_adjusted_intervals
        )

        return chart_grid

    def _categorical_difference_chart(
        self,
        absolute: bool,
        difference_df: DataFrame,
        groupby_columns: Union[str, Iterable],
        title: str,
        x_label: str,
        use_adjusted_intervals: bool,
    ) -> ChartGrid:
        LOWER, UPPER = (ADJUSTED_LOWER, ADJUSTED_UPPER) if use_adjusted_intervals else (CI_LOWER, CI_UPPER)
        axis_format, y_min, y_max = axis_format_precision(
            numbers=(
                difference_df[LOWER]
                .append(difference_df[DIFFERENCE])
                .append(difference_df[UPPER])
                .append(difference_df[NULL_HYPOTHESIS] if NULL_HYPOTHESIS in difference_df.columns else None)
            ),
            absolute=absolute,
        )

        df = (
            difference_df.assign(**{LOWER: to_finite(difference_df[LOWER], y_min)})
            .assign(**{UPPER: to_finite(difference_df[UPPER], y_max)})
            .assign(level_1=difference_df.level_1.map(level2str))
            .assign(level_2=difference_df.level_2.map(level2str))
            .set_index(groupby_columns)
            .assign(categorical_x=lambda df: df.index.to_numpy())
            .reset_index()
        )

        ch = Chart(x_axis_type="categorical")
        ch.plot.interval(
            data_frame=df.sort_values(groupby_columns),
            categorical_columns=groupby_columns,
            lower_bound_column=LOWER,
            upper_bound_column=UPPER,
            middle_column=DIFFERENCE,
            categorical_order_by="labels",
            categorical_order_ascending=False,
        )
        # Also plot transparent circles, just to be able to show hover box
        ch.style.color_palette.reset_palette_order()
        ch.figure.circle(
            source=df, x="categorical_x", y=DIFFERENCE, size=20, name="center", line_alpha=0, fill_alpha=0
        )
        if NULL_HYPOTHESIS in df.columns:
            ch.style.color_palette.reset_palette_order()
            dash_source = (
                df[~df[NIM].isna()]
                .assign(
                    color_column=lambda df: df.apply(
                        lambda row: "red"
                        if row[LOWER] < row[NULL_HYPOTHESIS] and row[NULL_HYPOTHESIS] < row[UPPER]
                        else "green",
                        axis=1,
                    )
                )
                .sort_values(groupby_columns)
            )
            ch.figure.dash(
                source=dash_source,
                x="categorical_x",
                y=NULL_HYPOTHESIS,
                size=320 / len(df),
                line_width=3,
                name="nim",
                line_color="color_column",
            )
        ch.axes.set_yaxis_label(self._get_difference_plot_label(absolute))
        ch.set_source_label("")
        ch.callout.line(0)
        ch.axes.set_yaxis_range(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        ch.axes.set_yaxis_tick_format(axis_format)
        ch.set_title(title)
        ch.axes.set_xaxis_label(x_label)
        ch.set_subtitle("")

        self.add_tools(
            chart=ch,
            df=(
                difference_df.set_index(groupby_columns)
                .assign(categorical_x=lambda df: df.index.to_numpy())
                .reset_index()
            ),
            center_name=DIFFERENCE,
            absolute=absolute,
            ordinal=False,
            use_adjusted_intervals=use_adjusted_intervals,
        )

        chart_grid = ChartGrid()
        chart_grid.charts.append(ch)

        return chart_grid

    def _summary_plot(self, level_name: Union[str, Tuple], level_df: DataFrame, groupby: Union[str, Iterable]):
        remaining_groups = get_remaning_groups(self._all_group_columns, groupby)
        if self._ordinal_group_column is not None and self._ordinal_group_column in remaining_groups:

            ch = self._ordinal_summary_plot(level_name, level_df, remaining_groups, groupby)
        else:
            ch = self._categorical_summary_plot(level_name, level_df, remaining_groups, groupby)
        return ch

    def _ordinal_summary_plot(
        self,
        level_name: Union[str, Tuple],
        level_df: DataFrame,
        remaining_groups: Union[str, Iterable],
        groupby: Union[str, Iterable],
    ):
        remaining_groups = get_remaning_groups(remaining_groups, self._ordinal_group_column)
        title = "Estimate of {} / {}".format(self._numerator, self._denominator)
        y_axis_label = "{} / {}".format(self._numerator, self._denominator)
        return self._ordinal_plot(
            POINT_ESTIMATE,
            level_df,
            groupby,
            level_name,
            remaining_groups,
            absolute=True,
            title=title,
            y_axis_label=y_axis_label,
            use_adjusted_intervals=False,
        )

    def _ordinal_plot(
        self,
        center_name: str,
        level_df: DataFrame,
        groupby: Union[str, Iterable],
        level_name: Union[str, Tuple],
        remaining_groups: Union[str, Iterable],
        absolute: bool,
        title: str,
        y_axis_label: str,
        use_adjusted_intervals: bool,
    ):
        LOWER, UPPER = (ADJUSTED_LOWER, ADJUSTED_UPPER) if use_adjusted_intervals else (CI_LOWER, CI_UPPER)
        df = add_color_column(level_df, remaining_groups)
        colors = "color" if remaining_groups else None
        axis_format, y_min, y_max = axis_format_precision(
            numbers=(
                df[LOWER]
                .append(df[center_name])
                .append(df[UPPER])
                .append(df[NULL_HYPOTHESIS] if NULL_HYPOTHESIS in df.columns else None)
            ),
            absolute=absolute,
        )
        ch = Chart(x_axis_type=self._ordinal_type())
        ch.plot.line(
            data_frame=df.sort_values(self._ordinal_group_column),
            x_column=self._ordinal_group_column,
            y_column=center_name,
            color_column=colors,
        )
        ch.style.color_palette.reset_palette_order()
        ch.plot.area(
            data_frame=(
                df.assign(**{LOWER: to_finite(df[LOWER], y_min)})
                .assign(**{UPPER: to_finite(df[UPPER], y_max)})
                .sort_values(self._ordinal_group_column)
            ),
            x_column=self._ordinal_group_column,
            y_column=LOWER,
            second_y_column=UPPER,
            color_column=colors,
        )
        if NULL_HYPOTHESIS in df.columns:
            ch.style.color_palette.reset_palette_order()
            ch.plot.line(
                data_frame=df.sort_values(self._ordinal_group_column),
                x_column=self._ordinal_group_column,
                y_column=NULL_HYPOTHESIS,
                color_column=colors,
                line_dash="dashed",
                line_width=1,
            )
        ch.axes.set_yaxis_label(y_axis_label)
        ch.axes.set_xaxis_label(self._ordinal_group_column)
        ch.set_source_label("")
        ch.axes.set_yaxis_range(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        ch.axes.set_yaxis_tick_format(axis_format)
        subtitle = "" if not groupby else "{}: {}".format(groupby, level_name)
        ch.set_subtitle(subtitle)
        ch.set_title(title)
        if colors:
            ch.set_legend_location("outside_bottom")
        self.add_tools(
            chart=ch,
            df=df,
            center_name=center_name,
            absolute=absolute,
            ordinal=True,
            use_adjusted_intervals=use_adjusted_intervals,
        )
        return ch

    def _categorical_summary_plot(self, level_name, summary_df, remaining_groups, groupby):
        if not remaining_groups:
            remaining_groups = listify(groupby)
        df = summary_df.set_index(remaining_groups).assign(categorical_x=lambda df: df.index.to_numpy()).reset_index()

        axis_format, y_min, y_max = axis_format_precision(
            numbers=(df[CI_LOWER].append(df[POINT_ESTIMATE]).append(df[CI_UPPER])), absolute=True
        )

        ch = Chart(x_axis_type="categorical")
        ch.plot.interval(
            (
                df.assign(**{CI_LOWER: to_finite(df[CI_LOWER], y_min)}).assign(
                    **{CI_UPPER: to_finite(df[CI_UPPER], y_max)}
                )
            ),
            categorical_columns=remaining_groups,
            lower_bound_column=CI_LOWER,
            upper_bound_column=CI_UPPER,
            middle_column=POINT_ESTIMATE,
            categorical_order_by="labels",
            categorical_order_ascending=True,
        )
        # Also plot transparent circles, just to be able to show hover box
        ch.style.color_palette.reset_palette_order()
        ch.figure.circle(
            source=df, x="categorical_x", y=POINT_ESTIMATE, size=20, name="center", line_alpha=0, fill_alpha=0
        )
        ch.set_title("Estimate of {} / {}".format(self._numerator, self._denominator))
        if groupby:
            ch.set_subtitle("{}: {}".format(groupby, level_name))
        else:
            ch.set_subtitle("")
        ch.axes.set_xaxis_label("{}".format(", ".join(remaining_groups)))
        ch.axes.set_yaxis_label("{} / {}".format(self._numerator, self._denominator))
        ch.set_source_label("")
        ch.axes.set_yaxis_tick_format(axis_format)
        self.add_tools(
            chart=ch, df=df, center_name=POINT_ESTIMATE, absolute=True, ordinal=False, use_adjusted_intervals=False
        )
        return ch

    def _ordinal_type(self):
        ordinal_column_type = self._df[self._ordinal_group_column].dtype.type
        axis_type = "datetime" if issubclass(ordinal_column_type, np.datetime64) else "linear"
        return axis_type

    def _ordinal_multiple_difference_plot(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Union[str, Iterable],
        level_as_reference: bool,
        use_adjusted_intervals: bool,
    ):
        remaining_groups = get_remaning_groups(groupby, self._ordinal_group_column)
        groupby_columns = self._add_level_column(remaining_groups, level_as_reference)
        title = self._get_multiple_difference_title(difference_df, level_as_reference)
        y_axis_label = self._get_difference_plot_label(absolute)
        ch = self._ordinal_plot(
            DIFFERENCE,
            difference_df,
            groupby=None,
            level_name="",
            remaining_groups=groupby_columns,
            absolute=absolute,
            title=title,
            y_axis_label=y_axis_label,
            use_adjusted_intervals=use_adjusted_intervals,
        )
        ch.callout.line(0)
        return ch

    def _categorical_multiple_difference_plot(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Union[str, Iterable],
        level_as_reference: bool,
        use_adjusted_intervals: bool,
    ):
        groupby_columns = self._add_level_column(groupby, level_as_reference)
        title = self._get_multiple_difference_title(difference_df, level_as_reference)
        x_label = "" if groupby is None else "{}".format(groupby)
        chart_grid = self._categorical_difference_chart(
            absolute, difference_df, groupby_columns, title, x_label, use_adjusted_intervals
        )

        return chart_grid

    def _get_multiple_difference_title(self, difference_df, level_as_reference):
        reference_level = "level_1" if level_as_reference else "level_2"
        title = "Comparison to {}".format(difference_df[reference_level].values[0])
        return title

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

    def _add_level_columns(self, groupby):
        levels = ["level_1", "level_2"]
        if groupby is None:
            groupby_columns = levels
        else:
            if isinstance(groupby, str):
                groupby_columns = [groupby] + levels
            else:
                groupby_columns = groupby + levels
        return groupby_columns

    def add_ci_to_chart_datasources(
        self, chart: Chart, df: DataFrame, center_name: str, ordinal: bool, use_adjusted_intervals: bool
    ):
        LOWER, UPPER = (ADJUSTED_LOWER, ADJUSTED_UPPER) if use_adjusted_intervals else (CI_LOWER, CI_UPPER)
        group_col = "color" if ordinal and "color" in df.columns else "categorical_x"
        for data in chart.data:
            if center_name in data.keys() or NULL_HYPOTHESIS in data.keys():
                index = data["index"]
                data[LOWER] = np.array(df[LOWER][index])
                data[UPPER] = np.array(df[UPPER][index])
                data["color"] = np.array(df[group_col][index])
            if DIFFERENCE in data.keys() or NULL_HYPOTHESIS in data.keys():
                index = data["index"]
                data[DIFFERENCE] = np.array(df[DIFFERENCE][index])
                data["p_value"] = np.array(df[P_VALUE][index])
                data["adjusted_p"] = np.array(df[ADJUSTED_P][index])
                if NULL_HYPOTHESIS in df.columns:
                    data["null_hyp"] = np.array(df[NULL_HYPOTHESIS][index])

    def add_tools(
        self,
        chart: Chart,
        df: DataFrame,
        center_name: str,
        absolute: bool,
        ordinal: bool,
        use_adjusted_intervals: bool,
    ):
        self.add_ci_to_chart_datasources(chart, df, center_name, ordinal, use_adjusted_intervals)
        LOWER, UPPER = (ADJUSTED_LOWER, ADJUSTED_UPPER) if use_adjusted_intervals else (CI_LOWER, CI_UPPER)

        if len(chart.figure.legend) > 0:
            chart.figure.legend.click_policy = "hide"
        axis_format, y_min, y_max = axis_format_precision(
            numbers=(
                df[LOWER]
                .append(df[center_name])
                .append(df[UPPER])
                .append(df[NULL_HYPOTHESIS] if NULL_HYPOTHESIS in df.columns else None)
            ),
            absolute=absolute,
            extra_zeros=2,
        )
        ordinal_tool_tip = [] if not ordinal else [(self._ordinal_group_column, f"@{self._ordinal_group_column}")]
        p_value_tool_tip = (
            (
                [("p-value", "@p_value{0.0000}")]
                + ([("adjusted p-value", "@adjusted_p{0.0000}")] if len(df) > 1 else [])
            )
            if center_name == DIFFERENCE
            else []
        )
        nim_tool_tip = [("null hypothesis", f"@null_hyp{{{axis_format}}}")] if NULL_HYPOTHESIS in df.columns else []
        tooltips = (
            [("group", "@color")]
            + ordinal_tool_tip
            + [(f"{center_name}", f"@{center_name}{{{axis_format}}}")]
            + [
                (
                    ("adjusted " if use_adjusted_intervals else "") + "confidence interval",
                    f"(@{{{LOWER}}}{{{axis_format}}}," f" @{{{UPPER}}}{{{axis_format}}})",
                )
            ]
            + p_value_tool_tip
            + nim_tool_tip
        )
        lines_with_hover = [] if ordinal else ["center", "nim"]
        hover = tools.HoverTool(tooltips=tooltips, names=lines_with_hover)
        box_zoom = tools.BoxZoomTool()

        chart.figure.add_tools(
            hover, tools.ZoomInTool(), tools.ZoomOutTool(), box_zoom, tools.PanTool(), tools.ResetTool()
        )
        chart.figure.toolbar.active_drag = box_zoom
