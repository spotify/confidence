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

import chartify
import numpy as np
import pandas as pd
from scipy.stats import beta

from spotify_confidence.analysis.bayesian.bayesian_base import BaseTest, \
    randomization_warning_decorator, axis_format_precision


class BetaBinomial(BaseTest):
    def __init__(self,
                 data_frame,
                 numerator_column,
                 denominator_column,
                 categorical_group_columns=None,
                 ordinal_group_column=None,
                 prior_alpha_column=None,
                 prior_beta_column=None,
                 interval_size=0.95):
        """
        Bayesian BetaBinomial model.

        See: https://en.wikipedia.org/wiki/Beta-binomial_distribution

        data_frame (pd.DataFrame): DataFrame
        numerator_column (str): Column name for numerator column.
        denominator_column (str): Column name for denominator column.
        categorical_group_columns (str or list): Column names
            for categorical groupings.
        ordinal_group_column (str): Column name for ordinal
            grouping (e.g. numeric or date values).
        prior_alpha_column (str): Column name to use for prior alpha.
        prior_beta_column (str): Column name to use for prior beta.
        interval_size (float): Size of credible intervals. Default 0.95
        """
        super().__init__(data_frame, categorical_group_columns,
                         ordinal_group_column,
                         numerator_column,
                         denominator_column,
                         interval_size)

        self._monte_carlo_sample_size = 500000

        # Initialize priors.
        if prior_alpha_column is None or prior_beta_column is None:
            self._alpha_prior, self._beta_prior = (0.5, 0.5)
        else:
            self._alpha_prior = data_frame[prior_alpha_column]
            self._beta_prior = data_frame[prior_beta_column]

    def _interval(self, row):
        interval = beta.interval(
            self._interval_size,
            row[self._numerator_column] + self._alpha_prior,
            row[self._denominator_column] - row[self._numerator_column] +
            self._beta_prior)
        return interval

    def _posterior_parameters(self, group_df):
        """Calculate parameters of posterior distribution.

        Returns:
            tuple of floats: posterior_alpha, posterior_beta"""
        numerator = group_df[self._numerator_column].values[0]
        denominator = group_df[self._denominator_column].values[0]
        posterior_alpha = numerator + self._alpha_prior
        posterior_beta = denominator - numerator + self._beta_prior
        return posterior_alpha, posterior_beta

    def _beta_pdf(self, group_df):
        """Beta pdfs for the given dataframe"""
        posterior_alpha, posterior_beta = self._posterior_parameters(group_df)
        epsilon = .001
        lower_range = beta.isf(1.0 - epsilon, posterior_alpha, posterior_beta)
        upper_range = beta.isf(epsilon, posterior_alpha, posterior_beta)
        x_range = np.linspace(lower_range, upper_range, 1000)
        beta_pdf = [
            beta.pdf(x, posterior_alpha, posterior_beta) for x in x_range
        ]
        beta_dist = pd.DataFrame({'x': x_range, 'y': beta_pdf})
        return beta_dist

    def _sample_posterior(self, group_df, posterior_sample_size=None):
        """MCMC sampling of posterior distribution.
        Used to calculate the posterior distribution of
        the difference in Beta RVs.

        Arguments:
            - seed (int): Seed for random number generator.
                Set it to make the posteriors deterministic.
            - posterior_sample_size (int): Number of posterior
                samples (affects precision)
        """
        if posterior_sample_size is None:
            posterior_sample_size = self._monte_carlo_sample_size
        posterior_alpha, posterior_beta = self._posterior_parameters(group_df)
        posterior_samples = np.random.beta(
            posterior_alpha, posterior_beta, size=posterior_sample_size)
        return posterior_samples

    def _categorical_summary_plot(self, level_name, level_df, remaining_groups,
                                  groupby):

        if not remaining_groups:
            remaining_groups = groupby
        grouped_df = level_df.groupby(remaining_groups)

        distributions = pd.DataFrame()
        for group_name, group_df in grouped_df:
            beta_dist = self._beta_pdf(group_df)
            beta_dist['group'] = str(group_name)
            distributions = pd.concat([distributions, beta_dist], axis=0)

        # Filter out the long tails of the distributions
        filtered_xs = distributions.groupby('x')['y'].max().reset_index().loc[
            lambda x: x['y'] > .01]
        distributions = distributions[distributions['x'].isin(
            filtered_xs['x'])]

        # Remove legend if only one color
        color_column = 'group' if len(grouped_df) > 1 else None

        ch = chartify.Chart()
        ch.plot.area(
            distributions,
            'x',
            'y',
            color_column=color_column,
            stacked=False,
            color_order=[str(x) for x in list(grouped_df.groups.keys())])
        ch.set_title("Estimate of {} / {}".format(self._numerator_column,
                                                  self._denominator_column))

        if groupby:
            ch.set_subtitle("{}: {}".format(groupby, level_name))
        else:
            ch.set_subtitle("")
        ch.axes.set_xaxis_label("{} / {}".format(self._numerator_column,
                                                 self._denominator_column))
        ch.axes.set_yaxis_label("Probability Density")
        ch.set_source_label("")
        ch.axes.set_yaxis_range(0)
        axis_format = axis_format_precision(distributions['x'].min(),
                                            distributions['x'].max(),
                                            absolute=True)
        ch.axes.set_xaxis_tick_format(axis_format)

        ch.style.color_palette.reset_palette_order()

        # Plot callouts for the means
        for group_name, group_df in grouped_df:
            posterior_alpha, posterior_beta = self._posterior_parameters(
                group_df)
            posterior_mean = posterior_alpha / (
                posterior_alpha + posterior_beta)
            density = beta.pdf(posterior_mean, posterior_alpha, posterior_beta)
            ch.callout.line(
                posterior_mean,
                orientation='height',
                line_color=ch.style.color_palette.next_color(),
                line_dash='dashed')
            ch.callout.text('{0:.1f}%'.format(posterior_mean * 100),
                            posterior_mean, density)

        ch.axes.hide_yaxis()
        if color_column:
            ch.set_legend_location('outside_bottom')
        return ch

    def _difference_posteriors(self, data, level_1, level_2, absolute=True):

        posterior_1 = self._sample_posterior(data.get_group(level_1))
        posterior_2 = self._sample_posterior(data.get_group(level_2))

        if absolute:
            difference_posterior = posterior_2 - posterior_1
        else:
            difference_posterior = posterior_2 / posterior_1 - 1.

        return difference_posterior

    def _differences(self, difference_posterior, level_1, level_2, absolute):
        # 95% credible interval for posterior
        credible_interval = (pd.Series(difference_posterior).quantile(
            (1. - self._interval_size) / 2),
            pd.Series(difference_posterior).quantile(
                (1. - self._interval_size) / 2 + self._interval_size))

        # Probability that posterior is greater
        # than zero (count occurences in the MC sample)
        p_gt_zero = (difference_posterior > 0).mean()

        expected_loss_v2 = (
            difference_posterior[difference_posterior < 0].sum() /
            len(difference_posterior))
        if (difference_posterior > 0).sum() == 0:
            expected_gain_v2 = 0
        else:
            expected_gain_v2 = (
                difference_posterior[difference_posterior > 0].sum() /
                len(difference_posterior))

        expected_loss_v1 = (
            (difference_posterior[difference_posterior * -1.0 < 0] *
             -1.0).sum() / len(difference_posterior))

        if (difference_posterior * -1.0 > 0).sum() == 0:
            expected_gain_v1 = 0
        else:
            expected_gain_v1 = (
                (difference_posterior[difference_posterior * -1.0 > 0] *
                 -1.0).sum() / len(difference_posterior))

        return pd.DataFrame(
            OrderedDict(
                [('level_1', str(level_1)), ('level_2', str(level_2)),
                 ('absolute_difference',
                  absolute), ('difference', difference_posterior.mean()),
                 ('ci_lower', [credible_interval[0]]), ('ci_upper', [
                     credible_interval[1]
                 ]), ('P(level_2 > level_1)', p_gt_zero),
                 ('level_1 potential loss',
                  expected_loss_v1), ('level_1 potential gain',
                                      expected_gain_v1),
                 ('level_2 potential loss',
                  expected_loss_v2), ('level_2 potential gain',
                                      expected_gain_v2)]))

    def _difference(self, level_name, level_df, remaining_groups, groupby,
                    level_1, level_2, absolute):

        difference_df, _ = self._difference_and_difference_posterior(
                                                            level_df,
                                                            remaining_groups,
                                                            level_2,
                                                            level_1,
                                                            absolute)

        self._add_group_by_columns(difference_df, groupby, level_name)

        return difference_df

    def _difference_and_difference_posterior(self, level_df, remaining_groups,
                                             level_2, level_1, absolute):
        self._validate_levels(level_df, remaining_groups, level_1)
        self._validate_levels(level_df, remaining_groups, level_2)
        # difference is posterior_2 - posterior_1
        difference_posterior = self._difference_posteriors(
            level_df.groupby(remaining_groups), level_1, level_2, absolute)
        difference_df = self._differences(difference_posterior, level_1,
                                          level_2, absolute)
        return difference_df, difference_posterior

    @randomization_warning_decorator
    def difference(self, level_1, level_2, absolute=True, groupby=None):
        """Return DataFrame with summary statistics of the difference between
        level 1 and level 2.

        Args:
            level_1 (str, tuple of str): Name of first level.
            level_2 (str, tuple of str): Name of second level.
            absolute (bool): If True then return the
                absolute difference (level2 - level1)
                otherwise return the relative difference (level2 / level1 - 1)
            groupby (str): Name of column.
                If specified, will return the difference for each level
                of the grouped dimension.

        Returns:
            Pandas DataFrame with the following columns:
            - level_1: Name of level 1.
            - level_2: Name of level 2.
            - absolute_difference: True if absolute.
                Absolute: level2 - level1
                Relative: level2 / level1 - 1
            - difference: Best estimate of the difference between level 2 and 1.
                Posterior mean of the difference between level 1 and level 2.
                https://en.wikipedia.org/wiki/Bayes_estimator
            - ci_lower: Lower credible interval bound of the difference.
            - ci_upper: Upper credible interval bound of the difference.
            - P(level_2 > level_1): Probability that the level 2 > level 1.
            - level_1 potential loss: The expected loss if we
                switch to level 1, but level 2 is actually better.
            - level_1 potential gain: The expected gain if we
                switch to level 1, and level 1 is actually better.
            - level_2 potential loss: The expected loss if we
                switch to level 2, but level 1 is actually better.
            - level_2 potential gain: The expected gain if we
                switch to level 2, and level 2 is actually better.
        """

        results_df = self._iterate_groupby_to_dataframe(
            self._difference,
            groupby=groupby,
            level_1=level_1,
            level_2=level_2,
            absolute=absolute)

        return results_df

    @randomization_warning_decorator
    def _categorical_difference_plot(self, level_1, level_2, absolute, groupby):
        chart_grid = self._iterate_groupby_to_chartgrid(
            self._categorical_difference_plot_,
            groupby=groupby,
            level_1=level_1,
            level_2=level_2,
            absolute=absolute)

        return chart_grid

    def _categorical_difference_plot_(self, level_name, level_df,
                                      remaining_groups, groupby,
                                      level_1, level_2, absolute):
        difference_df, difference_posterior = \
            self._difference_and_difference_posterior(level_df,
                                                      remaining_groups,
                                                      level_2,
                                                      level_1,
                                                      absolute)

        posterior_mean = difference_df['difference'][0]
        # potential_loss = difference_df['{} potential loss'.format(level_2)][0]

        # Take the difference posterior and create a chart
        df = pd.DataFrame({'values': difference_posterior})

        ch = chartify.Chart(y_axis_type='density', x_axis_type='linear')

        ch.plot.kde(df, 'values')

        ch.set_title("Change from {} to {}".format(level_1, level_2))

        subtitle = "" if not groupby else "{}: {}".format(groupby, level_name)
        ch.set_subtitle(subtitle)

        # Line at no difference
        ch.callout.line(0,
                        orientation='height',
                        line_color='black',
                        line_dash='dashed')
        # ch.callout.text('No change', 0, .5, angle=90)

        # Plot callout for the mean
        ch.callout.line(
            posterior_mean,
            orientation='height',
            line_color=ch.style.color_palette._colors[0],
            line_dash='dashed')
        # ch.callout.text(
        #     '{0:.2f}%'.format(posterior_mean * 100), posterior_mean, 0)
        ch.callout.text(
            'Expected change: {0:.2f}%'.format(posterior_mean * 100),
            posterior_mean,
            0,
            angle=90)

        # ch.callout.line(
        #     potential_loss,
        #     orientation='height',
        #     line_color=ch.style.color_palette._colors[1])
        # ch.callout.text(
        #     'Potential Loss: {0:.2f}%'.format(potential_loss * 100),
        #     potential_loss,
        #     1.5,
        #     angle=90)
        # ch.callout.text(
        #     '{0:.2f}%'.format(potential_loss * 100), potential_loss, 1.)

        ch.set_source_label("")
        ch.axes.set_yaxis_range(0)
        ch.axes.set_xaxis_label(self.get_difference_plot_label(absolute))
        ch.axes.set_yaxis_label("Probability Density")
        ch.axes.hide_yaxis()
        axis_format = axis_format_precision(df['values'].max() * 10,
                                            df['values'].min() * 10,
                                            absolute)
        ch.axes.set_xaxis_tick_format(axis_format)

        return ch

    def _multiple_difference_joint_dataframe(self, *args, **kwargs):

        return self._multiple_difference_joint_base(*args, **kwargs)[0]

    def _multiple_difference_joint_base(self,
                                        level_name,
                                        level_df,
                                        remaining_groups,
                                        groupby,
                                        level,
                                        absolute):

        grouped_df = level_df.groupby(remaining_groups)

        grouped_df_keys = tuple(grouped_df.groups.keys())

        self._validate_levels(level_df, remaining_groups, level)

        posteriors = [
            self._sample_posterior(grouped_df.get_group(level))
            for level in grouped_df_keys
        ]

        var_indx = grouped_df_keys.index(level)
        other_indx = [
            i for i, value in enumerate(grouped_df_keys) if value != level
        ]

        posterior_matrix = np.vstack(posteriors)

        ge_bool_matrix = posterior_matrix[var_indx, :] >= posterior_matrix[:, :]

        best_arr = ge_bool_matrix.all(axis=0)

        p_ge_all = best_arr.mean()

        end_value = posterior_matrix[var_indx]
        start_value = posterior_matrix[other_indx].max(axis=0)

        if absolute:
            difference_posterior = end_value - start_value
        else:
            difference_posterior = end_value / start_value - 1

        # E(level - best level | level != best)
        if not (~best_arr).sum():
            expected_loss = 0
        else:
            expected_loss = difference_posterior[~best_arr].mean()

        # E(level - median level | level = best)
        if not (best_arr).sum():
            expected_gain = 0
        else:
            expected_gain = difference_posterior[best_arr].mean()

        expectation = difference_posterior.mean()
        ci_l_expectation = pd.Series(difference_posterior).quantile(
            (1. - self._interval_size) / 2)
        ci_u_expectation = pd.Series(difference_posterior).quantile(
            (1. - self._interval_size) / 2 + self._interval_size)

        difference_df = pd.DataFrame(
            OrderedDict([
                ('level', [str(level)]),
                ('absolute_difference', absolute),
                ('difference', expectation),
                ('ci_lower', ci_l_expectation),
                ('ci_upper', ci_u_expectation),
                ('P({} >= all)'.format(level), p_ge_all),
                ('{} potential loss'.format(level), expected_loss),
                ('{} potential gain'.format(level), expected_gain),
            ]))
        self._add_group_by_columns(difference_df, groupby, level_name)

        return (difference_df, difference_posterior)

    @randomization_warning_decorator
    def multiple_difference_joint(self,
                                  level,
                                  absolute=True,
                                  groupby=None):
        """Calculate the joint probability that the given level is greater
        than all other levels in the test.

        Args:
            level (str, tuple of str): Name of level.
            absolute (bool): If True then return the absolute difference
                otherwise return the relative difference.
            groupby (str): Name of column.
                If specified, will return an interval for each level
                of the grouped dimension.

        Returns:
            Pandas DataFrame with the following columns:
            - level: Name of level
            - absolute_difference: True if absolute.
                Absolute: level2 - level1
                Relative: level2 / level1 - 1
            - difference: Difference between the level and the best performing
                among the other levels.
            - ci_lower: Lower credible interval bound of the difference.
            - ci_upper: Upper credible interval bound of the difference.
            - P(level > all): Probability that the level > all other levels.
            - potential loss: The expected loss if we
                switch to level, but some other level is actually better.
            - potential gain: The expected gain if we
                switch to level, and it is actually the best.
        """

        results_df = self._iterate_groupby_to_dataframe(
            self._multiple_difference_joint_dataframe,
            groupby=groupby,
            level=level,
            absolute=absolute
        )

        return results_df

    def _multiple_difference_joint_plot(self,
                                        level_name,
                                        level_df,
                                        remaining_groups,
                                        groupby,
                                        level,
                                        absolute):

        self._validate_levels(level_df, remaining_groups, level)

        difference_df, difference_posterior = \
            self._multiple_difference_joint_base(level_name,
                                                 level_df,
                                                 remaining_groups,
                                                 groupby,
                                                 level,
                                                 absolute)

        posterior_mean = difference_df.loc[:, 'difference'].values[0]

        # potential_loss = difference_df.loc[:, '{} potential loss'.format(
        #     level)].values[0]

        # Take the difference posterior and create a chart
        df = pd.DataFrame({'values': difference_posterior})

        ch = chartify.Chart(y_axis_type='density', x_axis_type='linear')

        ch.plot.kde(df, 'values')

        ch.set_title("Comparison to {}".format(level))

        subtitle = "" if not groupby else "{}: {}".format(groupby, level_name)
        ch.set_subtitle(subtitle)

        # Line at no difference
        ch.callout.line(0, orientation='height', line_color='black')

        # Plot callout for the mean
        ch.callout.line(
            posterior_mean,
            orientation='height',
            line_color=ch.style.color_palette._colors[0])

        ch.callout.text(
            'Expected change: {0:.2f}%'.format(posterior_mean * 100),
            posterior_mean,
            0,
            angle=90)

        ch.set_source_label("")
        ch.axes.set_yaxis_range(0)
        ch.axes.set_xaxis_label(self.get_difference_plot_label(absolute))
        ch.axes.set_yaxis_label("Probability Density")
        ch.axes.hide_yaxis()

        axis_format = axis_format_precision(df['values'].max() * 10,
                                            df['values'].min() * 10,
                                            absolute)
        ch.axes.set_xaxis_tick_format(axis_format)
        return ch

    @randomization_warning_decorator
    def multiple_difference_joint_plot(self,
                                       level,
                                       absolute=True,
                                       groupby=None):
        """Calculate the joint probability that the given level is greater
        than all other levels in the test.

        Args:
            level (str, tuple of str): Name of level.
            absolute (bool): If True then return the absolute difference
                otherwise return the relative difference.
            groupby (str): Name of column.
                If specified, will return an interval for each level
                of the grouped dimension.

        Returns:
            GroupedChart object.
        """

        results_df = self._iterate_groupby_to_chartgrid(
            self._multiple_difference_joint_plot,
            groupby=groupby,
            level=level,
            absolute=absolute
        )

        return results_df

    def _multiple_difference(self, level_name, level_df, remaining_groups,
                             groupby, level, absolute, level_as_reference):

        grouped_df = level_df.groupby(remaining_groups)

        grouped_df_keys = tuple(grouped_df.groups.keys())

        other_keys = [
            value for i, value in enumerate(grouped_df_keys) if value != level
        ]

        for key in other_keys:

            # Switch the subtraction order as specified.
            start_value, end_value = level, key
            if not level_as_reference:
                start_value, end_value = end_value, start_value

            difference_df = self._difference(
                level_name,
                level_df,
                remaining_groups,
                groupby,
                start_value,
                end_value,
                absolute=absolute)

            yield difference_df

    @randomization_warning_decorator
    def multiple_difference(self,
                            level,
                            absolute=True,
                            groupby=None,
                            level_as_reference=False):
        """Pairwise comparison of the given level to all others.

        Args:
            level (str, tuple of str): Name of level.
            absolute (bool): If True then return the absolute difference
                otherwise return the relative difference.
            groupby (str): Name of column.
                If specified, will return an interval for each level
                of the grouped dimension.
            level_as_reference (bool): If True, the given level is the reference
                value for the change. (level1)

        Returns:
            Pandas DataFrame with the following columns:
            - groupby (If groupby is not None): Grouped dimension
            - level_1: Name of level 1.
            - level_2: Name of level 2.
            - absolute_difference: True if absolute.
                Absolute: level2 - level1
                Relative: level2 / level1 - 1
            - difference: Best estimate of the difference between level 2 and 1.
                Posterior mean of the difference between level 1 and level 2.
                https://en.wikipedia.org/wiki/Bayes_estimator
            - ci_lower: Lower credible interval bound of the difference.
            - ci_upper: Upper credible interval bound of the difference.
            - P(level_2 > level_1): Probability that the level 2 > level 1.
            - level_1 potential loss: The expected loss if we
                switch to level 1, but level 2 is actually better.
            - level_1 potential gain: The expected gain if we
                switch to level 1, and level 1 is actually better.
            - level_2 potential loss: The expected loss if we
                switch to level 2, but level 1 is actually better.
            - level_2 potential gain: The expected gain if we
                switch to level 2, and level 2 is actually better.
        """

        results_df = self._iterate_groupby_to_dataframe(
            self._multiple_difference,
            groupby=groupby,
            level=level,
            absolute=absolute,
            level_as_reference=level_as_reference)

        results_df = results_df.reset_index(drop=True)

        return results_df

    def _categorical_multiple_difference_chart(self, level_name, level_df,
                                               remaining_groups, groupby, level,
                                               absolute, level_as_reference):

        grouped_df = level_df.groupby(remaining_groups)

        grouped_df_keys = tuple(grouped_df.groups.keys())

        self._validate_levels(level_df, remaining_groups, level)

        posteriors = [
            self._sample_posterior(grouped_df.get_group(level))
            for level in grouped_df_keys
        ]

        var_indx = grouped_df_keys.index(level)

        other_indx = [
            i for i, value in enumerate(grouped_df_keys) if value != level
        ]

        posterior_matrix = np.vstack(posteriors)

        start_value = posterior_matrix[var_indx]
        end_value = posterior_matrix
        if not level_as_reference:
            start_value, end_value = end_value, start_value

        if absolute:
            difference_posterior = end_value - start_value
        else:
            difference_posterior = end_value / start_value - 1

        df = pd.DataFrame()
        for group in other_indx:
            df = pd.concat([df,
                            pd.DataFrame({'values': difference_posterior[group],
                                          'group': str(grouped_df_keys[group])
                                          })],
                           axis=0)

        # Take the difference posterior and create a chart
        # df = pd.DataFrame({'values': difference_posterior})

        ch = chartify.Chart(y_axis_type='density', x_axis_type='linear')

        ch.plot.kde(df, 'values', color_column='group')

        title_change_label = 'from' if level_as_reference else 'to'
        ch.set_title("Change {} {}".format(title_change_label, level))

        subtitle = "" if not groupby else "{}: {}".format(groupby, level_name)
        ch.set_subtitle(subtitle)

        # Line at no difference
        ch.callout.line(0,
                        orientation='height',
                        line_color='black',
                        line_dash='dashed')
        # ch.callout.text('No change', 0, .5, angle=90)
        ch.style.color_palette.reset_palette_order()

        for group in other_indx:
            posterior_mean = difference_posterior[group].mean()
            # Plot callout for the mean
            ch.callout.line(
                posterior_mean,
                orientation='height',
                line_color=ch.style.color_palette.next_color(),
                line_dash='dashed')

            ch.callout.text(
                'Expected change: {0:.2f}%'.format(posterior_mean * 100),
                posterior_mean,
                0,
                angle=90)

        # ch.callout.line(
        #     potential_loss,
        #     orientation='height',
        #     line_color=ch.style.color_palette._colors[1])
        # ch.callout.text(
        #     'Potential Loss: {0:.2f}%'.format(potential_loss * 100),
        #     potential_loss,
        #     1.5,
        #     angle=90)
        # ch.callout.text(
        #     '{0:.2f}%'.format(potential_loss * 100), potential_loss, 1.)

        ch.set_source_label("")
        ch.axes.set_yaxis_range(0)
        ch.axes.set_xaxis_label(self.get_difference_plot_label(absolute))
        ch.axes.set_yaxis_label("Probability Density")
        ch.axes.hide_yaxis()
        axis_format = axis_format_precision(df['values'].max() * 10,
                                            df['values'].min() * 10,
                                            absolute)
        ch.axes.set_xaxis_tick_format(axis_format)

        return ch

    @randomization_warning_decorator
    def _categorical_multiple_difference_plot(self, level, absolute, groupby,
                                              level_as_reference):
        """Pairwise comparison of the given level to all others.

        Args:
            level (str, tuple of str): Name of level.
            absolute (bool): If True then return the absolute difference
                otherwise return the relative difference.
            groupby (str): Name of column.
                If specified, will return an interval for each level
                of the grouped dimension.
            level_as_reference (bool): If True, the given level is the reference
                value for the change. (level1)

        Returns:
            GroupedChart object.
        """

        results_df = self._iterate_groupby_to_chartgrid(
            self._categorical_multiple_difference_chart,
            groupby=groupby,
            level=level,
            absolute=absolute,
            level_as_reference=level_as_reference
        )

        return results_df

# class GammaPoisson(PoissonResponse):
#     pass


# class DirichetMultinomial(MultinomialResponse):
#     def __init__(self,
#                  data_frame,
#                  group_columns,
#                  category_column,
#                  value_column,
#                  prior_value_column=None):

#         super().__init__(data_frame, group_columns, category_column,
#                          value_column)


# class Gaussian(GaussianResponse):
#     def __init__(self,
#                  data_frame,
#                  groupings,
#                  mean_col,
#                  std_col,
#                  n_col,
#                  time_grouping=None,
#                  prior_columns=None):
#         self.prior_lambda_column = prior_lambda_column
#         super(BaseGaussianResponse, self).__init__(
#             data_frame, groups, mean_col, std_col, n_col, time_grouping)
#         raise (NotImplementedError)


# class DirichetCategorical(CategoricalResponse):
#     pass
