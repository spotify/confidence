import pytest

import spotify_confidence
import pandas as pd
import numpy as np

from spotify_confidence.analysis.constants import (
    INCREASE_PREFFERED,
    DECREASE_PREFFERED, POINT_ESTIMATE,
    CI_LOWER, CI_UPPER, P_VALUE,
    ADJUSTED_LOWER, ADJUSTED_UPPER,
    DIFFERENCE, BONFERRONI_ONLY_COUNT_TWOSIDED)


class TestBinary(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame({'variation_name':
                                 ['test', 'control', 'test2', 'test3'],
                                  'success': [50, 40, 10, 20],
                                  'total': [100, 100, 50, 60],
                                  })

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='success',
            numerator_sum_squares_column=None,
            denominator_column='total',
            categorical_group_columns='variation_name',
            correction_method='bonferroni')

    def test_init_sumsq_sum(self):
        spotify_confidence.ZTest(
            self.data,
            numerator_column='success',
            numerator_sum_squares_column='success',
            denominator_column='total',
            categorical_group_columns=['variation_name'])

    def test_summary(self):
        summary_df = self.test.summary()
        assert len(summary_df) == len(self.data)

    def test_difference(self):
        difference_df = self.test.difference(level_1='control',
                                             level_2='test',
                                             absolute=True)
        assert len(difference_df) == 1
        assert difference_df['difference'][0] == 0.5 - 0.4

    def test_difference_absolute_false(self):
        difference_df = self.test.difference(level_1='control',
                                             level_2='test',
                                             absolute=False)
        assert len(difference_df) == 1
        assert difference_df['difference'][0] == (0.5 - 0.4) / 0.4

    def test_multiple_difference(self):
        difference_df = self.test.multiple_difference(level='control',
                                                      level_as_reference=True)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert difference_df['difference'][0] == 0.5 - 0.4

    def test_multiple_difference_level_as_reference_false(self):
        difference_df_true_true = self.test.multiple_difference(
            level='control',
            level_as_reference=True,
            absolute=True)

        difference_df = self.test.multiple_difference(level='control',
                                                      level_as_reference=False,
                                                      absolute=True)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(difference_df['difference'],
                           -difference_df_true_true['difference'],
                           atol=0)
        assert np.allclose(difference_df['ci_lower'],
                           -difference_df_true_true['ci_upper'],
                           atol=0)
        assert np.allclose(difference_df['ci_upper'],
                           -difference_df_true_true['ci_lower'],
                           atol=0)
        assert np.allclose(difference_df['p-value'],
                           difference_df_true_true['p-value'],
                           atol=0)

    def test_multiple_difference_absolute_false(self):
        control_mean = (
            self.test
                .summary()
                .query("variation_name == 'control'")['point_estimate']
                .values[0]
        )
        difference_df_true_true = self.test.multiple_difference(
            level='control',
            level_as_reference=True,
            absolute=True)

        difference_df = self.test.multiple_difference(level='control',
                                                      level_as_reference=True,
                                                      absolute=False)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(difference_df['difference'],
                           difference_df_true_true['difference']/control_mean,
                           atol=0)
        assert np.allclose(difference_df['ci_lower'],
                           difference_df_true_true['ci_lower']/control_mean,
                           atol=0)
        assert np.allclose(difference_df['ci_upper'],
                           difference_df_true_true['ci_upper']/control_mean,
                           atol=0)
        assert np.allclose(difference_df['p-value'],
                           difference_df_true_true['p-value'],
                           atol=0)

    def test_multiple_difference_level_as_reference_false_absolute_false(self):
        reference_mean = (
            self.test
                .summary()
                .query("variation_name != 'control'")['point_estimate']
        )
        difference_df_true_true = self.test.multiple_difference(
            level='control',
            level_as_reference=True,
            absolute=True)

        difference_df = self.test.multiple_difference(level='control',
                                                      level_as_reference=False,
                                                      absolute=False)
        assert len(difference_df) == self.data.variation_name.unique().size - 1
        assert np.allclose(
            difference_df['difference'],
            -difference_df_true_true['difference'] / reference_mean.values,
            atol=0)
        assert np.allclose(
            difference_df['ci_lower'],
            -difference_df_true_true['ci_upper'] / reference_mean.values,
            atol=0)
        assert np.allclose(
            difference_df['ci_upper'],
            -difference_df_true_true['ci_lower'] / reference_mean.values,
            atol=0)
        assert np.allclose(difference_df['p-value'],
                           difference_df_true_true['p-value'],
                           atol=0)

    def test_summary_plot(self):
        chart_grid = self.test.summary_plot()
        assert len(chart_grid.charts) == 1

    def test_difference_plot(self):
        chartgrid = self.test.difference_plot(level_1='control', level_2='test')
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot(self):
        chartgrid = self.test.multiple_difference_plot(
            level='control',
            level_as_reference=True)
        assert len(chartgrid.charts) == 1


class TestCategoricalBinary(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame({'variation_name':
                                 ['test', 'test', 'control',
                                  'control', 'test2', 'test2',
                                  'test3', 'test3'],
                                  'success': [50, 60, 40, 140, 10, 20, 20, 20],
                                  'total': [100, 100, 100, 200, 50, 50, 60, 60],
                                  'country':
                                      ['us', 'ca', 'us', 'ca',
                                       'us', 'ca', 'us', 'ca']})

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='success',
            numerator_sum_squares_column=None,
            denominator_column='total',
            categorical_group_columns=['country', 'variation_name'])

    def test_init_sumsq_sum(self):
        spotify_confidence.ZTest(
            self.data,
            numerator_column='success',
            numerator_sum_squares_column='success',
            denominator_column='total',
            categorical_group_columns=['variation_name', 'country'])

    def test_init_sumsq_sum_one_country(self):
        spotify_confidence.ZTest(
            self.data.query('country == "us"'),
            numerator_column='success',
            numerator_sum_squares_column='success',
            denominator_column='total',
            categorical_group_columns='variation_name')

    def test_summary(self):
        summary_df = self.test.summary()
        assert len(summary_df) == len(self.data)

    def test_difference(self):
        difference_df = self.test.difference(
            level_1=('us', 'control'),
            level_2=('ca', 'test'))
        assert len(difference_df) == 1

    def test_difference_groupby(self):
        difference_df = self.test.difference(level_1='control',
                                             level_2='test',
                                             groupby='country')
        assert len(difference_df) == self.data.country.unique().size

    def test_multiple_difference(self):
        difference_df = self.test.multiple_difference(level=('us', 'control'),
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.country.unique().size +
            self.data.country.unique().size-1
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_level_as_reference_false(self):
        difference_df = self.test.multiple_difference(level=('us', 'control'),
                                                      level_as_reference=False)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.country.unique().size +
            self.data.country.unique().size-1
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby(self):
        difference_df = self.test.multiple_difference(level='control',
                                                      groupby='country',
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.country.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_summary_plot(self):
        chart_grid = self.test.summary_plot()
        assert len(chart_grid.charts) == 1

    def test_summary_plot_groupby(self):
        chart_grid = self.test.summary_plot(groupby='country')
        assert len(chart_grid.charts) == self.data.country.unique().size

    def test_difference_plot(self):
        chartgrid = self.test.difference_plot(
            level_1=('us', 'control'),
            level_2=('ca', 'test'))
        assert len(chartgrid.charts) == 1

    def test_difference_plot_groupby(self):
        chartgrid = self.test.difference_plot(level_1='control',
                                              level_2='test',
                                              groupby='country')
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot(self):
        chartgrid = self.test.multiple_difference_plot(
            level=('us', 'control'),
            level_as_reference=True)
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot_groupby(self):
        chartgrid = self.test.multiple_difference_plot(
            level='control',
            groupby='country',
            level_as_reference=True)
        assert len(chartgrid.charts) == 1


class TestCategoricalContinuous(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame({
            'variation_name':
            ['test', 'control', 'test2', 'test', 'control', 'test2'],
            'nr_of_items': [1969, 312, 2955, 195, 24, 330],
            'nr_of_items_sumsq': [5767, 984, 8771, 553, 80, 1010],
            'users': [1009, 104, 1502, 100, 10, 150],
            'country': ['us', 'us', 'us', 'gb', 'gb', 'gb']
        })

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='nr_of_items',
            numerator_sum_squares_column='nr_of_items_sumsq',
            denominator_column='users',
            categorical_group_columns=['country', 'variation_name'])

    def test_init_one_country(self):
        spotify_confidence.ZTest(
            self.data.query('country == "us"'),
            numerator_column='nr_of_items',
            numerator_sum_squares_column='nr_of_items_sumsq',
            denominator_column='users',
            categorical_group_columns='variation_name')

    def test_summary(self):
        summary_df = self.test.summary()
        assert len(summary_df) == len(self.data)

    def test_difference(self):
        difference_df = self.test.difference(level_1=('us', 'control'),
                                             level_2=('us', 'test'))
        assert len(difference_df) == 1

    def test_difference_groupby(self):
        difference_df = self.test.difference(level_1='control',
                                             level_2='test',
                                             groupby='country')
        assert len(difference_df) == self.data.country.unique().size

    def test_multiple_difference(self):
        difference_df = self.test.multiple_difference(level=('us', 'control'),
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.country.unique().size +
            self.data.country.unique().size-1
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby(self):
        difference_df = self.test.multiple_difference(level='control',
                                                      groupby='country',
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.country.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_summary_plot(self):
        chart_grid = self.test.summary_plot()
        assert len(chart_grid.charts) == 1

    def test_summary_plot_groupby(self):
        chart_grid = self.test.summary_plot(groupby='country')
        assert len(chart_grid.charts) == self.data.country.unique().size

    def test_difference_plot(self):
        chartgrid = self.test.difference_plot(
            level_1=('us', 'control'),
            level_2=('gb', 'test'))
        assert len(chartgrid.charts) == 1

    def test_difference_plot_groupby(self):
        chartgrid = self.test.difference_plot(level_1='control',
                                              level_2='test',
                                              groupby='country')
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot(self):
        chartgrid = self.test.multiple_difference_plot(
            level=('us', 'control'),
            level_as_reference=True)
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot_groupby(self):
        chartgrid = self.test.multiple_difference_plot(
            level='control',
            groupby='country',
            level_as_reference=True)
        assert len(chartgrid.charts) == 1


class TestOrdinal(object):
    def setup(self):

        self.data = pd.DataFrame({
            'variation_name': ['test', 'control', 'test2',
                               'test', 'control', 'test2',
                               'test', 'control', 'test2',
                               'test', 'control', 'test2',
                               'test', 'control', 'test2'],
            'nr_of_items': [500, 8, 100,
                            510, 8, 100,
                            520, 9, 104,
                            530, 7, 100,
                            530, 8, 103],
            'nr_of_items_sumsq': [2500, 12, 150,
                                  2510, 13, 140,
                                  2520, 14, 154,
                                  2530, 15, 160,
                                  2530, 16, 103],
            'users': [1010, 22, 150,
                      1000, 20, 153,
                      1030, 23, 154,
                      1000, 20, 150,
                      1040, 21, 155],
            'days_since_reg': [1, 1, 1,
                               2, 2, 2,
                               3, 3, 3,
                               4, 4, 4,
                               5, 5, 5],
        })

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='nr_of_items',
            numerator_sum_squares_column='nr_of_items_sumsq',
            denominator_column='users',
            categorical_group_columns='variation_name',
            ordinal_group_column='days_since_reg')

    def test_summary(self):
        summary_df = self.test.summary()
        assert len(summary_df) == len(self.data)

    def test_difference(self):
        difference_df = self.test.difference(level_1=('control', 1),
                                             level_2=('test', 1))
        assert len(difference_df) == 1

    def test_difference_groupby(self):
        difference_df = self.test.difference(level_1='control',
                                             level_2='test',
                                             groupby='days_since_reg')
        assert len(difference_df) == self.data.days_since_reg.unique().size

    def test_multiple_difference(self):
        difference_df = self.test.multiple_difference(level=('control', 1),
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.days_since_reg.unique().size +
            self.data.days_since_reg.unique().size-1
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby(self):
        difference_df = self.test.multiple_difference(level='control',
                                                      groupby='days_since_reg',
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.days_since_reg.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_summary_plot(self):
        chart_grid = self.test.summary_plot()
        assert len(chart_grid.charts) == 1

    def test_summary_plot_groupby(self):
        chart_grid = self.test.summary_plot(groupby='days_since_reg')
        assert len(chart_grid.charts) == self.data.days_since_reg.unique().size

    def test_difference_plot(self):
        chartgrid = self.test.difference_plot(
            level_1=('control', 1),
            level_2=('test', 2))
        assert len(chartgrid.charts) == 1

    def test_difference_plot_groupby(self):
        chartgrid = self.test.difference_plot(level_1='control',
                                              level_2='test',
                                              groupby='days_since_reg')
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot(self):
        chartgrid = self.test.multiple_difference_plot(
            level=('control', 1),
            level_as_reference=True)
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot_groupby(self):
        chartgrid = self.test.multiple_difference_plot(
            level='control',
            groupby='days_since_reg',
            level_as_reference=True)
        assert len(chartgrid.charts) == 1


class TestOrdinalPlusTwoCategorical(object):
    def setup(self):
        self.data = pd.DataFrame(
            {'variation_name': ['test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2', ],
             'nr_of_items': [500, 8, 100,
                             510, 8, 100,
                             520, 9, 104,
                             530, 7, 100,
                             530, 8, 103,
                             500, 8, 100,
                             510, 8, 100,
                             520, 9, 104,
                             530, 7, 100,
                             530, 8, 103, ],
             'nr_of_items_sumsq': [1010, 32, 250,
                                   1000, 30, 253,
                                   1030, 33, 254,
                                   1000, 30, 250,
                                   1040, 31, 255,
                                   1010, 22, 150,
                                   1000, 20, 153,
                                   1030, 23, 154,
                                   1000, 20, 150,
                                   1040, 21, 155, ],
             'users': [2010, 42, 250,
                       2000, 40, 253,
                       2030, 43, 254,
                       2000, 40, 250,
                       2040, 41, 255,
                       1010, 22, 150,
                       1000, 20, 153,
                       1030, 23, 154,
                       1000, 20, 150,
                       1040, 21, 155, ],
             'days_since_reg': [1, 1, 1,
                                2, 2, 2,
                                3, 3, 3,
                                4, 4, 4,
                                5, 5, 5,
                                1, 1, 1,
                                2, 2, 2,
                                3, 3, 3,
                                4, 4, 4,
                                5, 5, 5],
             'country': ['us', 'us', 'us', 'us', 'us', 'us', 'us',
                         'us', 'us', 'us', 'us', 'us', 'us', 'us',
                         'us',
                         'gb', 'gb', 'gb', 'gb', 'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb', 'gb', 'gb', 'gb', 'gb',
                         'gb', ]})

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='nr_of_items',
            numerator_sum_squares_column='nr_of_items_sumsq',
            denominator_column='users',
            categorical_group_columns=['variation_name', 'country'],
            ordinal_group_column='days_since_reg')

    def test_summary(self):
        summary_df = self.test.summary()
        assert len(summary_df) == len(self.data)

    def test_difference(self):
        difference_df = self.test.difference(level_1=('control', 'gb', 1),
                                             level_2=('test', 'us', 2))
        assert len(difference_df) == 1

    def test_difference_groupby(self):
        difference_df = self.test.difference(level_1='control',
                                             level_2='test',
                                             groupby=['country',
                                                      'days_since_reg'])
        assert len(difference_df) == self.data.days_since_reg.unique().size\
            * self.data.country.unique().size

    def test_multiple_difference(self):
        difference_df = self.test.multiple_difference(level=('control', 1),
                                                      groupby='country',
                                                      level_as_reference=True)
        assert len(difference_df) == (
            self.data.country.unique().size *
            ((self.data.variation_name.unique().size - 1)
             * self.data.days_since_reg.unique().size +
             self.data.days_since_reg.unique().size-1)
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby(self):
        difference_df = self.test.multiple_difference(level='control',
                                                      groupby=['days_since_reg',
                                                               'country'],
                                                      level_as_reference=True)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.days_since_reg.unique().size
            * self.data.country.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_differece_with_nims(self):
        df = self.test.difference(level_1=('test', 'us'),
                                  level_2=('control', 'us'),
                                  groupby='days_since_reg',
                                  non_inferiority_margins=(
                                      0.01, INCREASE_PREFFERED))
        assert (len(df) == 5)
        assert ('days_since_reg' in df.columns)

        df = self.test.difference(level_1=('test', 'us'),
                                  level_2=('control', 'us'),
                                  groupby=['days_since_reg'],
                                  non_inferiority_margins=(
                                      0.01, DECREASE_PREFFERED))
        assert (len(df) == 5)
        assert ('days_since_reg' in df.columns)

        df = self.test.difference(level_1=('test', 1),
                                  level_2=('control', 1),
                                  groupby=['country'],
                                  non_inferiority_margins={
                                      'us': (0.01, INCREASE_PREFFERED),
                                      'gb': (0.05, INCREASE_PREFFERED)})
        assert (len(df) == 2)
        assert ('country' in df.columns)

        df = self.test.difference(level_1='test',
                                  level_2='control',
                                  groupby=['country', 'days_since_reg'],
                                  non_inferiority_margins=(
                                      0.01, DECREASE_PREFFERED))
        assert (len(df) == 10)
        assert ('country' in df.columns)
        assert ('days_since_reg' in df.columns)

        nims = {('us', 1): (0.01, DECREASE_PREFFERED),
                ('us', 2): (0.1, INCREASE_PREFFERED),
                ('us', 3): (0.2, DECREASE_PREFFERED),
                ('us', 4): (0.5, INCREASE_PREFFERED),
                ('us', 5): (0.99, DECREASE_PREFFERED),
                ('gb', 1): (1.01, INCREASE_PREFFERED),
                ('gb', 2): (2.01, DECREASE_PREFFERED),
                ('gb', 3): (3.01, INCREASE_PREFFERED),
                ('gb', 4): (4.01, DECREASE_PREFFERED),
                ('gb', 5): (5.01, INCREASE_PREFFERED)}
        df = self.test.difference(level_1='test',
                                  level_2='control',
                                  groupby=['country', 'days_since_reg'],
                                  non_inferiority_margins=nims)
        assert (len(df) == 10)
        assert ('country' in df.columns)
        assert ('days_since_reg' in df.columns)

        df = self.test.multiple_difference(
            level='control',
            level_as_reference=True,
            groupby=['country', 'days_since_reg'],
            non_inferiority_margins=nims)
        assert (len(df) == 20)

    def test_summary_plot(self):
        chart_grid = self.test.summary_plot()
        assert len(chart_grid.charts) == 1

    def test_summary_plot_groupby(self):
        chart_grid = self.test.summary_plot(groupby='country')
        assert len(chart_grid.charts) == self.data.country.unique().size

    def test_summary_plot_groupby_2(self):
        chart_grid = self.test.summary_plot(groupby=['days_since_reg',
                                                     'country'])
        assert len(chart_grid.charts) == (
                self.data.country.unique().size
                * self.data.days_since_reg.unique().size)

    def test_difference_plot(self):
        chartgrid = self.test.difference_plot(
            level_1=('control', 'gb', 1),
            level_2=('test', 'us', 2))
        assert len(chartgrid.charts) == 1

    def test_difference_plot_groupby(self):
        chartgrid = self.test.difference_plot(level_1=('control', 'gb'),
                                              level_2=('test', 'us'),
                                              groupby='days_since_reg')
        assert len(chartgrid.charts) == 1

    def test_difference_plot_groupby_2(self):
        chartgrid = self.test.difference_plot(level_1='control',
                                              level_2='test',
                                              groupby=['days_since_reg',
                                                       'country'])
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot(self):
        chartgrid = self.test.multiple_difference_plot(
            level=('control', 1),
            groupby='country',
            level_as_reference=True)
        assert len(chartgrid.charts) == 1

    def test_multiple_difference_plot_groupby(self):
        chartgrid = self.test.multiple_difference_plot(
            level='control',
            groupby=['days_since_reg',
                     'country'],
            level_as_reference=True)
        assert len(chartgrid.charts) == 1

    def test_differece_plot_with_nims(self):
        ch = self.test.difference_plot(level_1=('test', 'us'),
                                       level_2=('control', 'us'),
                                       groupby='days_since_reg',
                                       non_inferiority_margins=(
                                           0.01, INCREASE_PREFFERED))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1=('test', 'us'),
                                       level_2=('control', 'us'),
                                       groupby=['days_since_reg'],
                                       non_inferiority_margins=(
                                           0.01, DECREASE_PREFFERED))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1=('test', 1),
                                       level_2=('control', 1),
                                       groupby=['country'],
                                       non_inferiority_margins={
                                             'us': (0.01, INCREASE_PREFFERED),
                                             'gb': (0.05, INCREASE_PREFFERED)})
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1='test',
                                       level_2='control',
                                       groupby=['country', 'days_since_reg'],
                                       non_inferiority_margins=(
                                           0.01, DECREASE_PREFFERED))
        assert (len(ch.charts) == 1)

        nims = {('us', 1): (0.01, DECREASE_PREFFERED),
                ('us', 2): (0.1, INCREASE_PREFFERED),
                ('us', 3): (0.2, DECREASE_PREFFERED),
                ('us', 4): (0.5, INCREASE_PREFFERED),
                ('us', 5): (0.99, DECREASE_PREFFERED),
                ('gb', 1): (1.01, INCREASE_PREFFERED),
                ('gb', 2): (2.01, DECREASE_PREFFERED),
                ('gb', 3): (3.01, INCREASE_PREFFERED),
                ('gb', 4): (4.01, DECREASE_PREFFERED),
                ('gb', 5): (5.01, INCREASE_PREFFERED)}
        ch = self.test.difference_plot(level_1='test',
                                       level_2='control',
                                       groupby=['country', 'days_since_reg'],
                                       non_inferiority_margins=nims)
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_plot(
            level='control',
            level_as_reference=True,
            groupby=['country', 'days_since_reg'],
            non_inferiority_margins=nims)
        assert (len(ch.charts) == 1)


class TestCategoricalBinomialData(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame({
            'variation_name':
            ['test', 'control', 'test2', 'test', 'control', 'test2'],
            'success': [500, 42, 1005, 50, 4, 100],
            'total': [1009, 104, 1502, 100, 10, 150],
            'country': [
                'us',
                'us',
                'us',
                'gb',
                'gb',
                'gb',
            ]
        })

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='success',
            numerator_sum_squares_column='success',
            denominator_column='total',
            categorical_group_columns=['country', 'variation_name'])

    def test_summary(self):
        """Area plot tests"""

        summary = self.test.summary()
        assert (np.array_equal(summary.country,
                               np.array(['us', 'us', 'us', 'gb', 'gb', 'gb'])))
        assert (np.array_equal(summary.point_estimate,
                               self.data.success / self.data.total))
        assert (np.allclose(
            summary['ci_lower'],
            np.array([
                0.4646901340180582, 0.30954466010970333, 0.6453118311511006,
                0.4020018007729973, 0.0963636851484016, 0.5912276177282552
            ])))
        assert (np.allclose(
            summary['ci_upper'],
            np.array([
                0.5263901434844195, 0.4981476475826044, 0.692903881232388,
                0.5979981992270027, 0.7036363148515985, 0.7421057156050781
            ])))

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference(('bad_value', 'bad_value'))

        diff = self.test.multiple_difference(('us', 'control'))
        assert (np.allclose(
            diff['adjusted p-value'],
            np.array([1e+00, 8.291843e-01, 9.971992e-05,
                      3.504662e-01, 4.504966e-07])))
        assert (np.allclose(
            diff['p-value'],
            np.array([9.81084197e-01, 1.65836862e-01, 1.99439850e-05,
                      7.00932382e-02, 9.00993166e-08])))
        assert (np.allclose(
            diff['adjusted ci_lower'],
            np.array([
                -0.41400184, -0.27489017, -0.42153065, -0.22209041, -0.39307973
            ])))
        assert (np.allclose(
            diff['adjusted ci_upper'],
            np.array(
                [0.42169415, 0.08258247, -0.10411038, 0.03870244,
                 -0.13744367])))

        diff = self.test.multiple_difference('test', groupby='country')
        assert (np.allclose(
            diff['adjusted p-value'],
            np.array([1.00000000e+00, 3.30302805e-02, 2.80372953e-01, 0.0])))
        assert (np.allclose(
            diff['p-value'],
            np.array([5.39020329e-01, 8.25757011e-03, 7.00932382e-02, 0.0])))
        assert (np.allclose(
            diff['adjusted ci_lower'],
            np.array([-0.30659699, -0.32426934, -0.03474758, -0.2232184])))
        assert (np.allclose(
            diff['adjusted ci_upper'],
            np.array([0.50659699, -0.00906399, 0.21813554, -0.12391703])))


class TestWithNims(object):
    def setup(self):
        self.data = pd.DataFrame(
            [
                {'group': "1",
                 'count': 5000,
                 'sum': 10021.0,
                 'sum_of_squares': 25142.0,
                 'avg': 2.004210,
                 'var': 1.0116668},
                {'group': "2",
                 'count': 5000,
                 'sum': 9892.0,
                 'sum_of_squares': 24510.0,
                 'avg': 1.978424,
                 'var': 0.9881132},
            ]
        )

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='sum',
            numerator_sum_squares_column='sum_of_squares',
            denominator_column='count',
            categorical_group_columns='group',
            interval_size=0.99)

    def test_compare_series_non_inferiority_improve_postitive(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg,
                           summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(
            level_1='1',
            level_2='2',
            non_inferiority_margins=(0.02, 'increase'))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_UPPER].values[0] == float('inf')
        np.testing.assert_almost_equal(diff[CI_LOWER].values[0], -0.0723, 3)
        assert diff[P_VALUE].values[0] > 0.01

    def test_compare_series_non_inferiority_improve_negative(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg,
                           summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(
            level_1='1',
            level_2='2',
            non_inferiority_margins=(0.02, 'decrease'))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_LOWER].values[0] == -float('inf')
        np.testing.assert_almost_equal(diff[CI_UPPER].values[0], 0.0207, 3)
        assert diff[P_VALUE].values[0] < 0.01

    def test_one_sided_ztest_positive(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg,
                           summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(
            level_1='1',
            level_2='2',
            non_inferiority_margins=(None, 'increase'))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_UPPER].values[0] == float('inf')
        np.testing.assert_almost_equal(diff[CI_LOWER].values[0], -0.0723, 3)
        assert diff[P_VALUE].values[0] > 0.01

    def test_one_sided_ztest_negative(self):
        summary = self.test.summary()
        control_avg = self.data.query("group == '1'").avg.values[0]
        assert np.allclose(control_avg,
                           summary.query("group == '1'")[POINT_ESTIMATE])

        diff = self.test.difference(
            level_1='1',
            level_2='2',
            non_inferiority_margins=(None, 'decrease'))

        np.testing.assert_almost_equal(diff[DIFFERENCE].values[0], -0.0258, 3)
        assert diff[CI_LOWER].values[0] == -float('inf')
        np.testing.assert_almost_equal(diff[CI_UPPER].values[0], 0.0207, 3)
        assert diff[P_VALUE].values[0] > 0.01


class TestSequentialOrdinalPlusTwoCategorical(object):
    def setup(self):
        np.random.seed(123)
        d = 50 + 1 * np.random.randn(60)
        u = np.floor(2000 + np.linspace(0, 1000, 60) + 10 * np.random.randn(60))
        self.data = pd.DataFrame(
            {'variation_name': ['test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2',
                                'test', 'control', 'test2'],
             'nr_of_items': d,
             'nr_of_items_sumsq': d/20,
             'users': u,
             'date': pd.to_datetime(['2021-04-01', '2021-04-01', '2021-04-01',
                                     '2021-04-02', '2021-04-02', '2021-04-02',
                                     '2021-04-03', '2021-04-03', '2021-04-03',
                                     '2021-04-04', '2021-04-04', '2021-04-04',
                                     '2021-04-05', '2021-04-05', '2021-04-05',
                                     '2021-04-01', '2021-04-01', '2021-04-01',
                                     '2021-04-02', '2021-04-02', '2021-04-02',
                                     '2021-04-03', '2021-04-03', '2021-04-03',
                                     '2021-04-04', '2021-04-04', '2021-04-04',
                                     '2021-04-05', '2021-04-05', '2021-04-05',
                                     '2021-04-01', '2021-04-01', '2021-04-01',
                                     '2021-04-02', '2021-04-02', '2021-04-02',
                                     '2021-04-03', '2021-04-03', '2021-04-03',
                                     '2021-04-04', '2021-04-04', '2021-04-04',
                                     '2021-04-05', '2021-04-05', '2021-04-05',
                                     '2021-04-01', '2021-04-01', '2021-04-01',
                                     '2021-04-02', '2021-04-02', '2021-04-02',
                                     '2021-04-03', '2021-04-03', '2021-04-03',
                                     '2021-04-04', '2021-04-04', '2021-04-04',
                                     '2021-04-05', '2021-04-05', '2021-04-05']),
             'country': ['us', 'us', 'us',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'us', 'us', 'us',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb',
                         'gb', 'gb', 'gb'],
             'metric': ['m1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm1', 'm1', 'm1',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2',
                        'm2', 'm2', 'm2']
             }
        )

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column='nr_of_items',
            numerator_sum_squares_column='nr_of_items_sumsq',
            denominator_column='users',
            categorical_group_columns=['variation_name', 'country', 'metric'],
            ordinal_group_column='date')

    def test_multiple_difference_groupby(self):
        final_sample_size = self.data.query("date == '2021-04-05'")['users'].sum()
        difference_df = self.test.multiple_difference(
            level='control',
            groupby=['date', 'country', 'metric'],
            level_as_reference=True,
            final_expected_sample_size=final_sample_size)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.date.unique().size
            * self.data.country.unique().size
            * self.data.metric.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby_onesided_decrease(self):
        final_sample_size = self.data.query("date == '2021-04-05'")['users'].sum()
        difference_df = self.test.multiple_difference(
            level='control',
            groupby=['date', 'country', 'metric'],
            level_as_reference=True,
            non_inferiority_margins=(0.05, 'decrease'),
            final_expected_sample_size=final_sample_size)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.date.unique().size
            * self.data.country.unique().size
            * self.data.metric.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby_onesided_increase(self):
        final_sample_size = self.data.query("date == '2021-04-05'")['users'].sum()
        difference_df = self.test.multiple_difference(
            level='control',
            groupby=['date', 'country', 'metric'],
            level_as_reference=True,
            non_inferiority_margins=(0.05, 'increase'),
            final_expected_sample_size=final_sample_size)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.date.unique().size
            * self.data.country.unique().size
            * self.data.metric.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)

    def test_multiple_difference_groupby_mixed_nims(self):
        nims = {(pd.to_datetime('2021-04-01'), 'us', 'm1'): (0.2, 'increase'),
                (pd.to_datetime('2021-04-02'), 'us', 'm1'): (0.2, 'increase'),
                (pd.to_datetime('2021-04-03'), 'us', 'm1'): (0.2, 'increase'),
                (pd.to_datetime('2021-04-04'), 'us', 'm1'): (0.2, 'increase'),
                (pd.to_datetime('2021-04-05'), 'us', 'm1'): (0.2, 'increase'),
                (pd.to_datetime('2021-04-01'), 'gb', 'm1'): (0.1, 'increase'),
                (pd.to_datetime('2021-04-02'), 'gb', 'm1'): (0.1, 'increase'),
                (pd.to_datetime('2021-04-03'), 'gb', 'm1'): (0.1, 'increase'),
                (pd.to_datetime('2021-04-04'), 'gb', 'm1'): (0.1, 'increase'),
                (pd.to_datetime('2021-04-05'), 'gb', 'm1'): (0.1, 'increase'),
                (pd.to_datetime('2021-04-01'), 'us', 'm2'): (0, None),
                (pd.to_datetime('2021-04-02'), 'us', 'm2'): (0, None),
                (pd.to_datetime('2021-04-03'), 'us', 'm2'): (0, None),
                (pd.to_datetime('2021-04-04'), 'us', 'm2'): (0, None),
                (pd.to_datetime('2021-04-05'), 'us', 'm2'): (0, None),
                (pd.to_datetime('2021-04-01'), 'gb', 'm2'): (0, None),
                (pd.to_datetime('2021-04-02'), 'gb', 'm2'): (0, None),
                (pd.to_datetime('2021-04-03'), 'gb', 'm2'): (0, None),
                (pd.to_datetime('2021-04-04'), 'gb', 'm2'): (0, None),
                (pd.to_datetime('2021-04-05'), 'gb', 'm2'): (0, None)}
        final_sample_size = self.data.query("date == '2021-04-05'")['users'].sum()
        difference_df = self.test.multiple_difference(
            level='control',
            groupby=['date', 'country', 'metric'],
            level_as_reference=True,
            non_inferiority_margins=nims,
            final_expected_sample_size=final_sample_size)
        assert len(difference_df) == (
            (self.data.variation_name.unique().size - 1)
            * self.data.date.unique().size
            * self.data.country.unique().size
            * self.data.metric.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)


DATE = 'date'
COUNT = 'count'
SUM = 'sum'
SUM_OF_SQUARES = 'sum_of_squares'
GROUP = 'group'


class TestSequentialOrdinalPlusTwoCategorical2(object):
    def setup(self):
        self.data = pd.DataFrame(
            [
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 2016.416,
                    SUM_OF_SQUARES: 5082.122,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 2028.478,
                    SUM_OF_SQUARES: 5210.193,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1991.554,
                    SUM_OF_SQUARES: 4919.282,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1958.713,
                    SUM_OF_SQUARES: 4818.665,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 2030.252,
                    SUM_OF_SQUARES: 5129.574,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1966.138,
                    SUM_OF_SQUARES: 4848.321,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1995.389,
                    SUM_OF_SQUARES: 4992.710,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1000,
                    SUM: 1952.098,
                    SUM_OF_SQUARES: 4798.772,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2986.667,
                    SUM_OF_SQUARES: 7427.582,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2989.488,
                    SUM_OF_SQUARES: 7421.710,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 3008.681,
                    SUM_OF_SQUARES: 7565.406,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2933.173,
                    SUM_OF_SQUARES: 7207.038,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2986.308,
                    SUM_OF_SQUARES: 7584.148,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 2985.802,
                    SUM_OF_SQUARES: 7446.539,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 3008.190,
                    SUM_OF_SQUARES: 7532.521,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_1d",
                    COUNT: 1500,
                    SUM: 3001.494,
                    SUM_OF_SQUARES: 7467.535,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 2016.416,
                    SUM_OF_SQUARES: 5082.122,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 2028.478,
                    SUM_OF_SQUARES: 5210.193,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1991.554,
                    SUM_OF_SQUARES: 4919.282,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1958.713,
                    SUM_OF_SQUARES: 4818.665,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 2030.252,
                    SUM_OF_SQUARES: 5129.574,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1966.138,
                    SUM_OF_SQUARES: 4848.321,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1995.389,
                    SUM_OF_SQUARES: 4992.710,
                },
                {
                    DATE: "2020-04-01",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1000,
                    SUM: 1952.098,
                    SUM_OF_SQUARES: 4798.772,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2986.667,
                    SUM_OF_SQUARES: 7427.582,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2989.488,
                    SUM_OF_SQUARES: 7421.710,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 3008.681,
                    SUM_OF_SQUARES: 7565.406,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "ios",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2933.173,
                    SUM_OF_SQUARES: 7207.038,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2986.308,
                    SUM_OF_SQUARES: 7584.148,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "swe",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 2985.802,
                    SUM_OF_SQUARES: 7446.539,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "1",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 3008.190,
                    SUM_OF_SQUARES: 7532.521,
                },
                {
                    DATE: "2020-04-02",
                    GROUP: "2",
                    "country": "fin",
                    "platform": "andr",
                    "metric": "bananas_per_user_7d",
                    COUNT: 1500,
                    SUM: 3001.494,
                    SUM_OF_SQUARES: 7467.535,
                }
            ]
        )
        self.data[DATE] = pd.to_datetime(self.data[DATE])
        self.data = (
            self.data
                .groupby([DATE, GROUP, 'country', 'platform', 'metric']).sum()
                .groupby([GROUP, 'country', 'platform', 'metric']).cumsum()
                .reset_index()
        )

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column=SUM,
            numerator_sum_squares_column=SUM_OF_SQUARES,
            denominator_column=COUNT,
            categorical_group_columns=[GROUP, 'country', 'platform', 'metric'],
            ordinal_group_column=DATE,
            interval_size=1-0.01,
            correction_method=BONFERRONI_ONLY_COUNT_TWOSIDED
        )

    def test_with_maual_correction(self):
        test = spotify_confidence.ZTest(
            self.data,
            numerator_column=SUM,
            numerator_sum_squares_column=SUM_OF_SQUARES,
            denominator_column=COUNT,
            categorical_group_columns=[GROUP, 'country', 'platform', 'metric'],
            ordinal_group_column=DATE,
            interval_size=1 - 0.01/4
        )

        difference_df = test.difference(
            level_1=('1', 'fin', 'andr', 'bananas_per_user_1d'),
            level_2=('2', 'fin', 'andr', 'bananas_per_user_1d'),
            groupby='date',
            final_expected_sample_size=5000
        )
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[0], -0.2016570, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_UPPER].values[0], 0.11507406, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[1], -0.1063633, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_UPPER].values[1], 0.06637345, 3)

        difference_df = test.difference(
            level_1=('1', 'swe', 'ios', 'bananas_per_user_1d'),
            level_2=('2', 'swe', 'ios', 'bananas_per_user_1d'),
            groupby='date',
            final_expected_sample_size=5000
        )
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[0], -0.1506963, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_UPPER].values[0], 0.17481994, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[1], -0.0812409, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_UPPER].values[1], 0.09314668, 3)

        difference_df = test.difference(
            level_1=('1', 'fin', 'andr', 'bananas_per_user_7d'),
            level_2=('2', 'fin', 'andr', 'bananas_per_user_7d'),
            groupby='date',
            non_inferiority_margins=(0.01, 'increase'),
            final_expected_sample_size=5000
        )
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[0], -0.1932786, 3)
        np.isinf(difference_df[ADJUSTED_UPPER].values[0])
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[1], -0.10027731, 3)
        np.isinf(difference_df[ADJUSTED_UPPER].values[1])

        difference_df = test.difference(
            level_1=('1', 'swe', 'ios', 'bananas_per_user_7d'),
            level_2=('2', 'swe', 'ios', 'bananas_per_user_7d'),
            groupby='date',
            non_inferiority_margins=(0.01, 'increase'),
            final_expected_sample_size=5000
        )
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[0], -0.1420855, 3)
        np.isinf(difference_df[ADJUSTED_UPPER].values[0])
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[1], -0.07509674, 3)
        np.isinf(difference_df[ADJUSTED_UPPER].values[1])

    def test_multiple_difference_groupby(self):
        final_sample_size = self.data.query("date == '2020-04-02'")['count'].sum()
        summary_df = self.test.summary()

        np.testing.assert_almost_equal(
            summary_df.query('date == "2020-04-01" and group == "1" and country == "fin" '
                             'and platform == "andr" and metric=="bananas_per_user_1d"')[
                POINT_ESTIMATE].values[0], 1.995389, 5)
        np.testing.assert_almost_equal(
            summary_df.query('date == "2020-04-01" and group == "2" and country == "fin" '
                             'and platform == "andr" and metric=="bananas_per_user_1d"')[
                POINT_ESTIMATE].values[0], 1.952098, 5)
        np.testing.assert_almost_equal(
            summary_df.query('date == "2020-04-01" and group == "1" and country == "swe" '
                             'and platform == "ios" and metric=="bananas_per_user_1d"')[
                POINT_ESTIMATE].values[0], 2.016416, 5)
        np.testing.assert_almost_equal(
            summary_df.query('date == "2020-04-01" and group == "2" and country == "swe" '
                             'and platform == "ios" and metric=="bananas_per_user_1d"')[
                POINT_ESTIMATE].values[0], 2.028478, 5)

        nims = {(pd.to_datetime('2020-04-01'), 'bananas_per_user_1d', 'fin', 'andr',): (0, None),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_7d', 'fin', 'andr',): (0.01, 'increase'),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_1d', 'fin', 'ios', ): (0, None),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_7d', 'fin', 'ios', ): (0.01, 'increase'),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_1d', 'swe', 'andr',): (0, None),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_7d', 'swe', 'andr',): (0.01, 'increase'),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_1d', 'swe', 'ios', ): (0, None),
                (pd.to_datetime('2020-04-01'), 'bananas_per_user_7d', 'swe', 'ios', ): (0.01, 'increase'),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_1d', 'fin', 'andr',): (0, None),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_7d', 'fin', 'andr',): (0.01, 'increase'),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_1d', 'fin', 'ios', ): (0, None),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_7d', 'fin', 'ios', ): (0.01, 'increase'),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_1d', 'swe', 'andr',): (0, None),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_7d', 'swe', 'andr',): (0.01, 'increase'),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_1d', 'swe', 'ios', ): (0, None),
                (pd.to_datetime('2020-04-02'), 'bananas_per_user_7d', 'swe', 'ios', ): (0.01, 'increase')}

        difference_df = self.test.multiple_difference(
            level='1',
            groupby=['date', 'metric', 'country', 'platform'],
            level_as_reference=True,
            final_expected_sample_size=final_sample_size,
            non_inferiority_margins=nims)
        assert len(difference_df) == (
            (self.data.group.unique().size - 1)
            * self.data.date.unique().size
            * self.data.country.unique().size
            * self.data.platform.unique().size
            * self.data.metric.unique().size
        )
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-01" and country == "fin" and platform == "andr" and metric=="bananas_per_user_1d"')[
                ADJUSTED_LOWER].values[0], -0.2016570, 3)
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-01" and country == "fin" and platform == "andr" and metric=="bananas_per_user_1d"')[
                ADJUSTED_UPPER].values[0], 0.11507406, 3)
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-01" and country == "swe" and platform == "ios" and metric=="bananas_per_user_1d"')[
                ADJUSTED_LOWER].values[0], -0.1506963, 3)
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-01" and country == "swe" and platform == "ios" and metric=="bananas_per_user_1d"')[
                ADJUSTED_UPPER].values[0], 0.17481994, 3)

        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-02" and country == "fin" and platform == "andr" and metric=="bananas_per_user_1d"')[
                ADJUSTED_LOWER].values[0], -0.1063633, 3)
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-02" and country == "fin" and platform == "andr" and metric=="bananas_per_user_1d"')[
                ADJUSTED_UPPER].values[0], 0.06637345, 3)
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-02" and country == "swe" and platform == "ios" and metric=="bananas_per_user_1d"')[
                ADJUSTED_LOWER].values[0], -0.0812409, 3)
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-02" and country == "swe" and platform == "ios" and metric=="bananas_per_user_1d"')[
                ADJUSTED_UPPER].values[0], 0.09314668, 3)

        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-01" and country == "fin" and platform == "andr" and metric=="bananas_per_user_7d"')[
                ADJUSTED_LOWER].values[0], -0.1932786, 3)
        np.isinf(
            difference_df.query(
                'date == "2020-04-01" and country == "fin" and platform == "andr" and metric=="bananas_per_user_7d"')[
                ADJUSTED_UPPER].values[0])
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-01" and country == "swe" and platform == "ios" and metric=="bananas_per_user_7d"')[
                ADJUSTED_LOWER].values[0], -0.1420855, 3)
        np.isinf(
            difference_df.query(
                'date == "2020-04-01" and country == "swe" and platform == "ios" and metric=="bananas_per_user_7d"')[
                ADJUSTED_UPPER].values[0])

        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-02" and country == "fin" and platform == "andr" and metric=="bananas_per_user_7d"')[
                ADJUSTED_LOWER].values[0], -0.10027731, 3)
        np.isinf(
            difference_df.query(
                'date == "2020-04-02" and country == "fin" and platform == "andr" and metric=="bananas_per_user_7d"')[
                ADJUSTED_UPPER].values[0])
        np.testing.assert_almost_equal(
            difference_df.query(
                'date == "2020-04-02" and country == "swe" and platform == "ios" and metric=="bananas_per_user_7d"')[
                ADJUSTED_LOWER].values[0], -0.07509674, 3)
        np.isinf(
            difference_df.query(
                'date == "2020-04-02" and country == "swe" and platform == "ios" and metric=="bananas_per_user_7d"')[
                ADJUSTED_UPPER].values[0])

        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01)


class TestSequentialOneSided(object):
    def setup(self):
        DATE = 'date'
        COUNT = 'count'
        SUM = 'sum'
        SUM_OF_SQUARES = 'sum_of_squares'
        GROUP = 'group'

        self.data = pd.DataFrame(
            [
                {
                    DATE: 1,
                    GROUP: "1",
                    COUNT: 1250,
                    SUM: 2510.0,
                    SUM_OF_SQUARES: 6304.0
                },
                {
                    DATE: 1,
                    GROUP: "2",
                    COUNT: 1250,
                    SUM: -2492.0,
                    SUM_OF_SQUARES: 6163.0
                },
            ]
        )

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column=SUM,
            numerator_sum_squares_column=SUM_OF_SQUARES,
            denominator_column=COUNT,
            categorical_group_columns=GROUP,
            ordinal_group_column=DATE,
            interval_size=0.99
        )

    def test_multiple_difference_groupby(self):
        difference_df = self.test.multiple_difference(
            level='1',
            groupby='date',
            level_as_reference=True,
            final_expected_sample_size=1e4,
            non_inferiority_margins=(0.01, 'increase'))
        assert len(difference_df) == (
                (self.data.group.unique().size - 1)
                * self.data.date.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01
        )
        assert np.isinf(difference_df[CI_UPPER].values[0])
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[0], -4.129515314002298, 3)
        np.testing.assert_almost_equal(difference_df[DIFFERENCE].values[0], -4.001416, 3)


class TestSequentialTwoSided(object):
    def setup(self):
        DATE = 'date'
        COUNT = 'count'
        SUM = 'sum'
        SUM_OF_SQUARES = 'sum_of_squares'
        GROUP = 'group'

        self.data = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime("2020-04-01"),
                    GROUP: "1",
                    COUNT: 1250,
                    SUM: 2510.0,
                    SUM_OF_SQUARES: 6304.0
                },
                {
                    "date": pd.to_datetime("2020-04-01"),
                    GROUP: "2",
                    COUNT: 1250,
                    SUM: 2492.0,
                    SUM_OF_SQUARES: 6163.0
                },
            ]
        )

        self.test = spotify_confidence.ZTest(
            self.data,
            numerator_column=SUM,
            numerator_sum_squares_column=SUM_OF_SQUARES,
            denominator_column=COUNT,
            categorical_group_columns=GROUP,
            ordinal_group_column=DATE,
            interval_size=0.99
        )

    def test_multiple_difference_groupby(self):
        difference_df = self.test.multiple_difference(
            level='1',
            groupby='date',
            level_as_reference=True,
            final_expected_sample_size=1e4
        )
        assert len(difference_df) == (
                (self.data.group.unique().size - 1)
                * self.data.date.unique().size
        )
        n_comp = len(difference_df)
        assert np.allclose(
            difference_df['p-value'].map(lambda p: min(1, n_comp * p)),
            difference_df['adjusted p-value'], rtol=0.01
        )
        np.testing.assert_almost_equal(difference_df[ADJUSTED_UPPER].values[0], 0.121, 3)
        np.testing.assert_almost_equal(difference_df[ADJUSTED_LOWER].values[0], -0.151, 3)
        np.testing.assert_almost_equal(difference_df[DIFFERENCE].values[0], -0.0149, 3)
