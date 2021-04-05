"""Tests for `confidence` categorical variables."""

import pytest
import spotify_confidence
import pandas as pd
import numpy as np

spotify_confidence.options.set_option('randomization_seed', 1)


class TestCategorical(object):
    def setup(self):

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

        self.test = spotify_confidence.BetaBinomial(
            self.data,
            numerator_column='success',
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
                0.46473987, 0.3132419, 0.64500415, 0.40317395, 0.1530671,
                0.58861002
            ])))
        assert (np.allclose(
            summary['ci_upper'],
            np.array([
                0.52636595, 0.49971958, 0.69256054, 0.59682605, 0.69632051,
                0.73836135
            ])))

    def test_summary_plot(self):
        """Area plot tests"""
        ch = self.test.summary_plot()
        assert (len(ch.charts) == 1)
        ch = self.test.summary_plot(groupby='country')
        assert (len(ch.charts) == 2)

    def test_difference(self):
        diff = self.test.difference(('us', 'control'), ('us', 'test'))
        assert (np.allclose(diff['difference'].iloc[0], 0.090737))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.0092189024))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.18712235))
        assert (np.allclose(
            diff["P(level_2 > level_1)"].iloc[0], 0.962782))
        assert (np.allclose(diff["level_1 potential loss"].iloc[0],
                            -0.091504838))
        assert (np.allclose(diff["level_1 potential gain"].iloc[0],
                            0.000767784))
        assert (np.allclose(diff["level_2 potential loss"].iloc[0],
                            -0.000767784))
        assert (np.allclose(diff["level_2 potential gain"].iloc[0],
                            0.091504838))

        diff = self.test.difference(
            ('us', 'control'), ('us', 'test'), absolute=False)
        assert (np.allclose(diff['difference'].iloc[0], 0.2416812))
        assert (np.allclose(
            diff["P(level_2 > level_1)"].iloc[0], 0.962782))
        assert (np.allclose(diff["level_1 potential loss"].iloc[0],
                            -0.2431594))

        diff = self.test.difference(
            'control', 'test', groupby='country', absolute=False)
        assert (np.allclose(diff['difference'], np.array([0.428745,
                                                          0.241799])))
        assert (np.allclose(diff['P(level_2 > level_1)'],
                            np.array([0.725982, 0.963228])))
        assert (np.allclose(
            diff['level_2 potential loss'],
            np.array([-0.042689, -0.001464]),
            rtol=1e-05,
            atol=1e-06))

    def test_difference_with_interval_sizes(self):
        self.test._interval_size = 0.99
        diff = self.test.difference(('us', 'control'), ('us', 'test'))
        assert (np.allclose(diff['difference'].iloc[0], 0.090737))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.040760))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.21614))

        self.test._interval_size = 0.999
        diff = self.test.difference(('us', 'control'), ('us', 'test'))
        assert (np.allclose(diff['difference'].iloc[0], 0.090737))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.0769830))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.2479583))

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(('bad_value', 'bad_value'),
                                      ('bad_value', 'bad_value'))

        ch = self.test.difference_plot(('us', 'control'), ('us', 'test'))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot('control', 'test', groupby='country')
        assert (len(ch.charts) == 2)

    def test_multiple_difference_joint(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_joint(('bad_value', 'bad_value'))

        diff = self.test.multiple_difference_joint(('us', 'test2'))
        print(np.random.get_state()[1][0])
        assert (np.allclose(
            diff['difference'], np.array([0.000743]), rtol=1e-05, atol=1e-06))
        assert (np.allclose(diff["P(('us', 'test2') >= all)"],
                            np.array([0.508644])))
        assert (np.allclose(diff["('us', 'test2') potential loss"],
                            np.array([-0.032459])))

        diff = self.test.multiple_difference_joint('test2', groupby='country')
        assert (np.allclose(diff['test2 potential loss'], np.array([-0.054338,
                                                                    0.])))

    def test_multiple_difference_joint_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_joint_plot(('bad_value', 'bad_value'),
                                                     ('bad_value', 'bad_value'))

        ch = self.test.multiple_difference_joint_plot(('us', 'control'))
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_joint_plot('test', groupby='country')
        assert (len(ch.charts) == 2)

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference('bad_value')

        md = self.test.multiple_difference('control', groupby='country')
        for i, row in md.iterrows():
            diff_value = self.test.difference(
                (row['country'], row['level_1']),
                (row['country'], row['level_2']))['difference'].values[0]
            assert (np.allclose(row['difference'], diff_value, rtol=1e-02))

    def test_multiple_difference_plot(self):
        ch = self.test.multiple_difference_plot(('us', 'control'))
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_plot('test', groupby='country')
        assert (len(ch.charts) == 2)

    def test_multiple_difference_level_as_reference(self):
        md = self.test.multiple_difference('control',
                                           groupby='country',
                                           level_as_reference=True)
        for i, row in md.iterrows():
            diff_value = self.test.difference(
                (row['country'], row['level_1']),
                (row['country'], row['level_2']))['difference'].values[0]
            assert (np.allclose(row['difference'], diff_value, rtol=1e-02))


class TestOrdinal:
    def setup(self):

        self.data = pd.DataFrame({
            'variation_name': [
                'test',
                'control',
                'test2',
                'test',
                'control',
                'test2',
                'test',
                'control',
                'test2',
                'test',
                'control',
                'test2',
                'test',
                'control',
                'test2',
            ],
            'success': [
                500,
                8,
                100,
                510,
                8,
                100,
                520,
                9,
                104,
                530,
                7,
                100,
                530,
                8,
                103,
            ],
            'total': [
                1010,
                22,
                150,
                1000,
                20,
                153,
                1030,
                23,
                154,
                1000,
                20,
                150,
                1040,
                21,
                155,
            ],
            'days_since_reg': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        })

        self.test = spotify_confidence.BetaBinomial(
            self.data,
            numerator_column='success',
            denominator_column='total',
            categorical_group_columns='variation_name',
            ordinal_group_column='days_since_reg')

    def test_summary(self):
        """Area plot tests"""
        summary = self.test.summary()
        assert (np.array_equal(
            summary['point_estimate'], summary[self.test._numerator_column] /
            summary[self.test._denominator_column]))
        assert (np.allclose(
            summary['ci_lower'],
            np.array([
                0.46426613, 0.18932289, 0.58861002, 0.479025, 0.21062941,
                0.57588269, 0.47434292, 0.21431068, 0.59864266, 0.49901672,
                0.17227621, 0.58861002, 0.47924165, 0.19940202, 0.5876837
            ])))
        assert (np.allclose(
            summary['ci_upper'],
            np.array([
                0.52586121, 0.57128953, 0.73836135, 0.54091721, 0.61607968,
                0.72555745, 0.53533858, 0.59380455, 0.74548633, 0.56080991,
                0.56776609, 0.73836135, 0.53993569, 0.592895, 0.73526909
            ])))

    def test_summary_plot(self):
        """Area plot tests"""
        ch = self.test.summary_plot()
        assert (len(ch.charts) == 1)

    def test_difference(self):
        with pytest.raises(ValueError):
            self.test.difference(('control', 'us'), ('test', 'usf'))

        diff = self.test.difference(('control', 1), ('test', 1))
        # Add more assertions here

        diff = self.test.difference(
            ('control', 1), ('test', 1), absolute=False)
        # Add more assertions here

        diff = self.test.difference(
            'control', 'test', groupby='days_since_reg')
        # Add more assertions here
        assert len(diff) > 0

    def test_multiple_difference(self):
        diff = self.test.multiple_difference(level='control',
                                             groupby='days_since_reg',
                                             level_as_reference=True)
        assert (len(diff) == 10)

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot('control', 'not_a_valid_level')

        ch = self.test.difference_plot(('control', 1), ('test', 1))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot('control', 'test',
                                       groupby='days_since_reg')
        assert (len(ch.charts) == 1)

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(level='control')

        ch = self.test.multiple_difference_plot(level='control',
                                                groupby=['days_since_reg'],
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)


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
             'success': [500, 8, 100,
                         510, 8, 100,
                         520, 9, 104,
                         530, 7, 100,
                         530, 8, 103,
                         500, 8, 100,
                         510, 8, 100,
                         520, 9, 104,
                         530, 7, 100,
                         530, 8, 103, ],
             'total': [2010, 42, 250,
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

        self.test = spotify_confidence.BetaBinomial(
            self.data,
            numerator_column='success',
            denominator_column='total',
            categorical_group_columns=['variation_name', 'country'],
            ordinal_group_column='days_since_reg')

    def test_summary_plot(self):
        ch = self.test.summary_plot()
        assert (len(ch.charts) == 1)

        ch = self.test.summary_plot(groupby=['country'])
        assert (len(ch.charts) == 2)

        ch = self.test.summary_plot(groupby=['days_since_reg'])
        assert (len(ch.charts) == 5)

        ch = self.test.summary_plot(groupby=['country', 'days_since_reg'])
        assert (len(ch.charts) == 10)

    def test_difference(self):
        with pytest.raises(ValueError):
            self.test.difference(('control', 'us'), ('test', 'usf'),
                                 groupby='days_since_reg')

        df = self.test.difference(level_1=('test', 'us'),
                                  level_2=('control', 'us'),
                                  groupby='days_since_reg')
        assert (len(df) == 5)
        assert ('days_since_reg' in df.columns)

        df = self.test.difference(level_1=('test', 'us'),
                                  level_2=('control', 'us'),
                                  groupby=['days_since_reg'])
        assert (len(df) == 5)
        assert ('days_since_reg' in df.columns)

        df = self.test.difference(level_1=('test', 1),
                                  level_2=('control', 1),
                                  groupby=['country'])
        assert (len(df) == 2)
        assert ('country' in df.columns)

        df = self.test.difference(level_1='test',
                                  level_2='control',
                                  groupby=['country', 'days_since_reg'])
        assert (len(df) == 10)
        assert ('country' in df.columns)
        assert ('days_since_reg' in df.columns)

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(('control', 'us', 10), ('test', 'us', 10))

        ch = self.test.difference_plot(level_1=('control', 'us', 1),
                                       level_2=('test', 'us', 1))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1=('control', 'us'),
                                       level_2=('test', 'us'),
                                       groupby='days_since_reg')
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1='control', level_2='test',
                                       groupby=['country', 'days_since_reg'])
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1=('control', 1),
                                       level_2=('test', 1), groupby='country')
        assert (len(ch.charts) == 2)

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(level='control')

        ch = self.test.multiple_difference_plot(level='control',
                                                groupby=['country',
                                                         'days_since_reg'],
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)
