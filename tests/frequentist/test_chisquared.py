"""Tests for `confidence` categorical variables."""

import pytest
import spotify_confidence
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from spotify_confidence.analysis.frequentist.statsmodels_computer \
    import ChiSquaredComputer
from spotify_confidence.analysis.constants import (POINT_ESTIMATE, VARIANCE,
                                                   SFX1, SFX2)
from spotify_confidence.analysis.confidence_utils import power_calculation


def chart_data(chart_object, series_name):
    """Retrieve data from chart object based on series name.

    Note: if there's only one series the name is None.

    Returns:
        Dictionary populated with data from the chart.
    """
    cannonical_series_name = chart_object.plot._cannonical_series_name(
        series_name)
    return chart_object.figure.select(cannonical_series_name)[0].data


class TestCategorical(object):
    def setup(self):
        np.random.seed(123)

        self.data = pd.DataFrame({
            'variation_name':
            ['test', 'control', 'test2', 'test', 'control', 'test2'],
            'success': [500, 42, 1005, 50, 40, 100],
            'total': [1009, 104, 1502, 100, 100, 150],
            'country': [
                'us',
                'us',
                'us',
                'gb',
                'gb',
                'gb',
            ]
        })

        self.test = spotify_confidence.ChiSquared(
            self.data,
            numerator_column='success',
            denominator_column='total',
            categorical_group_columns=['country', 'variation_name'])

    def test_bernoulli_squared_standard_error(self):
        n = 1000
        sample_size = int(1e7)
        p = 0.2

        x = np.random.binomial(n, p, sample_size) / n
        se = x.var()

        se_to_verify = ChiSquaredComputer._variance(
            df=pd.DataFrame({POINT_ESTIMATE: [p]}), self=None) / n

        assert (np.allclose(se_to_verify, se, atol=1e-6))

    def test_standard_error_diff_prop(self):
        sample_size = int(1e7)

        p1 = 0.1
        n1 = int(1e6)
        x1 = np.random.binomial(n1, p1, sample_size) / n1

        n2 = int(2e6)
        p2 = 0.2
        x2 = np.random.binomial(n2, p2, sample_size) / n2

        diff = x2 - x1
        std_diff = diff.std()

        comp = ChiSquaredComputer(data_frame=pd.DataFrame({'success': [1],
                                                           'n': [2]}),
                                  numerator_column='success',
                                  numerator_sum_squares_column=None,
                                  denominator_column='n',
                                  categorical_group_columns=[],
                                  ordinal_group_column=None,
                                  interval_size=None,
                                  correction_method='Bonferroni')
        diff_se = comp._std_err(pd.DataFrame({VARIANCE + SFX1: [p1*(1-p1)],
                                              VARIANCE + SFX2: [p2*(1-p2)],
                                              'n' + SFX1: [n1],
                                              'n' + SFX2: [n2]}))

        assert (np.allclose(std_diff, diff_se, atol=1e-6))

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
                0.4646901340180582, 0.309544, 0.6453118311511006,
                0.4020018007729973, 0.303982, 0.5912276177282552
            ])))
        assert (np.allclose(
            summary['ci_upper'],
            np.array([
                0.5263901434844195, 0.498147, 0.692903881232388,
                0.5979981992270027, 0.496018, 0.7421057156050781
            ])))

    def test_summary_plot(self):
        """Area plot tests"""
        ch = self.test.summary_plot()
        assert (len(ch.charts) == 1)
        ch = self.test.summary_plot(groupby='country')
        assert (len(ch.charts) == 2)

    def test_difference(self):
        with pytest.raises(ValueError):
            self.test.difference(('control', 'us'), ('test', 'usf'))

        diff = self.test.difference(('us', 'control'), ('us', 'test'))
        assert (np.allclose(diff['difference'].iloc[0], 0.091694))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.0075254411815593725))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.190913))
        assert (np.allclose(diff['p-value'].iloc[0], 0.074866))

        diff = self.test.difference(
            ('us', 'control'), ('us', 'test'), absolute=False)
        assert (np.allclose(diff['difference'].iloc[0], 0.227052))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.0186344))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.47273797))
        assert (np.allclose(diff['p-value'].iloc[0], 0.074866))

        diff = self.test.difference('control', 'test', groupby='country')
        assert (np.allclose(diff['difference'], np.array([0.100000,
                                                          0.091694])))
        assert (np.allclose(
            diff['ci_lower'], np.array([-0.03719748, -0.0075254])))
        assert (np.allclose(diff['ci_upper'], np.array([0.237197, 0.190913])))
        assert (np.allclose(diff['p-value'], np.array([0.155218, 0.074866])))

    def test_difference_with_interval_sizes(self):
        self.test._confidence_computer._interval_size = 0.99
        diff = self.test.difference(('us', 'control'), ('us', 'test'))
        assert (np.allclose(diff['difference'].iloc[0], 0.091694))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.0387024))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.22209))
        assert (np.allclose(diff['p-value'].iloc[0], 0.074866))

        self.test._confidence_computer._interval_size = 0.999
        diff = self.test.difference(('us', 'control'), ('us', 'test'))
        assert (np.allclose(diff['difference'].iloc[0], 0.091694))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.074883))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.258271))
        assert (np.allclose(diff['p-value'].iloc[0], 0.074866))

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(('bad_value', 'bad_value'),
                                      ('bad_value', 'bad_value'))

        ch = self.test.difference_plot(('us', 'control'), ('us', 'test'))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot('control', 'test', groupby='country')
        assert (len(ch.charts) == 1)

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference(('bad_value', 'bad_value'))

        diff = self.test.multiple_difference(('us', 'control'))
        assert (np.allclose(
            diff['adjusted p-value'],
            np.array([
                1.00000000e+00, 8.38342202e-01, 1.67371350e-04, 3.74330946e-01,
                1.98857657e-07
            ])))
        assert (np.allclose(
            diff['p-value'],
            np.array([
                9.553332e-01, 1.67668440e-01, 3.34742700e-05, 7.48661891e-02,
                3.97715314e-08
            ])))
        assert (np.allclose(
            diff['adjusted ci_lower'],
            np.array([
                -0.173024, -0.27489017, -0.42153065, -0.22209041, -0.39307973
            ])))
        assert (np.allclose(
            diff['adjusted ci_upper'],
            np.array(
                [0.180717, 0.08258247, -0.10411038, 0.03870244,
                 -0.13744367])))

        diff = self.test.multiple_difference('test', groupby='country')
        assert (np.allclose(
            diff['adjusted p-value'],
            np.array([
                6.208740e-01, 3.36319783e-02, 2.99464756e-01, 1.30744685e-17
            ])))
        assert (np.allclose(
            diff['p-value'],
            np.array([
                1.552185e-01, 8.40799458e-03, 7.48661891e-02, 3.26861713e-18
            ])))
        assert (np.allclose(
            diff['adjusted ci_lower'],
            np.array([-0.074839, -0.32426934, -0.03474758, -0.2232184])))
        assert (np.allclose(
            diff['adjusted ci_upper'],
            np.array([0.274839, -0.00906399, 0.21813554, -0.12391703])))

    def test_multiple_difference_level_as_reference(self):
        diff = self.test.multiple_difference('test',
                                             groupby='country',
                                             level_as_reference=True)
        assert (np.allclose(
            diff['adjusted ci_lower'],
            np.array([-0.274839, 0.00906399, -0.21813554, 0.12391703])))

    def test_multiple_difference_plot_level_as_reference(self):
        ch = self.test.multiple_difference_plot(('us', 'control'),
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_plot('test', groupby='country',
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(('bad_value', 'bad_value'),
                                               ('bad_value', 'bad_value'))

        ch = self.test.multiple_difference_plot(('us', 'control'))
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_plot('test', groupby='country')
        assert (len(ch.charts) == 1)

    def test_sample_ratio_test(self):
        expected = {
            ('us', 'test'): 0.35,
            ('us', 'control'): 0.04,
            ('us', 'test2'): 0.52,
            ('gb', 'test'): 0.035,
            ('gb', 'control'): 0.0035,
            ('gb', 'test2'): 0.0515,
        }

        with pytest.raises(TypeError):
            self.test.sample_ratio_test('not a dict')

        with pytest.raises(ValueError):
            negative = expected.copy()
            negative[('us', 'test2')] = -0.1
            self.test.sample_ratio_test(negative)

        with pytest.raises(ValueError):
            not_one = expected.copy()
            not_one[('us', 'test2')] += 0.1
            self.test.sample_ratio_test(not_one)

        with pytest.raises(ValueError):
            bad_group = expected.copy()
            bad_group[('X', 'test2')] = bad_group.pop(('us', 'test2'))
            self.test.sample_ratio_test(bad_group)

        p, df = self.test.sample_ratio_test(expected)

        obs_actual = df[self.test._denominator]
        obs_expected = sum(obs_actual) * df.expected_proportion
        _, p_ref = chisquare(obs_actual, obs_expected)

        assert(np.allclose(p, p_ref))

    def test_power_calculation_binomial(self):
        mde = 0.01
        baseline_var = 0.34 * (1 - 0.34)
        alpha = 0.03
        n1 = 225725
        n2 = 22572
        power = power_calculation(mde, baseline_var, alpha, n1, n2)

        expected_power = 0.8  # Based on G*Power calculation for 80% power

        assert(np.allclose(power, expected_power, 0.005))

    def test_power_calculation_continuous(self):
        mde = 0.03
        baseline_var = 45
        alpha = 0.05
        n1 = 859896
        n2 = 859896
        power = power_calculation(mde, baseline_var, alpha, n1, n2)

        expected_power = 0.8345828  # Based on G*Power calculation

        assert(np.allclose(power, expected_power, 0.005))

    def test_achieved_power(self):
        power_df = self.test.achieved_power(level_1=('us', 'control'),
                                            level_2=('us', 'test'),
                                            mde=0.1429118527,
                                            alpha=0.05)
        power = power_df.achieved_power
        expected_power = 0.8  # Based on G*Power calculation
        assert(np.allclose(power, expected_power, 0.01))

    def test_achieved_power_groupby(self):
        power_df = self.test.achieved_power(level_1='control',
                                            level_2='test',
                                            mde=0.1429118527,
                                            alpha=0.05,
                                            groupby='country')
        power = power_df.achieved_power
        expected_power = [0.53, 0.800]  # Based on G*Power calculation
        assert(np.allclose(power, expected_power, 0.01))


class TestOrdinal(object):
    def setup(self):
        self.data = pd.DataFrame({
            'variation_name': [
                'test', 'control', 'test2',
                'test', 'control', 'test2',
                'test', 'control', 'test2',
                'test', 'control', 'test2',
                'test', 'control', 'test2',
            ],
            'success': [
                500, 8, 100,
                510, 8, 100,
                520, 9, 104,
                530, 7, 100,
                530, 8, 103],
            'total': [
                1010, 22, 150,
                1000, 20, 153,
                1030, 23, 154,
                1000, 20, 150,
                1040, 21, 155],
            'days_since_reg': [
                1, 1, 1,
                2, 2, 2,
                3, 3, 3,
                4, 4, 4,
                5, 5, 5],
        })

        self.test = spotify_confidence.ChiSquared(
            self.data,
            numerator_column='success',
            denominator_column='total',
            categorical_group_columns='variation_name',
            ordinal_group_column='days_since_reg')

    def test_summary(self):
        """Area plot tests"""
        summary = self.test.summary()
        assert (np.array_equal(
            summary['point_estimate'], self.data.success / self.data.total))
        assert (np.allclose(
            summary['ci_lower'],
            np.array([
                0.46421506, 0.16262386, 0.59122762, 0.47901645, 0.1852967,
                0.57819867, 0.4743207, 0.19185087, 0.60136948, 0.49906608,
                0.1409627, 0.59122762, 0.47923305, 0.17325271, 0.59018498
            ])))
        assert (np.allclose(
            summary['ci_upper'],
            np.array([
                0.52588395, 0.56464887, 0.74210572, 0.54098355, 0.6147033,
                0.72899087, 0.53538804, 0.59075782, 0.74927987, 0.56093392,
                0.5590373, 0.74210572, 0.53999772, 0.58865206, 0.73884727
            ])))

    def test_summary_plot(self):
        """Area plot tests"""
        ch = self.test.summary_plot()
        assert (len(ch.charts) == 1)

        ch = self.test.summary_plot().charts[0]
        np.array_equal(
            chart_data(ch, 'control')['days_since_reg'],
            np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1]))

    def test_p_value_is_symmetric(self):
        diff1 = self.test.difference(('control', 1), ('test', 1))
        assert (np.allclose(diff1['difference'].iloc[0], 0.131413))
        assert (np.allclose(diff1['p-value'].iloc[0], 0.222578))

        diff2 = self.test.difference(('test', 1), ('control', 1))
        assert (np.allclose(diff2['difference'].iloc[0], -0.131413))
        assert (np.allclose(diff2['p-value'].iloc[0], 0.222578))

        assert (diff1['p-value'].iloc[0] == diff2['p-value'].iloc[0])

    def test_raise_error_with_nim(self):
        with pytest.raises(ValueError):
            self.test.difference(('control', 1),
                                 ('test', 1),
                                 non_inferiority_margins=('blah', 'hah'))

    def test_difference(self):
        with pytest.raises(ValueError):
            self.test.difference(('control', 'us'), ('test', 'usf'))

        diff = self.test.difference(('control', 1), ('test', 1))
        assert (np.allclose(diff['difference'].iloc[0], 0.131413))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.071951))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.334777))
        assert (np.allclose(diff['p-value'].iloc[0], 0.222578))

        diff = self.test.difference(
            ('control', 1), ('test', 1), absolute=False)
        assert (np.allclose(diff['difference'].iloc[0], 0.361386))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.1978640))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.92063629))
        assert (np.allclose(diff['p-value'].iloc[0], 0.222578))

        diff = self.test.difference(
            'control', 'test', groupby='days_since_reg')
        assert (np.allclose(
            diff['difference'],
            np.array([0.13141314, 0.11, 0.11355002, 0.18, 0.128663])))

        diff = self.test.difference(('control', 1), ('test', 1))
        assert (np.allclose(diff['difference'].iloc[0], 0.131413))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.071951))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.334777))
        assert (np.allclose(diff['p-value'].iloc[0], 0.222578))

        diff = self.test.difference(
            ('control', 1), ('test', 1), absolute=False)
        assert (np.allclose(diff['difference'].iloc[0], 0.361386))
        assert (np.allclose(diff['ci_lower'].iloc[0], -0.197864))
        assert (np.allclose(diff['ci_upper'].iloc[0], 0.9206363))
        assert (np.allclose(diff['p-value'].iloc[0], 0.222578))

        diff = self.test.difference(
            'control', 'test', groupby='days_since_reg')
        assert (np.allclose(
            diff['difference'],
            np.array([0.13141314, 0.11, 0.11355002, 0.18, 0.128663])))

    def test_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.difference_plot(('control', 'us'), ('test', 'usf'))

        ch = self.test.difference_plot(('control', 1), ('test', 1))
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot('control', 'test',
                                       groupby='days_since_reg')
        assert (len(ch.charts) == 1)

    def test_sample_ratio_test(self):
        expected = {
            ('control', 1): 0.004,
            ('control', 2): 0.003,
            ('control', 3): 0.004,
            ('control', 4): 0.003,
            ('control', 5): 0.004,
            ('test', 1): 0.15,
            ('test', 2): 0.168,
            ('test', 3): 0.173,
            ('test', 4): 0.168,
            ('test', 5): 0.195,
            ('test2', 1): 0.025,
            ('test2', 2): 0.026,
            ('test2', 3): 0.026,
            ('test2', 4): 0.025,
            ('test2', 5): 0.026,
        }

        p, df = self.test.sample_ratio_test(expected)

        obs_actual = df[self.test._denominator]
        obs_expected = sum(obs_actual) * df.expected_proportion
        _, p_ref = chisquare(obs_actual, obs_expected)

        assert(np.allclose(p, p_ref))


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

        self.test = spotify_confidence.ChiSquared(
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
            self.test.difference(('control', 'us'),
                                 ('test', 'usf'), groupby='days_since_reg')

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

    def test_multiple_difference(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference(level='non_existingg_level')

        df = self.test.multiple_difference(
            level='control',
            groupby=['country', 'days_since_reg'])
        assert (len(df) == 20)

        df = self.test.multiple_difference(
            level='us',
            groupby=['variation_name', 'days_since_reg'])
        assert (len(df) == 15)

        df = self.test.multiple_difference(
            level=1,
            groupby=['country', 'variation_name'])
        assert (len(df) == 24)

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
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1=('control', 'us'),
                                       level_2=('test', 'us'),
                                       groupby='days_since_reg')
        assert (len(ch.charts) == 1)

        ch = self.test.difference_plot(level_1='control', level_2='test',
                                       groupby=['country', 'days_since_reg'])
        assert (len(ch.charts) == 1)

    def test_multiple_difference_plot(self):
        with pytest.raises(ValueError):
            self.test.multiple_difference_plot(level='control')

        ch = self.test.multiple_difference_plot(level='control',
                                                groupby=['country',
                                                         'days_since_reg'],
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_plot(level='control',
                                                groupby=['country',
                                                         'days_since_reg'],
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)

        ch = self.test.multiple_difference_plot(level=1,
                                                groupby=['country',
                                                         'variation_name'],
                                                level_as_reference=True)
        assert (len(ch.charts) == 1)
