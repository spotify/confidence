import numpy as np
import pandas as pd
import pytest
import spotify_confidence
from spotify_confidence.analysis.constants import (
    INCREASE_PREFFERED,
    DECREASE_PREFFERED,
    POINT_ESTIMATE,
    CI_LOWER,
    CI_UPPER,
    P_VALUE,
    ADJUSTED_LOWER,
    ADJUSTED_UPPER,
    DIFFERENCE,
    BONFERRONI,
    BONFERRONI_DO_NOT_COUNT_NON_INFERIORITY,
    CORRECTION_METHODS,
    SPOT_1,
    CORRECTION_METHODS_THAT_SUPPORT_CI,
)

class Test(object):
    def setup(self):
        np.random.seed(123)
        n = 10000
        d = np.random.randint(2, size=n)
        x = np.random.standard_normal(size=n)
        y = 0.5 * d + 0.5 * x + np.random.standard_normal(size=n)
        country = np.random.choice(['us', 'se'], n)
        data = pd.DataFrame(
            {'variation_name': list(map(str, d)),
             'y': y,
             'x': x,
             'country': country})
        data = data.assign(xy=y * x) \
            .assign(x2=x ** 2) \
            .assign(y2=y ** 2) \
            .groupby(['variation_name', 'country']) \
            .agg({
            'y': ['sum', 'count'],
            'y2': 'sum',
            'x': 'sum',
            'x2': 'sum',
            'xy': 'sum'}) \
            .reset_index()

        data.columns = data.columns.map('_'.join).str.strip('_')
        self.data = data


        self.test = spotify_confidence.ZTestLinreg(
                   data_frame=data,
                   numerator_column='y_sum',
                   numerator_sum_squares_column='y2_sum',
                   denominator_column='y_count',
                   categorical_group_columns=['variation_name', 'country'],
                   feature_column = 'x_sum',
                   feature_sum_squares_column = 'x2_sum',
                   feature_cross_sum_column = 'xy_sum',
                   interval_size=0.99,
                   correction_method='bonferroni')

    def test_summary(self):
        summary_df = self.test.summary()
        print(summary_df)
        assert len(summary_df) == len(self.data)


