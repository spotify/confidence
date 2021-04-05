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

import numpy as np
from pandas import DataFrame
from scipy.stats import chi2
from typing import Dict, Tuple, Iterable


def sample_ratio_test(df: DataFrame,
                      all_group_columns: Iterable,
                      denominator: str,
                      expected_proportions: Dict) -> Tuple[float, DataFrame]:
    """Goodness of fit test of observed vs. expected group frequencies.

    Tests whether the observed proportion of total users in each group
    are likely to come from the sampling distribution using a Pearson's
    chi-squared test:
    https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

    Args:
        expected_proportions (dict): Expected proportion of observations in
            each group with group-keys as keys and proportions as values.

    Returns:
        float: p-value based on the null hypothesis that observed
            proportions are drawn from the sampling distribution.
        pandas.DataFrame with the following columns:
        - denominator column of original data.
        - observed_proportion: Observed share in the group.
        - expected_proportion: Expected share in the group.
        - difference: observed - expected shares.
    """

    if not isinstance(expected_proportions, dict):
        raise TypeError('`expected_proportions` must be a dict with '
                        'groupings as keys and expected proportions '
                        'as values')
    elif not np.allclose(sum(expected_proportions.values()), 1.0):
        raise ValueError('proportions must sum to one')
    elif not (np.array(list(expected_proportions.values())) > 0).all():
        raise ValueError('proportions must all be positive')

    all_groups = list(df.groupby(all_group_columns).groups.keys())
    if set(all_groups) != set(expected_proportions.keys()):
        raise ValueError(
            f"`expected_proportion` keys must match groupings in the "
            f"order {all_group_columns}")

    n_tot = df[denominator].sum()

    grouped_data = df.groupby(all_group_columns)
    sr_df = grouped_data.sum()
    sr_df['observed_proportion'] = np.zeros(len(sr_df))
    sr_df['expected_proportion'] = np.zeros(len(sr_df))
    sr_df['difference'] = np.zeros(len(sr_df))

    a = 0
    for grouping, expected_proportion in expected_proportions.items():
        try:
            n_group = (grouped_data.get_group(grouping)[denominator].iloc[0])
        except KeyError as e:
            raise KeyError(f"{e} is not a valid group")

        actual_proportion = n_group / n_tot
        diff = actual_proportion - expected_proportion
        sq_diff = np.power(diff, 2)
        a += sq_diff / expected_proportion

        sr_df.loc[grouping, 'observed_proportion'] = actual_proportion
        sr_df.loc[grouping, 'expected_proportion'] = expected_proportion
        sr_df.loc[grouping, 'difference'] = diff

    chi2_stat = n_tot * a
    deg_freedom = len(grouped_data) - 1
    p_value = 1 - chi2.cdf(chi2_stat, deg_freedom)

    return p_value, sr_df
