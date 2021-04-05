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

import pandas as pd
import numpy as np
from itertools import product


def example_data_binomial():
    """
    Returns an output dataframe with categorical
    features (country and test variation), and orginal features (date),
    as well as number of successes and total observations for each combination
    """
    countries = ['ca', 'us']
    dates = pd.date_range('2018-01-01', '2018-02-01')
    variation_names = ['test', 'control', 'test2']

    # test ca, test us, control ca, control us, test2 ca, test2 us
    success_rates = [.3, .32, .24, .22, .25, .42]
    n_observations = [50, 80, 30, 50, 40, 50]

    return_df = pd.DataFrame()

    for i, (country, variation) in enumerate(
            product(countries, variation_names)):
        df = pd.DataFrame({'date': dates})
        df['country'] = country
        df['variation_name'] = variation
        df['total'] = np.random.poisson(n_observations[i], size=len(dates))
        df['success'] = df['total'].apply(
            lambda x: np.random.binomial(x, success_rates[i]))
        return_df = pd.concat([return_df, df], axis=0)

    return return_df


def example_data_gaussian():
    df = pd.DataFrame({
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
        'nr_of_items': [
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
        'nr_of_items_sumsq': [
            2500,
            12,
            150,
            2510,
            13,
            140,
            2520,
            14,
            154,
            2530,
            15,
            160,
            2530,
            16,
            103,
        ],
        'users': [
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

    return df
