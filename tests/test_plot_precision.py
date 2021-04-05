import spotify_confidence
import pandas as pd

OUTPUT_DIR = './tests/outputs/precision/'

spotify_confidence.options.set_option('randomization_seed', 1)


def test_bayesian_precision_plots():
    data = pd.DataFrame({
        'variation_name': [
            'test',
            'control',
        ],
        'success': [
            50000000,
            50004000,
        ],
        'total': [
            100900000,
            100900000,
        ]
    })

    test = spotify_confidence.BetaBinomial(
        data,
        numerator_column='success',
        denominator_column='total',
        categorical_group_columns=['variation_name'])

    test.summary_plot().charts[0].save(OUTPUT_DIR + 'bayesian_1.html', 'html')

    data = pd.DataFrame({
        'variation_name': ['test', 'control', 'test2'],
        'success': [
            50000000,
            50004000,
            100500000,
        ],
        'total': [
            100900000,
            100900000,
            150200000,
        ]
    })

    test = spotify_confidence.BetaBinomial(
        data,
        numerator_column='success',
        denominator_column='total',
        categorical_group_columns=['variation_name'])

    test.summary_plot().charts[0].save(OUTPUT_DIR + 'bayesian_2.html', 'html')

    test.difference_plot('control', 'test').charts[0].save(
        OUTPUT_DIR + 'bayesian_3.html', 'html')

    test.difference_plot('control', 'test2').charts[0].save(
        OUTPUT_DIR + 'bayesian_4.html', 'html')

    test.multiple_difference_plot('test2').charts[0].save(
        OUTPUT_DIR + 'bayesian_5.html', 'html')


def test_frequentist_precision_plots():
    data = pd.DataFrame({
        'variation_name': ['test', 'control', 'test2'],
        'success': [
            50000000,
            50004000,
            100500000,
        ],
        'total': [
            100900000,
            100900000,
            150200000,
        ]
    })

    test = spotify_confidence.ChiSquared(
        data,
        numerator_column='success',
        denominator_column='total',
        categorical_group_columns=['variation_name'])

    test.summary_plot().charts[0].save(OUTPUT_DIR + 'frequentist_1.html',
                                       'html')

    test.difference_plot('control', 'test').charts[0].save(
        OUTPUT_DIR + 'frequentist_2.html', 'html')

    test.difference_plot('control', 'test2').charts[0].save(
        OUTPUT_DIR + 'frequentist_3.html', 'html')

    test = spotify_confidence.ChiSquared(
        data.head(2),
        numerator_column='success',
        denominator_column='total',
        categorical_group_columns=['variation_name'])
    test.summary_plot().charts[0].save(OUTPUT_DIR + 'frequentist_4.html',
                                       'html')


def test_ordinal_precision_plots():
    data = pd.DataFrame({
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
            520,
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

    test = spotify_confidence.BetaBinomial(
        data,
        numerator_column='success',
        denominator_column='total',
        categorical_group_columns='variation_name',
        ordinal_group_column='days_since_reg')

    test.summary_plot().charts[0].save(OUTPUT_DIR + 'ordinal_1.html', 'html')
    test.summary_plot('variation_name').charts[0].save(
        OUTPUT_DIR + 'ordinal_2.html', 'html')

    data = pd.DataFrame({
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
            500000,
            800,
            10000,
            500200,
            800,
            10000,
            500100,
            900,
            10400,
            500100,
            700,
            10000,
            500200,
            800,
            10300,
        ],
        'total': [
            1000010,
            2002,
            15000,
            1000000,
            2000,
            15300,
            1000030,
            2003,
            15400,
            1000000,
            2000,
            15000,
            1000000,
            2001,
            15500,
        ],
        'days_since_reg': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
    })

    test = spotify_confidence.BetaBinomial(
        data,
        numerator_column='success',
        denominator_column='total',
        categorical_group_columns='variation_name',
        ordinal_group_column='days_since_reg')
    test.summary_plot().charts[0].save(OUTPUT_DIR + 'ordinal_3.html', 'html')
    test.summary_plot('variation_name').charts[0].save(
        OUTPUT_DIR + 'ordinal_4.html', 'html')
