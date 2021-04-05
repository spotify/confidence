"""Tests for `confidence` sample size calculation."""

import pytest
import spotify_confidence as conf
import numpy as np


class TestSampleSizeCalc(object):

    def test_continuous_default(self):
        n_tot, _, allocations = conf.SampleSize.continuous(
            average_absolute_mde=0.3, baseline_variance=100)
        g_power_n_tot = 39906
        #: Reference sample size calculated with G*Power.
        assert (np.allclose(n_tot, g_power_n_tot, 0.005) and
                (allocations == [0.5, 0.5]).all())

    def test_binomial_default_small(self):
        n_tot, _, allocations = conf.SampleSize.binomial(
            absolute_percentage_mde=0.003, baseline_proportion=0.34)

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 897348
        assert (np.allclose(n_tot, g_power_n_tot, 0.005) and
                (allocations == [0.5, 0.5]).all())

    def test_binomial_default_large(self):
        n_tot, _, allocations = conf.SampleSize.binomial(
            absolute_percentage_mde=0.1, baseline_proportion=0.34)

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = np.average((738, 852))
        assert (np.allclose(n_tot, g_power_n_tot, 0.015) and
                (allocations == [0.5, 0.5]).all())

    def test_binomial_alpha(self):
        n_tot, _, _ = conf.SampleSize.binomial(absolute_percentage_mde=0.003,
                                               baseline_proportion=0.34,
                                               alpha=0.14)

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 630710
        np.allclose(n_tot, g_power_n_tot, 0.005)

    def test_binomial_alpha_power(self):
        n_tot, _, _ = conf.SampleSize.binomial(absolute_percentage_mde=0.003,
                                               baseline_proportion=0.34,
                                               alpha=0.14,
                                               power=0.67)

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 366424
        assert np.allclose(n_tot, g_power_n_tot, 0.005)

    def test_binomial_treatments(self):
        n_tot, _, allocations = conf.SampleSize.binomial(
            absolute_percentage_mde=0.003,
            baseline_proportion=0.34,
            treatments=3)

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 673157 + 336578 * 2
        assert (np.allclose(n_tot, g_power_n_tot, 0.005) and
                (allocations == [1/2, 1/4, 1/4]).all())

    def test_binomial_treatments_all_vs_all(self):
        n_tot, _, allocations = conf.SampleSize.binomial(
            absolute_percentage_mde=0.003,
            baseline_proportion=0.34,
            treatments=3,
            comparisons='all_vs_all')

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 448674 * 3
        assert (np.allclose(n_tot, g_power_n_tot, 0.005) and
                (allocations == [1/3, 1/3, 1/3]).all())

    def test_continuous_treatments(self):
        n_tot, _, allocations = conf.SampleSize.continuous(
            average_absolute_mde=0.3,
            baseline_variance=100,
            treatments=3)

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 29929 + 14965 * 2
        assert (np.allclose(n_tot, g_power_n_tot, 0.005) and
                (allocations == [1/2, 1/4, 1/4]).all())

    def test_continuous_treatments_all_vs_all(self):
        n_tot, _, allocations = conf.SampleSize.continuous(
            average_absolute_mde=0.3,
            baseline_variance=100,
            treatments=3,
            comparisons='all_vs_all')

        #: Reference sample size calculated with G*Power.
        g_power_n_tot = 19953 * 3
        assert (np.allclose(n_tot, g_power_n_tot, 0.005) and
                (allocations == [1/3, 1/3, 1/3]).all())

    def test_clean_treatments_int(self):
        returned_treatments = conf.SampleSize._clean_treatments(4)
        assert(returned_treatments == 4)

    def test_clean_treatments_negative(self):
        with pytest.raises(ValueError):
            conf.SampleSize._clean_treatments(-4)

    def test_clean_treatments_non_integer(self):
        with pytest.raises(ValueError):
            conf.SampleSize._clean_treatments(4.1)

    def test_clean_treatments_string(self):
        with pytest.raises(TypeError):
            conf.SampleSize._clean_treatments("X")

    def test_clean_comparisons_control_vs_all(self):
        comparison = 'control_vs_all'
        returned_comparisons = conf.SampleSize._clean_comparisons(comparison)
        assert(comparison == returned_comparisons)

    def test_clean_comparisons_all_vs_all(self):
        comparison = 'all_vs_all'
        returned_comparisons = conf.SampleSize._clean_comparisons(comparison)
        assert(comparison == returned_comparisons)

    def test_clean_comparisons_wrong_input(self):
        with pytest.raises(ValueError):
            conf.SampleSize._clean_comparisons('X')

    def test_num_comparisons_control_vs_all(self):
        comparisons = conf.SampleSize._num_comparisons(5, 'control_vs_all')
        assert(comparisons == 4)

    def test_num_comparisons_all_vs_all_two(self):
        comparisons = conf.SampleSize._num_comparisons(2, 'all_vs_all')
        assert(comparisons == 1)

    def test_num_comparisons_all_vs_all_five(self):
        comparisons = conf.SampleSize._num_comparisons(5, 'all_vs_all')
        assert(comparisons == 10)

    def test_get_comparison_matrix_control_vs_all(self):
        returned_comparisons = (conf.SampleSize
                                ._get_comparison_matrix(4, 'control_vs_all'))
        expected_comparisons = np.array([[0., 0., 0., 0.],
                                         [1., 0., 0., 0.],
                                         [1., 0., 0., 0.],
                                         [1., 0., 0., 0.]])
        assert(np.array_equal(returned_comparisons, expected_comparisons))

    def test_get_comparison_matrix_all_vs_all(self):
        returned_comparisons = (conf.SampleSize
                                ._get_comparison_matrix(4, 'all_vs_all'))
        expected_comparisons = np.array([[0., 0., 0., 0.],
                                         [1., 0., 0., 0.],
                                         [1., 1., 0., 0.],
                                         [1., 1., 1., 0.]])
        assert(np.array_equal(returned_comparisons, expected_comparisons))

    def test_clean_treatment_costs_default(self):
        cost_array = conf.SampleSize._clean_treatment_costs(3, None)
        assert (np.array_equal(np.ones(3), cost_array))

    def test_clean_treatment_costs_custom(self):
        custom_treatment_costs = np.array((1, 2, 3))
        returned_treatment_costs = (
            conf.SampleSize._clean_treatment_costs(3, custom_treatment_costs))
        assert (np.array_equal(custom_treatment_costs,
                               returned_treatment_costs))

    def test_clean_treatment_costs_negative(self):
        with pytest.raises(ValueError):
            cost_array = np.array([0.3, 1, -2])
            conf.SampleSize._clean_treatment_costs(3, cost_array)

    def test_clean_treatment_costs_contains_string(self):
        with pytest.raises(TypeError):
            cost_array = np.array([0.3, 1, "X"])
            conf.SampleSize._clean_treatment_costs(3, cost_array)

    def test_clean_treatment_costs_shape(self):
        with pytest.raises(TypeError):
            cost_array = np.array([0.3, 1, 2])
            conf.SampleSize._clean_treatment_costs(4, cost_array)

    def test_clean_treatment_costs_int(self):
        with pytest.raises(TypeError):
            cost_array = 2
            conf.SampleSize._clean_treatment_costs(4, cost_array)

    def test_get_treatment_allocations_auto_control_vs_all(self):
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'control_vs_all', None, None))
        expected_treatment_allocations = np.array((1/2, 1/4, 1/4))
        assert (np.array_equal(returned_treatment_allocations,
                               expected_treatment_allocations))

    def test_get_treatment_allocations_auto_all_vs_all(self):
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'all_vs_all', None, None))
        expected_treatment_allocations = np.array((1/3, 1/3, 1/3))
        assert (np.array_equal(returned_treatment_allocations,
                               expected_treatment_allocations))

    def test_get_treatment_allocations_auto_control_vs_all_custom_cost(self):
        custom_cost = np.array((1, 2, 2))
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'control_vs_all',
                                            custom_cost, None))
        ratios = np.array((1, 1/2 * np.sqrt(1/2), 1/2 * np.sqrt(1/2)))
        expected_treatment_allocations = ratios / np.sum(ratios)
        assert (np.array_equal(returned_treatment_allocations,
                               expected_treatment_allocations))

    def test_get_treatment_allocations_auto_all_vs_all_custom_cost(self):
        custom_cost = np.array((1, 2, 2))
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'all_vs_all',
                                            custom_cost, None))
        ratios = np.array((1, np.sqrt(1/2), np.sqrt(1/2)))
        expected_treatment_allocations = ratios / np.sum(ratios)
        assert (np.array_equal(returned_treatment_allocations,
                               expected_treatment_allocations))

    def test_get_treatment_allocations_custom(self):
        custom_treatment_allocations = np.array((0.34, 0.56, 0.10))
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'control_vs_all', None,
                                            custom_treatment_allocations))
        assert (np.array_equal(returned_treatment_allocations,
                               custom_treatment_allocations))

    def test_get_treatment_allocations_from_custom_list(self):
        custom_treatment_allocations = [0.34, 0.56, 0.10]
        expected_treatment_allocations = np.array(custom_treatment_allocations)
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'control_vs_all', None,
                                            custom_treatment_allocations))
        print('------------- returned:', returned_treatment_allocations)
        assert (np.array_equal(returned_treatment_allocations,
                               expected_treatment_allocations))

    def test_get_treatment_allocations_from_custom_tuple(self):
        custom_treatment_allocations = (0.34, 0.56, 0.10)
        expected_treatment_allocations = np.array(custom_treatment_allocations)
        returned_treatment_allocations = (
            conf.SampleSize
                ._get_treatment_allocations(3, 'control_vs_all', None,
                                            custom_treatment_allocations))
        print('------------- returned:', returned_treatment_allocations)
        assert (np.array_equal(returned_treatment_allocations,
                               expected_treatment_allocations))

    def test_get_treatment_allocations_custom_negative(self):
        with pytest.raises(ValueError):
            custom_treatment_allocations = np.array((0.34, 0.56, -0.10))
            conf.SampleSize._get_treatment_allocations(
                3, 'control_vs_all', None, custom_treatment_allocations)

    def test_get_treatment_allocations_custom_sum_too_large(self):
        with pytest.raises(ValueError):
            custom_treatment_allocations = np.array((0.35, 0.56, 0.10))
            conf.SampleSize._get_treatment_allocations(
                3, 'control_vs_all', None, custom_treatment_allocations)

    def test_get_treatment_allocations_custom_wrong_shape(self):
        with pytest.raises(TypeError):
            custom_treatment_allocations = np.array((0.34, 0.56, 0.05, 0.05))
            conf.SampleSize._get_treatment_allocations(
                3, 'control_vs_all', None, custom_treatment_allocations)

    def test_get_alpha_no_bonferroni(self):
        alpha = conf.SampleSize._get_alpha(0.13, 0.85, False,
                                           5, 'control_vs_all')
        assert(alpha == 0.13)

    def test_get_alpha_bonferroni_control_vs_all(self):
        alpha = conf.SampleSize._get_alpha(0.13, 0.85, True, 5,
                                           'control_vs_all')
        assert(alpha == 0.13/4)

    def test_get_alpha_bonferroni_all_vs_all(self):
        alpha = conf.SampleSize._get_alpha(0.13, 0.85, True, 5, 'all_vs_all')
        assert(alpha == 0.13/10)

    def test_get_alpha_wrong_bonferroni_type(self):
        with pytest.raises(TypeError):
            conf.SampleSize._get_alpha(0.13, 0.85, 0., 5, 'control_vs_all')

    def test_get_alpha_out_of_bounds(self):
        with pytest.raises(ValueError):
            conf.SampleSize._get_alpha(1.13, 0.85, False, 5, 'control_vs_all')

    def test_get_alpha_type(self):
        with pytest.raises(TypeError):
            conf.SampleSize._get_alpha(np.array((0.13)), 0.85, False,
                                       5, 'control_vs_all')

    def test_get_alpha_low_power(self):
        with pytest.raises(ValueError):
            conf.SampleSize._get_alpha(0.13, 0.10, False, 5, 'control_vs_all')

    def test_validate_percentage(self):
        num = conf.SampleSize._validate_percentage(0.47)
        assert(num == 0.47)

    def test_validate_percentage_out_of_bounds(self):
        with pytest.raises(ValueError):
            conf.SampleSize._validate_percentage(1.47)

    def test_validate_percentage_type(self):
        with pytest.raises(TypeError):
            conf.SampleSize._validate_percentage(np.array(0.47))

    def test_validate_positive(self):
        val = 0.3
        return_val = conf.SampleSize._validate_positive(val)
        assert(return_val == val)

    def test_validate_positive_infinity(self):
        val = np.inf
        return_val = conf.SampleSize._validate_positive(val)
        assert(return_val == val)

    def test_validate_positive_negative(self):
        with pytest.raises(ValueError):
            conf.SampleSize._validate_positive(-0.3)

    def test_validate_positive_type(self):
        with pytest.raises(TypeError):
            conf.SampleSize._validate_positive("X")

    def test_clean_continuous_mde(self):
        mde = conf.SampleSize._clean_continuous_mde(3.44)
        assert(mde == 3.44)

    def test_clean_continuous_mde_zero(self):
        with pytest.raises(ValueError):
            conf.SampleSize._clean_continuous_mde(0)

    def test_clean_binomial_mde(self):
        mde = conf.SampleSize._clean_binomial_mde(0.03, 0.47)
        assert(mde == 0.03)

    def test_clean_binomial_large(self):
        mde = conf.SampleSize._clean_binomial_mde(0.49, 0.47)
        assert(mde == 0.49)

    def test_clean_binomial_too_large(self):
        with pytest.raises(ValueError):
            conf.SampleSize._clean_binomial_mde(0.54, 0.47)
