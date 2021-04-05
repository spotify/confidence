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

from ipywidgets import widgets
from IPython.display import display
import scipy.stats as st
import numpy as np
import math


class SampleSize(object):
    """Frequentist sample size calculations.

    See: Duflo, E., Glennerster, R., & Kremer, M. (2007). Using
         randomization in development economics research: A toolkit.
         Handbook of Development Economics, 4, 3895–3962. pp28-31.

    Methods
    -------
    binomial()
        Calculate the required sample size for a binomial metric.
    binomial_interactive()
        Interactive version of the binomial() function for notebook use.
    continuous()
        Calculate the required sample size for a continuous metric.
    continuous_interactive()
        Interactive version of the continuous() function for notebook use.
    achieved_power()
        TODO: Calculate achieved power given reached sample size.

    """
    default_alpha = 0.05
    default_power = 0.85
    default_treatments = 2
    default_comparisons = 'control_vs_all'
    default_treatment_costs = None
    default_treatment_allocations = None
    default_bonferroni = False

    @staticmethod
    def continuous(average_absolute_mde,
                   baseline_variance,
                   alpha=default_alpha,
                   power=default_power,
                   treatments=default_treatments,
                   comparisons=default_comparisons,
                   treatment_costs=default_treatment_costs,
                   treatment_allocations=default_treatment_allocations,
                   bonferroni_correction=default_bonferroni):
        """Calculate the required sample size for a binomial metric.

        Args:
            average_absolute_mde (float): Average absolute minimal detectable
                effect size (mean difference) across all tests.
            baseline_variance (float): Baseline metric variance in
                target population.
            alpha (float, optional): Probability of Type I error
                (false positive). Defaults to 0.05.
            power (float, optional): 1 - B, where B is the probability of
                Type II error (false negative). Defaults to 0.85.
            treatments (int, optional): Number of treatment variants
                in the a/b test, including control. Defaults to 2.
            comparisons ({'control_vs_all', 'all_vs_all'}, optional): Which
                treatments to compare. Defaults to 'control_vs_all'.
            treatment_costs (numpy.ndarray, optional): Array with same length
                as the number of treatments containing positive floats
                specifying the treatments' relative costs. Defaults to equal
                cost for all treatments.
            treatment_allocations (numpy.ndarray, optional): Array with same
                length as the number of treatments containing proportion of
                sample allocated to each treatment. If not specified defaults
                to automatic allocation.
            bonferroni_correction (bool): Whether Bonferroni correction should
                be applied to control the false positive rate across all
                comparisons. Defaults to false.

        Returns:
            int: Total required sample size across all treatments.
            list of int: Required sample size for each treatment.
            list of float: Proportion of total sample allocated
                to each treatment.

        Raises:
            ValueError: If `power` is less than or equal to`alpha`.

        """
        mde = SampleSize._clean_continuous_mde(average_absolute_mde)
        baseline_variance = SampleSize._validate_positive(baseline_variance)

        return SampleSize._calculate_samplesize(mde,
                                                baseline_variance,
                                                alpha,
                                                power,
                                                treatments,
                                                comparisons,
                                                treatment_costs,
                                                treatment_allocations,
                                                bonferroni_correction)

    @staticmethod
    def continuous_interactive():
        SampleSize._calculate_sample_size_interactive('continuous')

    @staticmethod
    def binomial(absolute_percentage_mde,
                 baseline_proportion,
                 alpha=default_alpha,
                 power=default_power,
                 treatments=default_treatments,
                 comparisons=default_comparisons,
                 treatment_costs=default_treatment_costs,
                 treatment_allocations=default_treatment_allocations,
                 bonferroni_correction=default_bonferroni):
        """Calculate the required sample size for a binomial metric.

        Args:
            absolute_percentage_mde (float): Average absolute minimal
                detectable effect size across all tests.
            baseline_proportion (float): Baseline metric proportion in
                target population.
            alpha (float, optional): Probability of Type I error
                (false positive). Defaults to 0.05.
            power (float, optional): 1 - B, where B is the probability of
                Type II error (false negative). Defaults to 0.85.
            treatments (int, optional): Number of treatment variants
                in the a/b test, including control. Defaults to 2.
            comparisons ({'control_vs_all', 'all_vs_all'}, optional): Which
                treatments to compare. Defaults to 'control_vs_all'.
            treatment_costs (numpy.ndarray, optional): Array with same length
                as the number of treatments containing positive floats
                specifying the treatments' relative costs. Defaults to equal
                cost for all treatments.
            treatment_allocations (numpy.ndarray, optional): Array with same
                length as the number of treatments containing proportion of
                sample allocated to each treatment. If not specified defaults
                to automatic allocation.
            bonferroni_correction (bool): Whether Bonferroni correction should
                be applied to control the false positive rate across all
                comparisons. Defaults to false.

        Returns:
            int: Total required sample size across all treatments.
            list of int: Required sample size for each treatment.
            list of float: Proportion of total sample allocated
                to each treatment.

        Raises:
            ValueError: If `power` is less than or equal to`alpha`.
            ValueError: If `baseline_proportion` - `absolute_percentage_mde`
                < 0 and  `baseline_proportion` + `absolute_percentage_mde` > 1.
                I.e. if the mde always implies a non-valid percentage.

        """
        baseline = SampleSize._validate_percentage(baseline_proportion)
        mde = SampleSize._clean_binomial_mde(absolute_percentage_mde, baseline)
        baseline_variance = baseline * (1 - baseline)

        return SampleSize._calculate_samplesize(mde,
                                                baseline_variance,
                                                alpha,
                                                power,
                                                treatments,
                                                comparisons,
                                                treatment_costs,
                                                treatment_allocations,
                                                bonferroni_correction)

    @staticmethod
    def binomial_interactive():
        SampleSize._calculate_sample_size_interactive('binomial')

    @staticmethod
    def _calculate_samplesize(mde, baseline_variance, alpha, power, treatments,
                              comparisons, treatment_costs,
                              treatment_allocations, bonferroni):
        power = SampleSize._validate_percentage(power)
        treatments = SampleSize._clean_treatments(treatments)
        comparisons = SampleSize._clean_comparisons(comparisons)
        treatment_costs = SampleSize._clean_treatment_costs(treatments,
                                                            treatment_costs)

        alpha = SampleSize._get_alpha(alpha, power, bonferroni,
                                      treatments, comparisons)
        treatment_allocations = (
            SampleSize._get_treatment_allocations(treatments,
                                                  comparisons,
                                                  treatment_costs,
                                                  treatment_allocations))

        num_comparisons = SampleSize._num_comparisons(treatments, comparisons)
        comparison_matrix = SampleSize._get_comparison_matrix(treatments,
                                                              comparisons)

        z_alpha = st.norm.ppf(1 - alpha / 2)
        z_power = st.norm.ppf(power)

        a = np.power(1. / (num_comparisons * mde), 2)
        b = np.power(z_power + z_alpha, 2)
        c = baseline_variance
        d = 0
        for i in range(treatments):
            for j in range(treatments):
                if comparison_matrix[i, j] > 0:
                    d += np.sqrt(
                        1. / treatment_allocations[i] +
                        1. / treatment_allocations[j])
        d = np.power(d, 2)

        n_total = np.ceil(a * b * c * d).astype(int)
        n_allocation = np.ceil(treatment_allocations * n_total).astype(int)
        return n_total, n_allocation, treatment_allocations

    @staticmethod
    def _calculate_sample_size_interactive(metric):
        style = {'description_width': 'initial'}
        desc_layout = widgets.Layout(width='50%')
        if metric == 'continuous':
            mde_widget = widgets.FloatText(
                value=0.01,
                description='',
            )

            mde_desc = widgets.HTML("""
                <small>
                    This is the smallest absolute difference in averages that
                    any of your comparisons can detect at the given statistical
                    rigour.
                </small>
            """, layout=desc_layout)

            baseline_title = widgets.HTML("<strong>Baseline variance</strong>")
            baseline_widget = widgets.BoundedFloatText(
                value=1.0,
                min=0.00001,
                max=1000000000.0,
                description='',
            )
            baseline_desc = widgets.HTML("""
                    <small>
                        This is the expected variance of the metric among
                        users in your control group.
                    </small>
                """, layout=desc_layout)

        elif metric == 'binomial':
            mde_widget = widgets.FloatLogSlider(
                value=0.003,
                base=10,
                min=-4,
                max=np.log10(0.5),
                step=0.001,
                description='',
                readout_format='.4f'
            )

            mde_desc = widgets.HTML("""
                <small>
                    This is the smallest absolute difference (percentage
                    point / 100) that any of your comparisons can detect
                    at the given statistical rigour.
                </small>
            """, layout=desc_layout)

            baseline_title = widgets.HTML("<strong>Baseline "
                                          "proportion</strong>")
            baseline_widget = widgets.FloatSlider(
                value=0.5,
                min=0.00001,
                max=0.99999,
                step=0.01,
                description=''
            )
            baseline_desc = widgets.HTML("""
                    <small>
                        This is the expected value of the metric among
                        users in your control group.
                    </small>
                """, layout=desc_layout)

        else:
            raise ValueError('metric must be `continuous` or `binomial`')

        alpha_widget = widgets.FloatSlider(
            value=0.05,
            min=0.001,
            max=0.10,
            step=0.001,
            description=r'\(\alpha\)',
            readout_format='.3f')

        power_widget = widgets.FloatSlider(
            value=0.85,
            min=0.8,
            max=0.99,
            step=0.01,
            description=r'Power, \( 1-\beta\)')

        treatments_widget = widgets.IntSlider(
            value=2,
            min=2,
            max=20,
            step=1,
            description='Groups (including control)',
            style=style)

        comparisons_widget = widgets.RadioButtons(
            options=['Control vs. All', 'All vs. All'],
            value='Control vs. All',
            description='Groups to compare',
            style=style)

        control_group_widget = widgets.FloatLogSlider(
            value=1,
            step=0.1,
            base=10,
            min=0,
            max=4,
            description='Control group advantage',
            readout=False,
            style=style,
        )
        control_group_description = widgets.HTML("""
            <small>
                Sometime we want the control group to be bigger than what is
                strictly optimal. This can be either because we can collect
                samples quickly enough anyway or because we believe the
                treatment variants are riskier. Boosting the size of the
                control group comes at the cost of an increased total
                required sample.
            </small>
        """, layout=desc_layout)

        bonferroni_widget = widgets.Checkbox(
            value=False,
            description='Apply Bonferroni correction')

        risk_reset_btn = widgets.Button(
            description=' ',
            disabled=False,
            button_style='',
            tooltip='Reset variant risk',
            icon='repeat',
            layout=widgets.Layout(width="40px")
        )

        def reset_widget(b):
            control_group_widget.value = 1
        risk_reset_btn.on_click(reset_widget)

        ui = widgets.VBox([
            widgets.HTML('<h4>Target metric</h4>'),
            widgets.VBox(
                children=[
                    widgets.HTML("<strong>Minimal Detectable Effect "
                                 "size</strong>"),
                    mde_widget,
                    mde_desc],
            ),
            widgets.VBox(
                children=[
                    baseline_title,
                    baseline_widget,
                    baseline_desc],
            ),
            widgets.HTML('<h4>Statistical rigour</h4>'),
            alpha_widget,
            power_widget,
            bonferroni_widget,
            widgets.HTML('<h4>Treatment groups</h4>'),
            treatments_widget,
            comparisons_widget,
            widgets.VBox(
                children=[
                    widgets.HBox([control_group_widget, risk_reset_btn]),
                    control_group_description
                ]
            )])

        def show_samplesize(mde,
                            baseline,
                            alpha,
                            power,
                            treatments,
                            comparisons_readable,
                            bonferroni_correction,
                            relative_risk):
            if comparisons_readable == 'Control vs. All':
                comparisons = 'control_vs_all'
            else:
                comparisons = 'all_vs_all'

            treatment_costs = np.ones(treatments)
            treatment_costs[1:] = relative_risk
            treatment_allocations = None

            if metric == 'continuous':
                n_optimal, _, _ = SampleSize.continuous(
                    mde, baseline, alpha, power,
                    treatments, comparisons, None,
                    treatment_allocations, bonferroni_correction)
                n_tot, n_cell, prop_cell = SampleSize.continuous(
                    mde, baseline, alpha, power,
                    treatments, comparisons, treatment_costs,
                    treatment_allocations, bonferroni_correction)
                code_html = widgets.HTML(
                    "<pre><code>"
                    f"SampleSize.continuous(average_absolute_mde={ mde },\n"
                    f"                      baseline_variance={ baseline },\n"
                    f"                      alpha={ alpha },\n"
                    f"                      power={ power },\n"
                    f"                      treatments={ treatments },\n"
                    f"                      comparisons="
                    f"'{ comparisons }',\n"
                    f"                      treatment_costs="
                    f"{ list(treatment_costs) },\n"
                    f"                      treatment_allocations=None,\n"
                    f"                      bonferroni_correction="
                    f"{ bonferroni_correction })"
                    "<code></pre>")
            else:
                n_tot, n_cell, prop_cell = SampleSize.binomial(
                    mde, baseline, alpha, power,
                    treatments, comparisons, treatment_costs,
                    treatment_allocations, bonferroni_correction)
                n_optimal, _, _ = SampleSize.binomial(
                    mde, baseline, alpha, power,
                    treatments, comparisons, None,
                    treatment_allocations, bonferroni_correction)
                code_html = widgets.HTML(
                    "<pre><code>"
                    f"SampleSize.binomial(absolute_percentage_mde={ mde },\n"
                    f"                    baseline_proportion="
                    f"{ baseline },\n"
                    f"                    alpha={ alpha },\n"
                    f"                    power={ power },\n"
                    f"                    treatments={ treatments },\n"
                    f"                    comparisons="
                    f"'{ comparisons }',\n"
                    f"                    treatment_costs="
                    f"{ list(treatment_costs) },\n"
                    f"                    treatment_allocations=None,\n"
                    f"                    bonferroni_correction="
                    f"{ bonferroni_correction })"
                    "<code></pre>")

            def compare_against_optimal(current, optimal):
                if current == optimal:
                    return ''
                else:
                    return (f"<br><small><em>{current/optimal:.1f}x "
                            f"optimal group allocation of {optimal:,}."
                            f"</em></small>")

            display(widgets.HTML(
                f"<h4>Required sample size</h4>"
                f"<strong>Total:</strong><br>{n_tot:,}"
                f"{compare_against_optimal(n_tot, n_optimal)}"))
            cell_str = '<strong>Sample size in each cell</strong>'
            for i in range(len(n_cell)):
                if i == 0:
                    treatment = 'Control'
                else:
                    treatment = 'Variant ' + str(i)

                cell_str += (f"<br><em>{treatment}:</em> "
                             f"{n_cell[i]:,} ({prop_cell[i]*100:.1f}%)")

            display(widgets.HTML(cell_str))
            display(code_html)

        out = widgets.interactive_output(show_samplesize, {
            'mde': mde_widget,
            'baseline': baseline_widget,
            'alpha': alpha_widget,
            'power': power_widget,
            'treatments': treatments_widget,
            'comparisons_readable': comparisons_widget,
            'bonferroni_correction': bonferroni_widget,
            'relative_risk': control_group_widget
        })

        display(ui, out)

    @staticmethod
    def _clean_treatments(treatments):
        """Validate treatments input.

        Args:
            treatments (int): Number of treatment variants in the a/b test,
                including control. Defaults to 2.

        Returns:
            int: Number of treatment variants.

        Raises:
            TypeError: If `treatments` is not a number.
            ValueError: If `treatments` is not an integer greater than or
                equal to two.

        """
        error_string = 'Treatments must be a whole number ' \
                       'greater than or equal to two'
        try:
            remainder = treatments % 1
        except TypeError:
            raise TypeError(error_string)

        if remainder != 0:
            raise ValueError(error_string)
        elif treatments < 2:
            raise ValueError(error_string)
        else:
            return int(treatments)

    @staticmethod
    def _clean_comparisons(comparisons):
        """Validate comparisons input.

        Args:
            comparisons ({'control_vs_all', 'all_vs_all'}): Which treatments
                to compare.

        Returns:
            str: Which treatments to compare.

        Raises:
            ValueError: If `comparisons` is not one of 'control_vs_all' or
                'all_vs_all'.

        """
        if comparisons not in ('control_vs_all', 'all_vs_all'):
            raise ValueError('comparisons must be either '
                             '"control_vs_all" or "all_vs_all"')
        else:
            return comparisons

    @staticmethod
    def _num_comparisons(treatments, comparisons):
        """Calculate the number of hypothesis tests.

        When comparing all treatments against each other, calculating
        the number of hypothesis tests is an n-choose-k problem with
        n=treatments, and k=2: https://en.wikipedia.org/wiki/Combination.

        Args:
            treatments (int): Number of treatment variants in the a/b test,
                including control.
            comparisons ({'control_vs_all', 'all_vs_all'}): Which treatments
                to compare.

        Returns:
            int: Number of hypothesis tests to conduct.

        """
        treatments = SampleSize._clean_treatments(treatments)
        comparisons = SampleSize._clean_comparisons(comparisons)

        if comparisons == 'control_vs_all':
            num_comparisons = treatments - 1
        else:
            num_comparisons = math.factorial(treatments) /\
                              (2 * math.factorial(treatments - 2))

        return int(num_comparisons)

    @staticmethod
    def _get_comparison_matrix(treatments, comparisons):
        """Transform categorical comparison to matrix.

        Args:
            treatments (int): Number of treatment variants in the a/b test,
                including control.
            comparisons ({'control_vs_all', 'all_vs_all'}): Which treatments
                to compare.

        Returns:
            numpy.ndarray: Lower triangular matrix of size
                `treatments x treatments` with 1 in position i, j
                if treatment i is to be compared with treatment j.

        """
        treatments = SampleSize._clean_treatments(treatments)
        comparisons = SampleSize._clean_comparisons(comparisons)

        if comparisons == 'control_vs_all':
            comparison_matrix = np.zeros((treatments, treatments))
            comparison_matrix[1:, 0] = 1

        else:
            comparison_matrix = np.ones((treatments, treatments))
            comparison_matrix = np.tril(comparison_matrix, -1)

        return comparison_matrix

    @staticmethod
    def _clean_treatment_costs(treatments, treatment_costs):
        """Validate or generate treatment cost array.

        Args:
            treatment_costs (numpy.ndarray, None): Array with same length as
                the number of treatments containing positive floats specifying
                the treatments' relative costs. None also accepted in which
                case equal relative costs are returned.
            treatments (int): Number of treatment variants in the a/b test,
                including control.

        Returns:
            numpy.ndarray: Array with each treatment's cost.

        Raises:
            TypeError: If `treatment_costs` is not None or a numpy.ndarray.
            TypeError: If the length of customs `treatment_costs` is not the
                same as the number of treatments.
            ValueError: If the values of custom `treatment_costs` are not all
                positive and sum to one.

        """
        treatments = SampleSize._clean_treatments(treatments)

        if treatment_costs is None:
            # Default equal cost of all cells
            return np.ones(treatments)

        elif (not (isinstance(treatment_costs, np.ndarray) or
                   isinstance(treatment_costs, list)) or
              len(treatment_costs) != treatments):
            raise TypeError('treatment_costs must be a list or numpy array of'
                            'the same length as the number of treatments')

        try:
            treatment_costs = np.array(treatment_costs)
            if not (treatment_costs > 0).all():
                raise ValueError('treatment_costs values must all be positive')

        except TypeError:
            raise TypeError('treatment_costs array must only contain numbers')

        return treatment_costs

    @staticmethod
    def _get_treatment_allocations(treatments, comparisons, treatment_costs,
                                   treatment_allocations):
        """Validate or generate treatment allocation array.

        See the footnote on page 31 of "Duflo, E., Glennerster, R., & Kremer,
        M. (2007). Using randomization in development economics research: A
        toolkit. Handbook of Development Economics, 4, 3895–3962." for math.

        Args:
            treatments (int, optional): Number of treatment variants in the a/b
                test, including control. Defaults to 2.
            comparisons ({'control_vs_all', 'all_vs_all'}, optional): Which
                treatments to compare. Defaults to 'control_vs_all'.
            treatment_costs (numpy.ndarray, optional): Array with same length
                as the number of treatments containing positive floats
                specifying the treatments' relative costs. Defaults to equal
                cost for all treatments.
            treatment_allocations (numpy.ndarray/list/tuple, optional): Array
                with same length as the number of treatments containing
                proportion of sample allocated to each treatment. If not
                specified defaults to automatic allocation.

        Returns:
            numpy.ndarray: Array with same length as the number of treatments
                containing proportion of sample allocated to each treatment.

        Raises:
            TypeError: If `treatment_allocations` is not None or a
                numpy.ndarray.
            TypeError: If the length of custom `treatment_allocations` is not
                the same as the number of treatments.
            ValueError: If the values of custom `treatment_allocations` are
                not all positive and sum to one.

        """
        treatments = SampleSize._clean_treatments(treatments)

        if treatment_allocations is not None:
            if (isinstance(treatment_allocations, list) or
                    isinstance(treatment_allocations, tuple)):
                treatment_allocations = np.array(treatment_allocations)

            if (not isinstance(treatment_allocations, np.ndarray) or
                    len(treatment_allocations) != treatments):
                raise TypeError('treatment_allocations must be a numpy array '
                                'or list of the same length as the number of '
                                'treatments')

            elif not (treatment_allocations > 0).all():
                raise ValueError('treatment_allocations values '
                                 'must all be positive')

            elif not math.isclose(treatment_allocations.sum(), 1.0):
                raise ValueError('treatment_allocations values '
                                 'must sum to one')

            else:
                return np.array(treatment_allocations)

        comparisons = SampleSize._get_comparison_matrix(treatments,
                                                        comparisons)
        weighted_comparisons = comparisons/np.sum(comparisons)
        treatment_costs = SampleSize._clean_treatment_costs(treatments,
                                                            treatment_costs)

        ratios = np.zeros((treatments, treatments))
        for i in range(treatments):
            sum_importance_i = (np.sum(weighted_comparisons[:, i]) +
                                np.sum(weighted_comparisons[i, :]))
            for j in range(treatments):
                sum_importance_j = (np.sum(weighted_comparisons[:, j]) +
                                    np.sum(weighted_comparisons[j, :]))
                ratios[i, j] = (sum_importance_i / sum_importance_j *
                                np.sqrt(treatment_costs[j] /
                                        treatment_costs[i]))

        treatment_allocations = ratios[:, 0] / np.sum(ratios[:, 0])

        return treatment_allocations

    @staticmethod
    def _get_alpha(alpha, power, bonferroni, treatments, comparisons):
        """Validate and potentially correct false positive rate.

        Args:
            alpha (float): Probability of Type I error (false positive).
            bonferroni (bool): Whether Bonferroni correction should be applied
                to control the false positive rate across all comparisons.
            treatments (int): Number of treatment variants in the a/b test,
                including control.
            comparisons ({'control_vs_all', 'all_vs_all'}, optional): Which
                treatments to compare.

        Returns:
            float: False positive rate, potentially Bonferroni corrected.

        Raises:
            ValueError: If `power` is less than or equal to `alpha`.
            TypeError: If `bonferroni` is not a bool.

        """
        power = SampleSize._validate_percentage(power)
        alpha = SampleSize._validate_percentage(alpha)

        if power <= alpha:
            raise ValueError('alpha must be less than power')
        elif not isinstance(bonferroni, bool):
            raise TypeError('bonferroni must be a bool')

        num_comparisons = SampleSize._num_comparisons(treatments, comparisons)

        if bonferroni:
            return alpha / num_comparisons
        else:
            return alpha

    @staticmethod
    def _validate_percentage(num):
        """Validate that num is a percentage.

        Args:
            num(float): Valid percentage.

        Returns:
            float: Valid percentage.

        Raises:
            TypeError: If `num` is not a float.
            ValueError: If `num` is not between zero and one.

        """
        if not isinstance(num, float):
            raise TypeError('num must be a float')
        elif not 0 < num < 1:
            raise ValueError('num must be between 0 and 1')
        else:
            return num

    @staticmethod
    def _validate_positive(val):
        """Validate that val is positive.

        Args:
            val (float): Value to validate.

        Returns:
            float: Value.

        Raises:
            ValueError: If value is non-positive.

        """
        if not val > 0:
            raise ValueError('value must be positive')
        else:
            return val

    @staticmethod
    def _clean_continuous_mde(average_absolute_mde):
        """Validate that mde is not equal to zero.

        Args:
            average_absolute_mde (float): Average absolute minimal detectable
                effect size (mean difference) across all tests.

        Returns:
            float: Average absolute minimal detectable effect size.

        Raises:
            ValueError: If `average_absolute_mde` is zero.

        """
        if math.isclose(average_absolute_mde, 0.):
            raise ValueError('average_absolute_mde cannot be zero')
        else:
            return average_absolute_mde

    @staticmethod
    def _clean_binomial_mde(absolute_percentage_mde, baseline_proportion):
        """Validate that mde is percentage and not too large.

        Args:
            absolute_percentage_mde (float): Average absolute minimal
                detectable effect size across all tests.
            baseline_proportion (float): Baseline metric proportion in
                target population.

        Returns:
            float: Average absolute minimal detectable effect size.

        """
        mde = SampleSize._validate_percentage(absolute_percentage_mde)
        baseline = SampleSize._validate_percentage(baseline_proportion)

        if baseline - mde < 0 and baseline + mde > 1:
            raise ValueError('absolute_percentage_mde is too large '
                             'given baseline_proportion')
        else:
            return mde
