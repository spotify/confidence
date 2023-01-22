=======
History
=======

2.7.7 (2023-01-21)
------------------
* Fixed bug that led to unexpected behaviour when using non_inferiority_margins=False. Not passing False produces the same result as passing None
* Fixed bug in chartify grapher that caused a crash when attempting to plot a mix of metrics where only some had non-inferiority margins

2.7.6 (2022-11-22)
------------------
* Fixed bug in compute_sequential_adjusted_alpha where we since 2.7.6 were taking the max sample size rowwise

2.7.5 (2022-11-15)
------------------
* Major refactoring, splitting the code into more files for improved readability
* Fixed bugs related to group sequential testing that resulted in to narrow confidence bands for tests with multiple treatment groups
* Bump Chartify version
* Minor changes to get rid of warnings

2.7.4 (2022-08-31)
------------------
* Fixed bug in sample size calculator check for binary metrics when there are nans

2.7.3 (2022-08-26)
------------------

* Fixed bug in SampleSizeCalculator.optimal_weights_and_sample_size.
* Added check to make point estimates and variances match for binary metrics when using the sample size calculator.

2.7.2 (2022-08-08)
------------------

* Adding a constant to the variance reduction point estimate so that the adjusted value is close in magnitude to the origina value.

2.7.1 (2022-08-08)
------------------

* Bug fix for calculating relative NIMs/MDEs with variance reduction.

2.7.0 (2022-05-17)
------------------

* Added split_plots_by_group argument to difference plot methods to enable less cluttered plots. See example notebooks for example usage.

2.6.7 (2022-04-22)
------------------

* Updated example notebooks
* Updated dev requirements
* Minor changes to get rid of deprecation warnings

2.6.6 (2022-03-25)
------------------

* Require jinja to <3.1.0 because of incompatibility with bokeh

2.6.5 (2022-03-25)
------------------

* Fixed bug in chartify_grapher that led to incorrect CIs when the relative difference between the lower or upper bounds were more than two orders of magnitude apart.

2.6.4 (2022-03-17)
------------------

* Enabling sequential testing also for variance reduction

2.6.3 (2022-03-09)
------------------

* Added more complete support for variance reduction

2.6.2 (2022-02-16)
------------------

* Fixed bug that led to incorrect title of difference plots
* Fixed bug that led to too many comparisons for certain combinations of `levels` passed to the differences method

2.6.1 (2022-01-21)
------------------

* Fixed bug that led to crash when a segment had zero observations

2.6.0 (2021-12-21)
------------------

* Added support for variance reduction using linear regression

2.5.0 (2021-12-16)
------------------

* Added SampleSizeCalculator class to allow for more complex sample size calculations involving several metrics and dimensions

2.4.3 (2021-12-06)
------------------

* Improve performance by using more optimal pandas operations and by running computations in paralell over groupby dimensions.

2.4.2 (2021-11-24)
------------------

* Bugfix: When you had some metrics with NIMs and some with MDEs, one of them overrid the other
* Switching to using N-1 instead of N in denominator of variance estimate to get unbiased estimator for smaller sample sizes. For binary metrics we still use the old formula, equivalent to p*(1-p).
* Removing powered_effect_metric column because it's identical to powered_effect after bugfix in 2.4.1
* Minor performace and robustness tweaks to sequential bounds solver.

2.4.1 (2021-11-12)
------------------

* Bugfix: The field "powered_effect_for_metric" in the output of the difference methods (when verbose=True) was computed using incorrect current_number_of_units

2.4.0 (2021-11-08)
------------------

* Introduce Experiment class for mixed method testing: the method_column argument specifies which column in the input dataframe that contain method names, including "z-test", "t-test" etc.
* Difference methods now return required sample sizes and powered effects when verbose=True and metric_column, treatment_column and power are passed into the constructor, and minimum_detectable_effects_column is passed to the difference method.
* Support for bootstrap. Pass in a "bootstrap_samples_column" to specify which column that contain bootstrap point estimates and get CIs using the usual difference methods.
* For confidence developers: you can (and should!) now use `make black` to auto-format the code before committing.

2.3.7 (2021-10-22)
------------------

* Added differences method that takes a list of tuples of pairs of levels to compare, so that one can for example do all-vs-all comparisons.

2.3.6 (2021-10-20)
------------------

* Use alpha column rather than 1 - interval_size in sequential tests to allow for different alphas for different dimensions

2.3.5 (2021-10-19)
------------------

* Keep initial preference even when it it is not used in tests

2.3.4 (2021-10-19)
------------------

* Changed CI behaviour of some correction methods

2.3.3 (2021-10-19)
------------------

* Added some more multiple correction strategies that use two sided CIs when one sided are not available

2.3.2 (2021-10-19)
------------------

* Minor bug-fix: No need to raise error when using alternative correction methods, since we return None CIs and print out warning instead

2.3.1 (2021-10-19)
------------------

* Added even more multiple correction strategies

2.3.0 (2021-10-18)
------------------

* Added additional multiple correction methods (Holm, Hommel, Simes-Hochberg) for one sided tests.
* Added verbose mode to summary and difference methods that returns all intermediate columns that are used in the computations

2.2.0 (2021-09-28)
------------------

* Changed how sequential tests are done. Now, instead of passing a single (number) final_expected_sample_size, you pass a column name final_expected_sample_size. This is to ensure that groupby works as expected, i.e. different groups can have different expected sample sizes.

2.1.4 (2021-09-28)
------------------

* Added support for NaNs in NIMs

2.1.3 (2021-09-28)
------------------

* Added option to pass non_inferiority_margins=True which then uses NIMs in source data_frame, rather than passing in dict of tuples with NIMs

2.1.2 (2021-09-02)
------------------

* Fixed typos in README

2.1.1 (2021-04-21)
------------------

* Fixed broken setup.cfg that led to pip install spotify-confidence not installing correctly

2.1.0 (2021-04-21)
------------------

* Introduced sequential testing - see https://cran.r-project.org/web/packages/ldbounds/index.html for details.

2.0.2 (2021-04-15)
------------------

* Introduce frequentist test superclass to reduce code duplication
* Change default of level_as_reference to None to require more explicit intents

2.0.1 (2021-03-10)
------------------

* Remove internal dependencies and references to prepare to open source

2.0.0 (2021-03-09)
------------------

* Cleaned up inconsistencies, e.g. now using only computations from statsmodels, actually using T-statistics for computing t-test CIs (rather than Z-statistic). This changes CI slightly for small sample sizes.
* Not using Delta correction for relative CIs anymore, i.e. CI computations are done on the abbsolute scale, and then optionally displayed on relative scale. This is a more consistent approach, since Delta corrections were only applied to CIs and not to p-values. This will give more narrow CIs, with the difference being similar in magnitude to the relative change of the point estimates.
* Added ZTestComputer that relies on statsmodels rather than internal libraries to facilitate open-sourcing
* Added open source licence, copyright headers and code of conduct

1.6.3 (2021-01-13)
------------------

* Small fix to make it compatible with pandas 1.2

1.6.3 (2020-10-27)
------------------

* Fixed bug that led to NIMs sometimes being outside chart axis range

1.6.1 (2020-10-13)
------------------

* NIMs are now shown as green if significant else red

1.6.0 (2020-10-09)
------------------

* Added support for non-inferiority margins and one sided tests
* Added support for plotting adjusted CIs
* Fixed a bug that caused alpha to always be 0.99 for ZTest CIs in version 1.5.x

1.5.1 (2020-09-25)
------------------

* Made sure adjusted p-value is never larger than 1

1.5.0 (2020-09-01)
------------------

* Added ZTest class
* Refactored code to make it easier to maintain and more performant

1.4.2 (2020-08-04)
------------------

* Updated requirements of pandas and chartify
* Made it possible to pass in custom allocations as lists or tuples
* Increased the max value of sample size calculator's baseline variance input

1.4.1 (2019-04-08)
------------------

* Fixed titles and axis labels in ordinal difference plots
* Improved axis formatting

1.4.0 (2019-04-05)
------------------

* Added sample ratio test to the frequentist test objects.
* Added achived power to the frequentist test objects.
* Added sample size calculator.
* Made significance level/interval_size configurable
* Fixed formatting to not always show %
* Now requires Python 3.6+ for the use of f-strings.


1.3.1 (2019-03-14)
------------------

* Added ordinal multiple difference plot
* Some more refactoring, moving code to base to reduce duplication

1.3.0 (2019-03-06)
------------------

* Refactored to reduce code duplication and make it easier to add new functionality
* Minor breaking change: Changed names of output columns from "probability" to "point estimate"
  and from "probability_ci_lower/upper" to just "ci_lower/upper"

1.2.2 (2019-01-09)
------------------

* Fixed bug that caused p-value > 1 for positive mean differences

1.2.1 (2018-11-23)
------------------

* Fixed bug that breaks ChiSquared.multiple_difference_plot
  when level_as_reference=True. Thanks for finding @lagerroth!


1.2.0 (2018-11-07)
------------------

* .multiple_difference now always performs pairwise comparisons.
* Added Bayesian multiple_difference_joint methods for joint comparisons.
* Difference data frames now always return consistent column names.
* Add level_as_reference to multiple_difference methods to
  provide control over the order of the difference.
* Added as_cumulative method to create models based on
  a cumulative representation of the data.
* Added CI/CD to the library.
