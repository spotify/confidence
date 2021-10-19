=======
History
=======

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
