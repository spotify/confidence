Spotify Confidence
========

![Status](https://img.shields.io/badge/Status-Beta-blue.svg)
![Latest release](https://img.shields.io/badge/release-2.6.3-green.svg "Latest release: 2.6.3")
![Python](https://img.shields.io/badge/Python-3.6-blue.svg "Python")
![Python](https://img.shields.io/badge/Python-3.7-blue.svg "Python")

Python library for AB test analysis.

Why use Spotify Confidence?
-----------------

Spotify Confidence provides convinience wrappers around statsmodel's various functions for computing p-values and confidence intervalls. 
With Spotify Confidence it's easy to compute several p-values and confidence bounds in one go, e.g. one for each country or for each date. 
Each function comes in two versions: 
 - one that return a pandas dataframe,
 - one that returns a [Chartify](https://github.com/spotify/chartify) chart.

Spotify Confidence has support calculating p-values and confidence intervals using Z-statistics, Student's T-statistics 
(or more exactly [Welch's T-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)), as well as Chi-squared statistics.

There is also a Bayesian alternative in the BetaBinomial class.

Examples
--------
```
import spotify_confidence as confidence
import pandas as pd

data = pd.DataFrame(
    {'variation_name': ['treatment1', 'control', 'treatment2', 'treatment3'],
     'success': [50, 40, 10, 20],
     'total': [100, 100, 50, 60]
    }
)

test = confidence.ZTest(
    data,
    numerator_column='success',
    numerator_sum_squares_column=None,
    denominator_column='total',
    categorical_group_columns='variation_name',
    correction_method='bonferroni')
    
test.summary()
test.difference(level_1='control', level_2='treatment1')
test.multiple_difference(level='control', level_as_reference=True)

test.summary_plot().show()
test.difference_plot(level_1='control', level_2='treatment1').show()
test.multiple_difference_plot(level='control', level_as_reference=True).show()
```

See jupyter notebooks in `examples` folder for more complete examples.

Installation
------------
Spotify Confidence can be installed via pip:

```pip install spotify-confidence```

[Find the latest release version here](https://github.com/spotify/confidence/releases)

### Code of Conduct

This project adheres to the [Open Code of Conduct](https://github.com/spotify/code-of-conduct/blob/master/code-of-conduct.md) By participating, you are expected to honor this code.

