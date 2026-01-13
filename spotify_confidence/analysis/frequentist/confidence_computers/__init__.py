from spotify_confidence.analysis.constants import BOOTSTRAP, CHI2, TTEST, ZTEST, ZTESTLINREG
from spotify_confidence.analysis.frequentist.confidence_computers import (
    bootstrap_computer as bootstrap_computer,
)
from spotify_confidence.analysis.frequentist.confidence_computers import (
    chi_squared_computer as chi_squared_computer,
)
from spotify_confidence.analysis.frequentist.confidence_computers import (
    t_test_computer as t_test_computer,
)
from spotify_confidence.analysis.frequentist.confidence_computers import (
    z_test_computer as z_test_computers,
)
from spotify_confidence.analysis.frequentist.confidence_computers import (
    z_test_linreg_computer as z_test_linreg_computer,
)

confidence_computers = {
    CHI2: chi_squared_computer,
    TTEST: t_test_computer,
    ZTEST: z_test_computers,
    BOOTSTRAP: bootstrap_computer,
    ZTESTLINREG: z_test_linreg_computer,
}
