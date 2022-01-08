# import sys
# sys.path.append('..')

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from scipy.integrate import trapz
from scipy.stats import uniform

from climate_metrics import (
    AGWP_CO2,
    AGWP_CH4_no_CO2,
    AGTP_CO2,
    AGWP_N2O,
    AGTP_non_CO2,
    GWP,
    GTP
)


def test_AGWP_CO2():
    """
    References
    ----------
    IPCC, 2013. AR5, WG1, Chapter 8.  Appendix 8.A.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    assert np.isclose(AGWP_CO2(100)/9.17e-14, 1, atol=1e-02)
    assert np.isclose(AGWP_CO2(20)/2.49e-14, 1, atol=1e-02)


def test_AGWP_CH4_no_CO2():
    """
    References
    ----------
    IPCC, 2013. AR5, WG1, Chapter 8.  Appendix 8.A.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    assert np.isclose(AGWP_CH4_no_CO2(20)/2.09e-12, 1, atol=1e-03)
    assert np.isclose(AGWP_CH4_no_CO2(100)/2.61e-12, 1, atol=1e-03)


def test_AGWP_N2O():
    assert np.isclose(AGWP_N2O(20)/6.58e-12, 1, atol=1e-03)
    assert np.isclose(AGWP_N2O(100)/2.43e-11, 1, atol=1e-03)


def test_AGTP_CO2():
    """
    References
    ----------
    IPCC, 2013. AR5, WG1, Chapter 8.  Appendix 8.A.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    assert np.isclose(6.84e-16/AGTP_CO2(20), 1, atol=1e-2)
    assert np.isclose(6.17e-16/AGTP_CO2(50), 1, atol=1e-2)
    assert np.isclose(5.47e-16/AGTP_CO2(100), 1, atol=1e-2)


def test_AGTP_non_CO2():
    """
    References
    ----------
    IPCC, 2013. AR5, WG1, Chapter 8.  Appendix 8.A.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    assert np.isclose(4.62e-14/AGTP_non_CO2(20, 'ch4'), 1, atol=1e-2)
    assert np.isclose(2.34e-15/AGTP_non_CO2(100, 'ch4'), 1, atol=1e-2)
    assert np.isclose(1.89e-13/AGTP_non_CO2(20, 'n2o'), 1, atol=1e-2)
    assert np.isclose(1.28e-13/AGTP_non_CO2(100, 'n2o'), 1, atol=1e-2)


def test_dynamic_GWP():
    # initalize parameters
    time_horizon = 100
    time_step = 0.1
    emission = 1

    # First unit_pulse test.
    emission_year = 0
    expected_GWP_0, emission_pulse = compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step)
    actual_GWP_0 = GWP(
        time_horizon,
        # We need to scale the emission impulse so the emission
        # integral (emission * time_step)is 1.
        emission_pulse/time_step,
        'co2',
        time_step)

    np.testing.assert_array_almost_equal(expected_GWP_0, actual_GWP_0, decimal=4)

    # Second unit_pulse test
    emission_year = 25
    expected_GWP_25, emission_pulse = compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step)
    actual_GWP_25 = GWP(
        time_horizon,
        # We need to scale the emission impulse so the emission
        # integral (emission * time_step) is 1.
        emission_pulse/time_step,
        'co2',
        time_step)

    np.testing.assert_array_almost_equal(expected_GWP_25, actual_GWP_25, decimal=4)

    # Third test, continuous emission pulse
    t = np.arange(0, time_horizon+time_step, time_step)
    emissions = uniform.pdf(t, scale=time_horizon)
    actual_result = GWP(
            time_horizon,
            emissions,  # see comment above
            'co2',
            time_step)

    # The sifting property leads to an equivalence between a convolution
    # of a function (f) with a shifted function (g) to time tau, and a shifted
    # function:  f(t)*g(t-tau) = f(t-tau).
    # If we treat each annual emission (g) in year tau, AGWP(t)*emission(t-tau)
    # can be represented as AGWP(100-tau).  For emissions at year 0 we apply
    # AGWP(100), emissions at year 1 we apply AGWP(99), etc.
    expected_result = \
        trapz(emissions*np.flip(AGWP_CO2(t)), dx=time_step) / AGWP_CO2(100)
    np.testing.assert_array_almost_equal(expected_result, actual_result, decimal=3)


def compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step):
    """
    Dynamic GWP == GWP(time_horizon - emission_year)
    so we can validate the implementation with an alternative
    calculation."""
    total_steps = int(time_horizon/time_step) + 1

    emission_index = int(emission_year/time_step)
    emission_pulse = np.zeros(total_steps)
    emission_pulse[emission_index] = emission
    expected_GWP = AGWP_CO2(time_horizon - emission_year)/AGWP_CO2(time_horizon)

    return expected_GWP, emission_pulse

# Emission vectors for testing
C_EMISSIONS = 1
STEP_SIZE_1 = 1
TIME_HORIZON = 100
ONE_ZERO_100 = np.zeros(TIME_HORIZON)
ONE_ZERO_100[0] = C_EMISSIONS
ONE_FIFTY_100 = np.zeros(TIME_HORIZON)
ONE_FIFTY_100[50] = C_EMISSIONS
# A step size of 0.1
ONE_FIFTY_1000 = np.zeros(int(TIME_HORIZON/0.1))
ONE_FIFTY_1000[int(50/0.1)] = C_EMISSIONS

# time_horizon, emissions, emission_type, step_size
GWP_tests = [
    ((100, 1, 'co2', 1), 1),
    # Scale emissions by step_size so integral of emissions == 1
    ((100, 1/0.1, 'co2', 0.1), 1),
    ((100, ONE_ZERO_100, 'CO2', 1), 1),
    ((100, 1, 'ch4', 1), 28.40146),
    # Scale emissions by step_size so integral of emissions == 1
    ((100, 1/0.1, 'ch4', 0.1), 28.40146),
    ((100, ONE_ZERO_100, 'ch4', 1), 28.40146),
    ((100, ONE_FIFTY_100, 'co2', 1), 0.57808),
    # Scale emissions by step_size so integral of emissions == 1
    ((100, ONE_FIFTY_1000/0.1, 'co2', 0.1), 0.57808),
    ((100, ONE_FIFTY_100, 'ch4', 1), 27.90656),
    ]


@pytest.mark.parametrize("test_input, expected", GWP_tests)
def test_GWP(test_input, expected):
    assert_array_almost_equal(GWP(*test_input), expected, decimal=3)


GTP_tests1 = [
    ((100, ONE_FIFTY_100, 'co2'), 1.1276),
    ((100, ONE_FIFTY_100, 'ch4'), 15.8396),
    ]


@pytest.mark.parametrize("test_input, expected", GTP_tests1)
def test_GTP(test_input, expected):
    assert_array_almost_equal(GTP(*test_input), expected, decimal=3)


GTP_tests2 = [
    ((20, 1, 'co2'), 1),
    ((20, 1, 'CH4'), 67),
    ((20, 1, 'n2o'), 277),
    ((50, 1, 'co2'), 1),
    ((50, 1, 'CH4'), 14),
    ((50, 1, 'n2o'), 282),
    ((100, 1, 'co2'), 1),
    ((100, 1, 'CH4'), 4),
    ((100, 1, 'n2o'), 234),
    ((100, ONE_ZERO_100, 'CO2'), 1),
    ((100, ONE_ZERO_100, 'ch4'), 4),
    ((100, ONE_ZERO_100, 'n2o'), 234),
    ]


@pytest.mark.parametrize("test_input, expected", GTP_tests2)
def test_GTP_no_decimal(test_input, expected):
    """IPCC values are rounded to the nearest whole number"""
    assert_array_almost_equal(GTP(*test_input), expected, decimal=0)
