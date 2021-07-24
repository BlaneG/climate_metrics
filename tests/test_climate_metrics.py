# import sys
# sys.path.append('..')

import numpy as np
from scipy.integrate import trapz
from scipy.stats import uniform

from climate_metrics import (
    AGWP_CO2,
    AGWP_CH4_no_CO2,
    dynamic_GWP,
    AGTP_CO2,
    AGWP_N2O,
    AGTP_non_CO2,
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
    actual_GWP_0 = dynamic_GWP(
        time_horizon,
        emission_pulse,
        'co2',
        time_step,
        is_unit_impulse=True)
    result_0_isclose = np.isclose(expected_GWP_0, actual_GWP_0, rtol=1e-4)
    assert(result_0_isclose)

    # Second unit_pulse test
    emission_year = 25
    expected_GWP_25, emission_pulse = compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step)
    actual_GWP_25 = dynamic_GWP(
        time_horizon,
        emission_pulse,
        'co2',
        time_step,
        is_unit_impulse=True)
    result_25_isclose = np.isclose(expected_GWP_25, actual_GWP_25, rtol=1e-4)
    assert(result_25_isclose)

    # Third test, continuous emission pulse
    t = np.arange(0, time_horizon+time_step, time_step)
    emissions = uniform.pdf(t, scale=time_horizon)
    actual_result = dynamic_GWP(
            time_horizon,
            emissions,
            'co2',
            time_step,
            is_unit_impulse=False)

    # The sifting property leads to an equivalence between a convolution
    # of a function (f) with a shifted function (g) to time tau, and a shifted
    # function:  f(t)*g(t-tau) = f(t-tau)
    # If we treat each annual emission (g) in year tau, AGWP(t)*emission(t-tau)
    # can be represented as AGWP(100-tau).  For emissions at year 0 we apply
    # AGWP(100), emissions at year 1 we apply AGWP(99), etc.
    expected_result = trapz(emissions*np.flip(AGWP_CO2(t)), dx=0.1)/AGWP_CO2(100)
    result_is_close = np.isclose(actual_result, expected_result, rtol=1e-3)
    assert(result_is_close)
    assert(np.isclose(actual_result, 0.5, atol=1e-1))


def compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step):
    """
    Dynamic GWP is just the GWP(time-horizon - emission_year)
    so we can validate the implementation with an alternative
    calculation."""
    total_steps = int(time_horizon/time_step) + 1

    emission_index = int(emission_year/time_step)
    emission_pulse = np.zeros(total_steps)
    emission_pulse[emission_index] = emission
    expected_GWP = AGWP_CO2(time_horizon - emission_year)/AGWP_CO2(time_horizon)

    return expected_GWP, emission_pulse
