"""
This module implements methods for computing climate metrics as described
by the International Governmental Panel on Climate Change."""


import numpy as np


#######################
# Physical models and parameters
# These are the building blocks for GWP and GTP
#######################
# W m–2 ppbv-1
RADIATIVE_EFFICIENCY_ppbv = {"co2": 1.37e-5, "ch4": 3.63e-4, "n2o": 3.00e-3}
COEFFICIENT_WEIGHTS = np.array([0.2173, 0.2240, 0.2824, 0.2763])
TIME_SCALES = np.array([394.4, 36.54, 4.304])


def _get_GHG_lifetime(GHG):
    ghg_lifetimes = dict(
        ch4=12.4,
        n2o=121
    )
    return ghg_lifetimes[GHG]


def _ppbv_to_kg_conversion(GHG):
    """
    Convert the radiative efficiency from ppbv normalization to kg normalization.

    References
    --------------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. p. 8SM-15.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    # kg per kmol
    molecular_weight = {"co2": 44.01, "ch4": 16.04, "n2o": 44.013}

    total_mass_atmosphere = 5.1352e18  # kg
    mean_molecular_weight_air = 28.97  # kg per kmol
    molecular_weight_ghg = molecular_weight[GHG]
    mass_ratio = mean_molecular_weight_air/molecular_weight_ghg
    return mass_ratio * (1e9/total_mass_atmosphere)


def _get_radiative_efficiency_kg(GHG):
    """Get the radiative efficiency of a GHG in W m–2 kg–1.
    """
    ppv_to_kg = _ppbv_to_kg_conversion(GHG)
    return ppv_to_kg * RADIATIVE_EFFICIENCY_ppbv[GHG]


def CO2_irf(time_horizon):
    """The impulse response function of CO2.

    Parameters
    -----------
    time_horizon : int
        The time since the original CO2 emission occurred.

    References
    --------------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. Equation 8.SM.10
    https://www.ipcc.ch/report/ar5/wg1/
    """

    exponential_1 = np.exp(-time_horizon/TIME_SCALES[0])
    exponential_2 = np.exp(-time_horizon/TIME_SCALES[1])
    exponential_3 = np.exp(-time_horizon/TIME_SCALES[2])

    return (
        COEFFICIENT_WEIGHTS[0]
        + COEFFICIENT_WEIGHTS[1]*exponential_1
        + COEFFICIENT_WEIGHTS[2]*exponential_2
        + COEFFICIENT_WEIGHTS[3]*exponential_3
    )


def impulse_response_function(t, GHG):
    """The impulse response function for non-CO2/CH4 GHGs.

    References
    -----------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. Equation 8.SM.8.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    life_time = {"ch4": 12.4, "n2o": 121}
    if GHG.lower() == "co2":
        return CO2_irf(t)
    else:

        return np.exp(-t/life_time[GHG.lower()])


def radiative_forcing_per_kg(t, GHG):
    """Computes the radiative forcing at time `t` for a GHG emission at time 0.

    Parameters
    --------------
    t : array_like
        Time at which radiative forcing is computed.
    """
    GHG = GHG.lower()

    if GHG == 'co2':
        radiative_efficiency = _get_radiative_efficiency_kg(GHG)

    elif GHG == 'ch4':
        radiative_efficiency = _get_radiative_efficiency_kg(GHG)
        radiative_efficiency *= _scaled_radiative_efficiency_from_O3_and_H2O()

    elif GHG == 'n2o':
        radiative_efficiency = _N2O_radiative_efficiency_after_methane_adjustment()

    return radiative_efficiency * impulse_response_function(t, GHG)


def radiative_forcing_from_emissions_scenario(
        time_horizon,
        emissions,
        GHG,
        step_size,
        mode='full'):
    """
    Parameters
    ---------------
    time_horizon : int
        Time period over which radiative forcing is computed for `emissions`.
    emissions : array_like
        GHG emissions (kg) at each time step.
    GHG : str
    step_size : float
        step_size for emissions to create a time index (`t`)
    mode : {'full', 'valid'}, optional
        mode passed to np.convolve.
        'full':
            Use full to get the temporal change in radiative forcing
            from 0-`time_horizon`.
            Output shape is `len(time_horizon) + len(emissions) + 1`.
        'same':
            Use to get the radiative forcing at `time_horizon`.
            Output shape is `max(len(time_horizon), len(emissions))`.

    Returns
    ---------------
    ndarray
    """
    t = np.arange(0, time_horizon+step_size, step_size)
    assert len(t) >= len(emissions)
    rf = radiative_forcing_per_kg(t, GHG)
    steps = int(time_horizon/step_size)
    return _convolve_metric(steps, emissions, rf, mode)


def _convolve_metric(steps, emissions, metric, mode):
    if mode == 'full':
        return np.convolve(emissions, metric, mode=mode)[0:steps+1]
    elif mode == 'valid':
        return np.convolve(emissions, metric, mode=mode)
    else:
        raise ValueError(f'Received invalid mode value: {mode}')


###############################
# GWP implementation
###############################

def AGWP_CO2(t):
    radiative_efficiency = _get_radiative_efficiency_kg("co2")
    exponential_1 = 1 - np.exp(-t/TIME_SCALES[0])
    exponential_2 = 1 - np.exp(-t/TIME_SCALES[1])
    exponential_3 = 1 - np.exp(-t/TIME_SCALES[2])
    cumulative_concentration = (
        COEFFICIENT_WEIGHTS[0]*t
        + COEFFICIENT_WEIGHTS[1]*TIME_SCALES[0]*exponential_1
        + COEFFICIENT_WEIGHTS[2]*TIME_SCALES[1]*exponential_2
        + COEFFICIENT_WEIGHTS[3]*TIME_SCALES[2]*exponential_3
        )

    return radiative_efficiency * cumulative_concentration


def _scaled_radiative_efficiency_from_O3_and_H2O():
    indirect_O3 = 0.5
    indirect_H2O = 0.15
    return 1 + indirect_O3 + indirect_H2O


def AGWP_CH4_no_CO2(t):
    """
    Parameters
    -----------
    t : int

    Note
    ------
    Does not include indirect effects from CO2 as a result of CH4 conversion to CO2.
    """
    radiative_efficiency = _get_radiative_efficiency_kg("ch4")
    methane_adjustments = _scaled_radiative_efficiency_from_O3_and_H2O()

    return (
        radiative_efficiency
        * methane_adjustments
        * _get_GHG_lifetime('ch4')
        * (1 - impulse_response_function(t, 'ch4'))
    )


def _N2O_radiative_efficiency_after_methane_adjustment():
    indirect_effect_of_N2O_on_CH4 = 0.36
    methane_adjustments = _scaled_radiative_efficiency_from_O3_and_H2O()
    radiative_efficiency_CH4_ppbv = RADIATIVE_EFFICIENCY_ppbv['ch4']
    radiative_efficiency_N2O_ppbv = RADIATIVE_EFFICIENCY_ppbv['n2o']
    radiative_efficiency_methane_adjustment = (
        indirect_effect_of_N2O_on_CH4
        * methane_adjustments
        * (radiative_efficiency_CH4_ppbv / radiative_efficiency_N2O_ppbv)
    )
    radiative_efficiency_N2O = _get_radiative_efficiency_kg("n2o")

    net_radiative_efficiency = (
        radiative_efficiency_N2O
        * (1 - radiative_efficiency_methane_adjustment)
    )
    return net_radiative_efficiency


def AGWP_N2O(t):
    net_radiative_efficiency = _N2O_radiative_efficiency_after_methane_adjustment()
    lifetime_N2O = _get_GHG_lifetime('n2o')
    irf_N2O = impulse_response_function(t, 'n2o')

    return (
        net_radiative_efficiency
        * lifetime_N2O
        * (1 - irf_N2O)
    )


def AGWP(t, GHG):
    if GHG.lower() == 'co2':
        return AGWP_CO2(t)
    elif GHG.lower() == 'ch4':
        return AGWP_CH4_no_CO2(t)
    elif GHG.lower() == 'n2o':
        return AGWP_N2O(t)
    else:
        raise NotImplementedError(f'AGWP methods have not been implemented for {GHG}')


def cumulative_radiative_forcing(
        time_horizon,
        emissions,
        GHG,
        step_size,
        annual=False):
    """Computes the cumulative radiative forcing in reponse to an emission scenario.

    This is a wrapper around _dynamic_AGWP providing a more user-friendly name.

    When `emissions` is a single value, instead of a temporal emission scenario,
    `temperature_response` returns the same result as `AGWP`.

    Parameters
    ------------------
    time_horizon : int
        The time at which the temperature response is computed.
    emissions : ndarray
        Emissions in kg of a `GHG`.
    GHG : str
    step_size : float or int
        The step size used to generate the time axis.

    """
    if annual:
        _dynamic_AGWP(time_horizon, emissions, GHG, step_size, mode='full')
    else:
        _dynamic_AGWP(time_horizon, emissions, GHG, step_size, mode='full')[0]


def _dynamic_AGWP(time_horizon, net_emissions, GHG, step_size, mode='full'):
    """
    """
    return _dynamic_absolute_climate_metric_template(
        'AGWP',
        time_horizon,
        net_emissions,
        GHG,
        step_size,
        mode
        )


def GWP(time_horizon,
        emissions,
        GHG,
        step_size=1,
        annual=False):
    """Computes the CO2 equivalent radiative forcing of net_emissions.

    Can be used to compute GWP over `time_horizon` of a single pulse emission,
    or a flow of emissions over time (referred to variously in the literature
    as tonne-year [1]_, [2]_, [3]_, GWP [4]_, and GWP_bio [5]_).

    Parameters
    ---------------
    time_horizon : int
    emissions : int or ndarray
        If emissions is an int, the emission is assumed to 
            occur at time=0.
    GHG : str {'CO2', 'CH4', 'N2O'}, optional
        Type of GHG emission in `emissions`.
    step_size : float or int
        Step size of `emissions` in years.
    annual : bool
        If `True`, returns annual GWP over the `time_horizon`. If `False`,
            returns the single value at `time_horizon`.


    Notes
    ------------
    If step_size < 1, the sum of the net_emissions vector must be equal
    to total emissions. So if a probability density function were used
    to simulate net_emissions over time, net_emissions would first have
    to be weighted by step_size before being passed to this function.

    Global Warming Potential is defined as the cumulative radiative forcing
    of :math:`GHG_x` emitted in year = 0 over a given time-horizon
    (:math:`t`):

    .. math:
        GWP(t) = \\frac{cumulativeRadiativeForcingGHG\\_x(t)}
                    {cumulativeRadiativeForcing\\_CO2(t)}

    Dynamic GWP ([1]_, [2]_ [3]_, [4]_) computes the cumulative radiative forcing
    of annual (:math:`t'`) emissions ():math:`GHG_x`) over a give time-horizon
    (:math:`t`):

    .. math:
        dynamicGWP_x(t, t')
                    = {\\mathbf{emission_x}(t')}\\cdot{\\mathbf{GWP_x}(t-t')}
                    = \\sum_{t'}{\\mathbf{emission_x}(t'){\\mathbf{GWP_x}(t-t')}}
                    = \frac{
                    \\sum_{t'}{cumulativeRadiativeForcingGHG_x(t-t')}}
                    {cumulativeRadiativeForcing_{CO2}(t)}


    References
    --------------

    .. [1] IPCC, 2000.  https://archive.ipcc.ch/ipccreports/sres/land_use/index.php?idp=74  # noqa: E501
    .. [2] Fearnside et al. 2000.  https://link.springer.com/article/10.1023/A:1009625122628  # noqa: E501
    .. [3] Moura Costa et al. 2000.  https://link.springer.com/article/10.1023/A:1009697625521  # noqa: E501
    .. [4] Levassuer et al. 2010.  https://pubs.acs.org/doi/10.1021/es9030003
    .. [5] Cherubini et al. 2011.  https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1757-1707.2011.01102.x  # noqa: E501


    """
    return _climate_metric_template(
        'GWP',
        time_horizon,
        emissions,
        GHG,
        step_size,
        annual)


###############################
# GTP implementation
###############################

# Short-term and long-term temperature response
# (Kelvin per (Watt per m2)) to radiative forcing
TEMPERATURE_RESPONSE_COEFFICIENTS = [0.631, 0.429]
# Temporal scaling factors (years)
TEMPORAL_WEIGHTS = [8.4, 409.5]


def AGTP_CO2(t):
    """

    References
    ------------
    1. 8.SM.15 in https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
    """
    radiative_efficiency = _get_radiative_efficiency_kg("co2")

    temperature_response = 0
    for j in range(2):
        short_term_temperature_response = COEFFICIENT_WEIGHTS[0] \
            * TEMPERATURE_RESPONSE_COEFFICIENTS[j]
        temporal_weight_1 = np.exp(-t/TEMPORAL_WEIGHTS[j])
        weighted_short_term_temperature_response = short_term_temperature_response \
            * (1 - temporal_weight_1)

        weighted_long_term_temperature_response = 0
        for i in range(3):
            temporal_weight_2_linear = TIME_SCALES[i] \
                / (TIME_SCALES[i] - TEMPORAL_WEIGHTS[j])
            long_term_temperature_response = COEFFICIENT_WEIGHTS[i+1] \
                * TEMPERATURE_RESPONSE_COEFFICIENTS[j]
            long_term_temperature_response = long_term_temperature_response \
                * temporal_weight_2_linear
            temporal_weight_2_exponential = np.exp(-t/TIME_SCALES[i])
            weighted_long_term_temperature_response += (
                long_term_temperature_response
                * (temporal_weight_2_exponential - temporal_weight_1)
            )

        temperature_response += (
            weighted_short_term_temperature_response
            + weighted_long_term_temperature_response
        )
    return radiative_efficiency * temperature_response


def AGTP_non_CO2(t, GHG):
    GHG = GHG.lower()
    radiative_efficiency = _get_radiative_efficiency_kg(GHG)
    ghg_lifetime = _get_GHG_lifetime(GHG)

    temperature_response = 0
    for i in range(2):
        temporal_weight_linear = ghg_lifetime / (ghg_lifetime - TEMPORAL_WEIGHTS[i])
        temperature_response_coefficient = TEMPERATURE_RESPONSE_COEFFICIENTS[i]
        irf_GHG = impulse_response_function(t, GHG)
        delayed_temperature_response = np.exp(-t/TEMPORAL_WEIGHTS[i])
        temperature_response += (
            temporal_weight_linear
            * temperature_response_coefficient
            * (irf_GHG - delayed_temperature_response)
        )

    if GHG.lower() == 'ch4':
        methane_adjustments = _scaled_radiative_efficiency_from_O3_and_H2O()
        return (
            methane_adjustments
            * radiative_efficiency
            * temperature_response
        )
    elif GHG.lower() == 'n2o':
        net_radiative_efficiency = _N2O_radiative_efficiency_after_methane_adjustment()
        return net_radiative_efficiency * temperature_response
    else:
        return radiative_efficiency * temperature_response


def AGTP(t, GHG):
    if GHG.lower() == 'co2':
        return AGTP_CO2(t)
    else:
        return AGTP_non_CO2(t, GHG)


def temperature_response(
        time_horizon,
        emissions,
        GHG,
        step_size,
        annual=False):
    """Computes the global mean temperature change at in response to an emission scenario.

    This is a wrapper around `_dynamic_AGTP` providing a more user-friendly
    name.  `dynamic_AGTP` is computed using a convolution between the emission
    vector and absolute global temperature change potential (`AGTP`):

    .. math:
        {\\Delta}T = \\int{_{0}^{t}emissions_{GHG_i}(s)AGTP_{GHG_i}(t-s)ds}

    When `emissions` is a single value, instead of a temporal emission scenario,
    `temperature_response` returns the same result as `AGTP`.


    Parameters
    ------------------
    time_horizon : int
        The time at which the temperature response is computed.
    emissions : ndarray
        Emissions in kg of a `GHG`.
    GHG : str
    step_size : float or int
        The step size used to generate the time axis.

    Notes
    -------------


    References
        .. [1] Equation 8.1 in https://www.ipcc.ch/site/assets/uploads/2018/02/WG1AR5_Chapter08_FINAL.pdf  # noqa: E501

    """
    if annual:
        _dynamic_AGTP(time_horizon, emissions, GHG, step_size, mode='full')
    else:
        _dynamic_AGTP(time_horizon, emissions, GHG, step_size, mode='full')[0]


def _dynamic_AGTP(time_horizon, emissions, GHG, step_size, mode='valid'):
    """
    Global average surface temperature change at `time_horizon` due to `emissions`.

    `emissions` in kg of `GHG`.

    Parameters
    ------------------
    time_horizon : int
        The time at which the temperature response is computed.
    emissions : ndarray
        Emissions in kg of a `GHG`.
    GHG : str
    step_size : float or int
        The step size used to generate the time axis.
    mode : {'full' or 'valid'}, optional
        'full':
            This provides the full temporal profile of the temperature response
            over the time 0-time_horizon.
        'valid':
            This provides the temperature response from the emission vector
            at `time_horizon`.
    """

    return _dynamic_absolute_climate_metric_template(
        'AGTP',
        time_horizon,
        emissions,
        GHG,
        step_size,
        mode
        )


def GTP(time_horizon,
        emissions,
        GHG,
        step_size=1,
        annual=False):
    """Compute average global temperature change potential of emissions.

    Parameters
    -----------
    time_horizon : int
    emissions : int or ndarray
        If emissions is an int, the emission is assumed to 
            occur at time=0.
    GHG : str {'CO2', 'CH4', 'N2O'}, optional
        Type of GHG emission in `emissions`.
    step_size : float or int
        Step size of `emissions` in years.
    annual : bool
        If `True`, returns annual GWP over the `time_horizon`. If `False`,
            returns the single value at `time_horizon`.

    Notes
    ---------------

    References
    .. [1] IPCC, 2011. https://www.ipcc.ch/site/assets/uploads/2018/02/WG1AR5_Chapter08_FINAL.pdf  # noqa: E501
    """

    return _climate_metric_template(
        'GTP',
        time_horizon,
        emissions,
        GHG,
        step_size,
        annual)


#############################
# Generic templates used to construct GWP and GTP
#############################
def _climate_metric_template(
        method,
        time_horizon,
        emissions,
        GHG,
        step_size=1,
        annual=False):
    """

    Parameters
    -----------
    method : str {'GWP', 'GTP'}
    time_horizon : int
    emissions : int or ndarray
        If emissions is an int, the emission is assumed to
            occur at time=0.
    GHG : str {'CO2', 'CH4', 'N2O'}, optional
        Type of GHG emission in `emissions`.
    step_size : float or int
        Step size of `emissions` in years.
    annual : bool
        If `True`, returns annual GWP over the `time_horizon`. If `False`,
            returns the single value at `time_horizon`.
    """
    _check_method(method)

    if type(emissions) is int:
        empty_array = np.zeros(time_horizon+step_size)
        empty_array[0] = emissions
        emissions = empty_array

    if method == 'GWP':
        physical_metric = _dynamic_AGWP(
            time_horizon, emissions, GHG, step_size, mode='full')
        result = physical_metric / AGWP_CO2(time_horizon)

    elif method == 'GTP':
        physical_metric = _dynamic_AGTP(
            time_horizon, emissions, GHG, step_size, mode='full')
        result = physical_metric / AGTP_CO2(time_horizon)

    if annual:
        return result
    else:
        return result[time_horizon * int(1/step_size)]


def _dynamic_absolute_climate_metric_template(
        method,
        time_horizon,
        emissions,
        GHG,
        step_size,
        mode='valid'
        ):

    _check_method(method)

    t = np.arange(0, time_horizon+step_size, step_size)

    if len(t) < len(emissions):
        raise ValueError("Expected time vector to be longer than the emissions vector")

    if method == 'AGWP':
        absolute_GHG_metric = AGWP(t, GHG)

    elif method == 'AGTP':
        absolute_GHG_metric = AGTP(t, GHG)

    steps = int(time_horizon/step_size)
    return _convolve_metric(steps, emissions, absolute_GHG_metric, mode)


def _check_method(method):
    expected_values = [
        'GWP', 'GTP',
        '_dynamic_AGTP', '_dynamic_AGWP',
        'AGTP', 'AGWP']
    if method not in expected_values:
        raise ValueError(f'str is not in list of expected values: {expected_values}')
